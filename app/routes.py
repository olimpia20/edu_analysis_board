from flask import Blueprint, render_template, redirect, url_for, flash, request, session, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from app.forms import LoginForm, RegistrationForm, ForgotPasswordForm, UploadCSVForm
from app.models import User
from app import db, login_manager
import os
import pandas as pd
from werkzeug.utils import secure_filename
import numpy as np
import math


main = Blueprint('main', __name__)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@main.route('/')
@login_required
def index():
    return render_template('index.html')

@main.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)  # add remember
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('main.index'))
        else:
            flash('Invalid email or password', 'danger')

    return render_template('login.html', form=form)


@main.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form = RegistrationForm()
    if form.validate_on_submit():
        if User.query.filter_by(email=form.email.data).first():
            flash('Email already registered', 'danger')
        else:
            user = User()
            user.first_name = form.first_name.data
            user.last_name = form.last_name.data
            user.email = form.email.data
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful. Please login.', 'success')
            return redirect(url_for('main.login'))

    return render_template('register.html', form=form)

@main.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('main.login'))

@main.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        email = form.email.data
        # TODO: verific dacă emailul există în DB și trimit email cu link de reset
        flash('If this email exists in our system, you will receive a reset link.', 'info')
        return redirect(url_for('main.login'))
    return render_template('forgot-password.html', form=form)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'app', 'uploads')

@main.app_context_processor
def inject_upload_form():
    return dict(form=UploadCSVForm())

@main.route('/tables', methods=['GET', 'POST'])
@login_required
def tables():
    form = UploadCSVForm()
    df = None
    stats = []
    num_head = request.form.get('num_head', type=int)
    num_tail = request.form.get('num_tail', type=int)
    page_length = request.form.get('num_head') or request.form.get('num_tail') or 10

    if form.validate_on_submit():
        file = form.file.data
        if file:
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            filename = f"user_{current_user.id}_{secure_filename(file.filename)}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            session['csv_file'] = filepath
            df = pd.read_csv(filepath).convert_dtypes()
        else:
            # formular trimis dar fără fișier nou
            if 'csv_file' in session:
                df = pd.read_csv(session['csv_file']).convert_dtypes()
            else:
                return render_template('tables.html', form=form, uploaded=False)
    else:
        # GET sau POST invalid
        if 'csv_file' in session:
            df = pd.read_csv(session['csv_file']).convert_dtypes()
        else:
            return render_template('tables.html', form=form, uploaded=False)

    # Afișare doar primele sau ultimele N rânduri
    if num_head and num_head > 0:
        df = df.head(num_head)
    elif num_tail and num_tail > 0:
        df = df.tail(num_tail)

    # Always calculate statistics if df is available
    if df is not None:
        stats_df = df.describe().round(6).reset_index()
        stats_columns = stats_df.columns.tolist()
        stats_data = stats_df.values.tolist()
    else:
        stats_columns = []
        stats_data = []

    columns = list(df.columns)
    table_data = df.values.tolist()
    shape = df.shape
    dtypes = df.dtypes.astype(str).to_dict()

    # Listează fișierele încărcate pentru utilizatorul curent
    user_files = []
    prefix = f"user_{current_user.id}_"
    if os.path.exists(UPLOAD_FOLDER):
        user_files = [
            f.replace(prefix, '') for f in os.listdir(UPLOAD_FOLDER)
            if f.startswith(prefix)
        ]

    # Extract current table name from session['csv_file'] if available
    table_name = None
    if 'csv_file' in session:
        table_name = os.path.basename(session['csv_file']).replace(prefix, '')

    return render_template(
        'tables.html',
        form=form,
        columns=columns,
        table_data=table_data,
        uploaded=True,
        shape=shape,
        dtypes=dtypes,
        stats_columns=stats_columns,
        stats_data=stats_data,
        uploaded_files = user_files,
        table_name=table_name,
        page_length=page_length
    )

@main.route('/data_table_ajax')
@login_required
def data_table_ajax():
    import numpy as np
    import pandas as pd

    # Get parameters from DataTables
    start = int(request.args.get('start', 0))
    length = int(request.args.get('length', 10))
    search_value = request.args.get('search[value]', '')

    # Load the CSV (or keep it in memory for performance)
    df = pd.read_csv(session['csv_file']).convert_dtypes()

    # Filter if searching
    if search_value:
        df = df[df.apply(lambda row: row.astype(str).str.contains(search_value, case=False).any(), axis=1)]

    total_records = df.shape[0]

    if df.empty:
        # Return an empty data/columns response
        return jsonify({
            'draw': int(request.args.get('draw', 1)),
            'recordsTotal': 0,
            'recordsFiltered': 0,
            'data': [],
            'columns': []
        })

    # Paginate and robustly convert all missing values to None
    page_df = df.iloc[start:start+length]
    page_df = page_df.replace({pd.NA: None, np.nan: None})

    # This will ensure every value is either a Python type or None
    records = page_df.to_dict(orient="records")
    # Convert all NaN/NA in the dicts to None
    def clean_record(row):
        return {k: (None if (v is None or (isinstance(v, float) and math.isnan(v))) else v) for k, v in row.items()}

    data = [clean_record(row) for row in records]

    columns = list(page_df.columns)


    return jsonify({
        'draw': int(request.args.get('draw', 1)),
        'recordsTotal': total_records,
        'recordsFiltered': total_records,
        'data': data,
        'columns': columns
    })


