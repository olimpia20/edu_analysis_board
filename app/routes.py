from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_login import login_user, logout_user, login_required, current_user
from app.forms import LoginForm, RegistrationForm, ForgotPasswordForm, UploadCSVForm
from app.models import User
from app import db, login_manager
import os
import pandas as pd
from werkzeug.utils import secure_filename


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

    # Dacă s-a apăsat butonul "Show Stats"
    if request.method == 'POST' and 'show_stats' in request.form:
        if df is None:
            return render_template('tables.html', form=form, error="Please upload a CSV file first.")
        stats_df = df.describe(include='all').reset_index()
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
        uploaded_files = user_files
    )


