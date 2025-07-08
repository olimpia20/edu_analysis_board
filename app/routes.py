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
import threading
from sklearn.preprocessing import OrdinalEncoder
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score


main = Blueprint('main', __name__)

df_cache = {}
df_cache_lock = threading.Lock()

def get_user_cache_key():
    user_id = current_user.id
    csv_file = session.get('csv_file')
    if not csv_file:
        return None
    return f"{user_id}:{csv_file}"

def get_processed_df():
    cache_key = get_user_cache_key()
    if not cache_key:
        return None
    with df_cache_lock:
        if cache_key in df_cache:
            return df_cache[cache_key]['processed']
        else:
            # Load both original and processed
            df = pd.read_csv(session['csv_file']).convert_dtypes()
            df_cache[cache_key] = {'original': df, 'processed': df.copy()}
            return df_cache[cache_key]['processed']

def set_processed_df(new_df):
    cache_key = get_user_cache_key()
    if not cache_key:
        return
    with df_cache_lock:
        if cache_key in df_cache:
            df_cache[cache_key]['processed'] = new_df

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@main.route('/')
@login_required
def index():
    return redirect(url_for('main.tables'))

@main.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.tables'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)  # add remember
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('main.tables'))
        else:
            flash('Invalid email or password', 'danger')

    return render_template('login.html', form=form)


@main.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.table'))

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
def inject_upload_form_and_files():
    form = UploadCSVForm()
    user_files = []
    if current_user.is_authenticated:
        prefix = f"user_{current_user.id}_"
        if os.path.exists(UPLOAD_FOLDER):
            user_files = [
                f.replace(prefix, '') for f in os.listdir(UPLOAD_FOLDER)
                if f.startswith(prefix)
            ]
    return dict(form=form, uploaded_files=user_files)

@main.route('/tables', methods=['GET', 'POST'])
@login_required
def tables():
    form = UploadCSVForm()

    # List user files
    user_files = []
    prefix = f"user_{current_user.id}_"
    if os.path.exists(UPLOAD_FOLDER):
        user_files = [
            f.replace(prefix, '') for f in os.listdir(UPLOAD_FOLDER)
            if f.startswith(prefix)
        ]

    # Handle file switching via ?csv= param
    csv_param = request.args.get('csv')
    if csv_param and csv_param in user_files:
        session['csv_file'] = os.path.join(UPLOAD_FOLDER, f"user_{current_user.id}_{csv_param}")

    # If no file selected, auto-select the first available
    if not session.get('csv_file') and user_files:
        session['csv_file'] = os.path.join(UPLOAD_FOLDER, f"user_{current_user.id}_{user_files[0]}")

    # Now load the DataFrame for the selected file (if any)
    df = get_processed_df()
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
            # Initialize both original and processed in cache
            cache_key = get_user_cache_key()
            with df_cache_lock:
                df_cache[cache_key] = {'original': df, 'processed': df.copy()}
            session['uploaded_csv_path'] = filepath
            session['uploaded_csv_name'] = os.path.basename(filepath)
        else:
            if 'csv_file' in session:
                df = get_processed_df()
            else:
                return render_template('tables.html', form=form, uploaded=False)
    else:
        if 'csv_file' in session:
            df = get_processed_df()
        else:
            return render_template('tables.html', form=form, uploaded=False)

    # Afișare doar primele sau ultimele N rânduri
    if df is not None:
        if num_head and num_head > 0:
            df = df.head(num_head)
        elif num_tail and num_tail > 0:
            df = df.tail(num_tail)

        # Always calculate statistics if df is available
        stats_df = df.describe().round(6)
        stats_columns = stats_df.columns.tolist()
        stats_data = stats_df.reset_index().values.tolist()

        columns = list(df.columns)
        table_data = df.values.tolist()
        shape = df.shape
        dtypes = df.dtypes.astype(str).to_dict()
    else:
        stats_columns = []
        stats_data = []
        columns = []
        table_data = []
        shape = (0, 0)
        dtypes = {}

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

    # Get DataTables params
    start = int(request.args.get('start', 0))
    length = int(request.args.get('length', 10))
    search_value = request.args.get('search[value]', '')
    num_head = request.args.get('num_head', type=int)
    num_tail = request.args.get('num_tail', type=int)

    df = get_processed_df()
    if df is None:
        return jsonify({
            'draw': int(request.args.get('draw', 1)),
            'recordsTotal': 0,
            'recordsFiltered': 0,
            'data': [],
            'columns': []
        })

    # Filter if searching
    if search_value:
        df = df[df.apply(lambda row: row.astype(str).str.contains(search_value, case=False).any(), axis=1)]

    total_records = df.shape[0]

    # Handle head/tail
    if num_head and num_head > 0:
        df = df.head(num_head)
    elif num_tail and num_tail > 0:
        df = df.tail(num_tail)

    # Paginate
    page_df = df.iloc[start:start+length]
    page_df = page_df.replace({pd.NA: None, np.nan: None})

    def clean_record(row):
        return {k: (None if (v is None or (isinstance(v, float) and math.isnan(v))) else v) for k, v in row.items()}

    data = [clean_record(row) for row in page_df.to_dict(orient="records")]
    columns = list(page_df.columns)

    return jsonify({
        'draw': int(request.args.get('draw', 1)),
        'recordsTotal': total_records,
        'recordsFiltered': df.shape[0],
        'data': data,
        'columns': columns,
        'shape': [df.shape[0], df.shape[1]]
    })

@main.route('/details')
@login_required
def details():
    # Get the current table info
    dtypes = {}
    table_name = None
    if 'csv_file' in session:
        import os
        prefix = f"user_{current_user.id}_"
        table_name = os.path.basename(session['csv_file']).replace(prefix, '')
        df = get_processed_df()
        if df is not None:
            dtypes = df.dtypes.astype(str).to_dict()
    return render_template('details.html', dtypes=dtypes, table_name=table_name)

@main.route('/statistics')
@login_required
def statistics():
    stats_columns = []
    stats_data = []
    table_name = None
    if 'csv_file' in session:
        import os
        prefix = f"user_{current_user.id}_"
        table_name = os.path.basename(session['csv_file']).replace(prefix, '')
        df = get_processed_df()
        if df is not None:
            stats_df = df.describe().round(6)
            stats_columns = stats_df.columns.tolist()
            stats_data = stats_df.reset_index().values.tolist()
    return render_template('statistics.html', stats_columns=stats_columns, stats_data=stats_data, table_name=table_name)

@main.route('/dropna', methods=['POST'])
@login_required
def dropna():
    import pandas as pd
    import os
    if 'csv_file' in session:
        original_path = session['csv_file']
        df = pd.read_csv(original_path).convert_dtypes()
        df_clean = df.dropna()
        # Create a new filename
        base, ext = os.path.splitext(original_path)
        new_path = base + '_no_empty' + ext
        df_clean.to_csv(new_path, index=False)
        # Return the new file name (without user prefix)
        prefix = f"user_{current_user.id}_"
        new_file = os.path.basename(new_path).replace(prefix, '')
        return jsonify(success=True, new_file=new_file)
    return jsonify(success=False)

@main.route('/clean_preview')
@login_required
def clean_preview():
    import pandas as pd
    import os
    if 'csv_file' in session:
        df = pd.read_csv(session['csv_file']).convert_dtypes()
        df_clean = df.dropna()
        columns = df_clean.columns.tolist()
        data = df_clean.head(100).to_dict(orient='records')  # Preview first 100 rows
        # Store the cleaned df in session or cache for later use
        session['clean_preview'] = df_clean.to_json()  # or use a cache
        return render_template('clean_preview.html', columns=columns, data=data)
    return redirect(url_for('main.tables'))

@main.route('/save_cleaned', methods=['POST'])
@login_required
def save_cleaned():
    import pandas as pd
    import os
    import json
    if 'clean_preview' in session and 'csv_file' in session:
        df_clean = pd.read_json(session['clean_preview'])
        original_path = session['csv_file']
        base, ext = os.path.splitext(original_path)
        new_path = base + '_no_empty' + ext
        df_clean.to_csv(new_path, index=False)
        # Remove preview from session
        session.pop('clean_preview', None)
        # Get new file name for sidebar
        prefix = f"user_{current_user.id}_"
        new_file = os.path.basename(new_path).replace(prefix, '')
        return redirect(url_for('main.tables', csv=new_file))
    return redirect(url_for('main.tables'))

@main.route('/ajax_clean_data', methods=['POST'])
@login_required
def ajax_clean_data():
    df = get_processed_df()
    data = request.get_json()
    columns_to_check = data.get('columns', []) if data else []
    if df is not None:
        if columns_to_check:
            # Drop rows only where selected columns are missing
            df_clean = df.dropna(subset=columns_to_check)
        else:
            # Fallback: drop rows where any value is missing
            df_clean = df.dropna()
        # Fill missing values in other columns with 0
        df_clean = df_clean.fillna(0)
        set_processed_df(df_clean)
        return jsonify(success=True)
    return jsonify(success=False, error='No data loaded')

@main.route('/ajax_drop_columns', methods=['POST'])
@login_required
def ajax_drop_columns():
    data = request.get_json()
    columns_to_drop = data.get('columns', [])
    df = get_processed_df()
    if df is not None and columns_to_drop:
        # Only drop columns that exist in the DataFrame
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df_dropped = df.drop(columns=columns_to_drop)
        set_processed_df(df_dropped)
        return jsonify(success=True)
    return jsonify(success=False, error='No data or columns to drop')

@main.route('/ajax_normalize_data', methods=['POST'])
@login_required
def ajax_normalize_data():
    data = request.get_json()
    col_transforms = data.get('transforms', {})
    df = get_processed_df()
    if df is None or not col_transforms:
        return jsonify(success=False, error='No data or transformations provided')

    # Apply transformations
    df_new = df.copy()
    # First, handle one-hot encoding (get_dummies)
    onehot_cols = [col for col, t in col_transforms.items() if t == 'onehot']
    if onehot_cols:
        df_new = pd.get_dummies(df_new, columns=onehot_cols, drop_first=False)
        # Convert any boolean columns to integers (0/1)
        for col in df_new.columns:
            if pd.api.types.is_bool_dtype(df_new[col]):
                df_new[col] = df_new[col].astype(int)
    # Then, handle ordinal encoding
    ordinal_cols = [col for col, t in col_transforms.items() if t == 'ordinal']
    if ordinal_cols:
        encoder = OrdinalEncoder()
        for col in ordinal_cols:
            if col in df_new.columns:
                # OrdinalEncoder expects 2D input
                df_new[col] = encoder.fit_transform(df_new[[col]])
    # Then, handle frequency encoding
    frequency_cols = [col for col, t in col_transforms.items() if t == 'frequency']
    if frequency_cols:
        for col in frequency_cols:
            if col in df_new.columns:
                # Replace each category with its frequency count
                value_counts = df_new[col].value_counts()
                mapping = dict(value_counts)
                df_new[col] = df_new[col].map(lambda x: mapping.get(x, 0))

    # Convert all boolean columns to integers (0/1)
    for col in df_new.columns:
        if pd.api.types.is_bool_dtype(df_new[col]):
            df_new[col] = df_new[col].astype(int)

    set_processed_df(df_new)
    return jsonify(success=True)

@main.route('/get_normalization_recommendations')
@login_required
def get_normalization_recommendations():
    df = get_processed_df()
    if df is None:
        return jsonify({})
    recommendations = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_vals = df[col].dropna().unique()
        n_unique = len(unique_vals)
        if dtype.startswith('float') or dtype.startswith('int'):
            recommendations[col] = 'none'
        elif dtype == 'object' or dtype == 'string':
            lower_vals = [str(v).lower() for v in unique_vals]
            # Example ordinal pattern
            if set(lower_vals).issubset({'low', 'medium', 'high'}):
                recommendations[col] = 'ordinal'
            elif n_unique <= 10:
                recommendations[col] = 'onehot'
            else:
                recommendations[col] = 'frequency'
        else:
            recommendations[col] = 'none'
    return jsonify(recommendations)



def is_normalized(df, features=None):
    cols = features if features is not None else df.columns
    return all(pd.api.types.is_numeric_dtype(df[col]) for col in cols)

@main.route('/train_model', methods=['POST'])
@login_required
def train_model():
    """
    Train AI model with user's data - supports Regression, Classification, and Clustering
    """
    print("=== SERVER DEBUG ===")
    print(f"Request method: {request.method}")
    print(f"Content-Type: {request.content_type}")
    print(f"Raw data: {request.get_data()}")
    
    try:
        data = request.get_json()
        print(f"Parsed JSON data: {data}")
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return jsonify(success=False, error=f'Invalid JSON data: {str(e)}'), 400
    
    algorithm = data.get('algorithm')
    features = data.get('features', [])
    target = data.get('target')
    original_features = data.get('original_features', features)  # Fallback to cleaned if not provided
    original_target = data.get('original_target', target)  # Fallback to cleaned if not provided
    n_clusters = data.get('n_clusters', 3)  # For clustering algorithms
    
    print(f"Training {algorithm} model...")
    print(f"Algorithm: {algorithm}")
    print(f"Original Features: {original_features}")
    print(f"Clean Features: {features}")
    print(f"Original Target: {original_target}")
    print(f"Clean Target: {target}")
    print(f"Number of clusters: {n_clusters}")
    print("===================")

    # Define supported algorithms
    supported_algorithms = {
        'LinearRegression': 'regression',
        'RandomForestRegressor': 'regression',
        'LogisticRegression': 'classification',
        'RandomForestClassifier': 'classification',
        'KMeans': 'clustering'
    }
    
    # Validate algorithm
    if algorithm not in supported_algorithms:
        return jsonify(success=False, error=f'Algorithm {algorithm} is not supported. Supported algorithms: {list(supported_algorithms.keys())}'), 400

    # Get user's processed data
    df = get_processed_df()
    if df is None:
        return jsonify(success=False, error='No data loaded. Please upload and process a CSV file first.'), 400

    print(f"Data shape: {df.shape}")
    print(f"Available columns: {list(df.columns)}")

    # Validate features using original column names
    if not original_features:
        return jsonify(success=False, error='Please select at least one feature column'), 400
    
    missing_features = [f for f in original_features if f not in df.columns]
    if missing_features:
        return jsonify(success=False, error=f'Features not found in dataset: {missing_features}'), 400

    # Validate target for regression and classification (not needed for clustering)
    task_type = supported_algorithms[algorithm]
    if task_type in ['regression', 'classification']:
        if not original_target:
            return jsonify(success=False, error='Please select a target column'), 400
        
        if original_target not in df.columns:
            return jsonify(success=False, error=f'Target column "{original_target}" not found in dataset'), 400

        # Check if target is selected as a feature (common mistake)
        if original_target in original_features:
            return jsonify(success=False, error=f'Target column "{original_target}" cannot be selected as a feature'), 400

    # Prepare training data using original column names
    X = df[original_features].values  # Input features
    
    if task_type in ['regression', 'classification']:
        y = df[original_target].values    # Target variable
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        print(f"Sample X: {X[:3]}")
        print(f"Sample y: {y[:3]}")
        
        # Check for missing values
        if np.isnan(X).any() or np.isnan(y).any():
            return jsonify(success=False, error='Data contains missing values. Please clean your data first.'), 400
    else:  # clustering
        print(f"Training data shape: X={X.shape}")
        print(f"Sample X: {X[:3]}")
        
        # Check for missing values
        if np.isnan(X).any():
            return jsonify(success=False, error='Data contains missing values. Please clean your data first.'), 400

    # Create and train model based on algorithm type
    try:
        if algorithm == 'LinearRegression':
            model = LinearRegression()
            model.fit(X, y)
            coefficients = dict(zip(original_features, model.coef_))
            intercept = model.intercept_
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            model_data = {
                'model': model,
                'task': 'regression',
                'algorithm': algorithm,
                'features': original_features,
                'target': original_target,
                'coefficients': coefficients,
                'intercept': intercept,
                'r2_score': r2,
                'mse': mse
            }
            success_message = f'Linear Regression model trained successfully! R² = {r2:.4f}, MSE = {mse:.4f}'
            if coefficients:
                coef_text = ', '.join([f'{f}: {c:.4f}' for f, c in coefficients.items()])
                success_message += f' Model equation: {original_target} = {coef_text} + {intercept:.4f}'
        elif algorithm == 'RandomForestRegressor':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            feature_importance = dict(zip(original_features, model.feature_importances_))
            model_data = {
                'model': model,
                'task': 'regression',
                'algorithm': algorithm,
                'features': original_features,
                'target': original_target,
                'feature_importance': feature_importance,
                'r2_score': r2,
                'mse': mse
            }
            success_message = f'Random Forest Regressor trained successfully! R² = {r2:.4f}, MSE = {mse:.4f}'
        elif algorithm == 'LogisticRegression':
            le = None
            if y.dtype == 'object':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X, y_encoded)
            y_pred = model.predict(X)
            accuracy = accuracy_score(y_encoded, y_pred)
            model_data = {
                'model': model,
                'task': 'classification',
                'algorithm': algorithm,
                'features': original_features,
                'target': original_target,
                'accuracy': accuracy,
                'label_encoder': le
            }
            success_message = f'Logistic Regression model trained successfully! Accuracy = {accuracy:.4f}'
        elif algorithm == 'RandomForestClassifier':
            le = None
            if y.dtype == 'object':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y_encoded)
            y_pred = model.predict(X)
            accuracy = accuracy_score(y_encoded, y_pred)
            feature_importance = dict(zip(original_features, model.feature_importances_))
            model_data = {
                'model': model,
                'task': 'classification',
                'algorithm': algorithm,
                'features': original_features,
                'target': original_target,
                'accuracy': accuracy,
                'feature_importance': feature_importance,
                'label_encoder': le
            }
            success_message = f'Random Forest Classifier trained successfully! Accuracy = {accuracy:.4f}'
        elif algorithm == 'KMeans':
            model = KMeans(n_clusters=n_clusters, random_state=42)
            model.fit(X)
            cluster_labels = model.labels_
            model_data = {
                'model': model,
                'task': 'clustering',
                'algorithm': algorithm,
                'features': original_features,
                'n_clusters': n_clusters,
                'cluster_labels': cluster_labels,
                'inertia': model.inertia_
            }
            success_message = f'KMeans clustering completed! {n_clusters} clusters created. Inertia = {model.inertia_:.4f}'
        print(f"Model trained successfully!")
    except Exception as e:
        print(f"Model training error: {str(e)}")
        return jsonify(success=False, error=f'Model training failed: {str(e)}'), 400

    # Save model to user_models directory
    user_id = current_user.id
    model_dir = os.path.join('app', 'user_models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'user_{user_id}_model.pkl')
    
    joblib.dump(model_data, model_path)
    
    return jsonify(success=True, message=success_message)

@main.route('/predict', methods=['POST'])
@login_required
def predict():
    """
    Make predictions using trained model - supports Regression, Classification, and Clustering
    """
    import os
    import joblib
    import json
    import numpy as np

    user_id = current_user.id
    model_path = os.path.join('app', 'user_models', f'user_{user_id}_model.pkl')
    
    # Check if model exists
    if not os.path.exists(model_path):
        return jsonify(success=False, error='No trained model found. Please train a model first.')

    try:
        # Load model and metadata
        model_data = joblib.load(model_path)
        model = model_data['model']
        features = model_data['features']
        task = model_data['task']
        algorithm = model_data['algorithm']
        print(f"Loaded {algorithm} model for task: {task}")
        print(f"Features: {features}")
    except Exception as e:
        return jsonify(success=False, error=f'Error loading model: {str(e)}')

    # Get input features from request
    input_data = request.get_json()
    if not input_data:
        return jsonify(success=False, error='No input data provided. Please enter values for prediction.')

    # Validate input format
    if not isinstance(input_data, dict):
        return jsonify(success=False, error='Invalid input format. Expected dictionary of feature values.')

    # Prepare input data
    try:
        X_input = []
        missing_features = []
        for feature in features:
            if feature in input_data:
                value = input_data[feature]
                try:
                    X_input.append(float(value))
                except (ValueError, TypeError):
                    return jsonify(success=False, error=f'Invalid value for {feature}: "{value}". Please enter a valid number.')
            else:
                missing_features.append(feature)
        if missing_features:
            return jsonify(success=False, error=f'Missing values for features: {missing_features}')
        X_input = np.array(X_input).reshape(1, -1)
        print(f"Input data: {X_input}")
    except Exception as e:
        return jsonify(success=False, error=f'Error preparing input data: {str(e)}')

    # Make prediction based on task type
    try:
        if task == 'regression':
            prediction = model.predict(X_input)
            predicted_value = prediction[0]
            # Convert to Python type
            if hasattr(predicted_value, 'item'):
                predicted_value = predicted_value.item()
            result = {
                'prediction': [predicted_value],
                'target': model_data.get('target', 'target'),
                'features': features,
                'input_values': input_data,
                'task': 'regression',
                'algorithm': algorithm
            }
            
            # Add model equation for linear regression
            if algorithm == 'LinearRegression':
                coefficients = model_data.get('coefficients', {})
                intercept = model_data.get('intercept', 0)
                if coefficients:
                    result['model_equation'] = f"{result['target']} = " + " + ".join([f"{coef:.4f}×{feat}" for feat, coef in coefficients.items()]) + f" + {intercept:.4f}"
            
        elif task == 'classification':
            prediction = model.predict(X_input)
            predicted_class = prediction[0]
            if hasattr(predicted_class, 'item'):
                predicted_class = predicted_class.item()
            
            # Decode class label if label encoder exists
            label_encoder = model_data.get('label_encoder')
            if label_encoder:
                predicted_class = label_encoder.inverse_transform([predicted_class])[0]
            
            # Get prediction probabilities if available
            try:
                probabilities = model.predict_proba(X_input)[0]
                if label_encoder:
                    class_names = label_encoder.classes_
                else:
                    class_names = [str(i) for i in range(len(probabilities))]
                prob_dict = dict(zip(class_names, probabilities))
            except:
                prob_dict = {}
            
            result = {
                'prediction': [predicted_class],
                'probabilities': prob_dict,
                'target': model_data.get('target', 'target'),
                'features': features,
                'input_values': input_data,
                'task': 'classification',
                'algorithm': algorithm
            }
            
        elif task == 'clustering':
            cluster = model.predict(X_input)[0]
            
            result = {
                'prediction': [int(cluster)],
                'cluster': int(cluster),
                'features': features,
                'input_values': input_data,
                'task': 'clustering',
                'algorithm': algorithm,
                'n_clusters': model_data.get('n_clusters', 3)
            }
        
        print(f"Prediction made: {result['prediction']}")
        
    except Exception as e:
        return jsonify(success=False, error=f'Prediction error: {str(e)}')

    # Save prediction to log
    try:
        log_dir = os.path.join('app', 'user_models')
        log_path = os.path.join(log_dir, f'user_{user_id}_predictions.json')
        log_entry = {
            'timestamp': str(pd.Timestamp.now()),
            'input': input_data,
            'prediction': result['prediction'][0],
            'task': task,
            'algorithm': algorithm
        }
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                log = json.load(f)
        else:
            log = []
        log.append(log_entry)
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Could not save prediction log: {e}")
        # Don't fail prediction if logging fails

    return jsonify(success=True, **result)



# Global error handlers to ensure JSON responses
@main.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad Request', 'details': str(e)}), 400

@main.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not Found', 'details': str(e)}), 404

@main.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Server Error', 'details': str(e)}), 500

@main.errorhandler(Exception)
def handle_exception(e):
    import traceback
    print("Unhandled exception:", traceback.format_exc())
    return jsonify({'error': 'Unhandled Exception', 'details': str(e)}), 500

@main.route('/train')
@login_required
def train_page():
    df = get_processed_df()
    columns = list(df.columns) if df is not None else []
    normalized = is_normalized(df) if df is not None else False

    table_name = None
    if 'csv_file' in session:
        import os
        prefix = f"user_{current_user.id}_"
        table_name = os.path.basename(session['csv_file']).replace(prefix, '')
    return render_template('train.html', columns=columns, is_normalized=normalized, table_name=table_name)

@main.route('/is_normalized_status')
@login_required
def is_normalized_status():
    df = get_processed_df()
    normalized = is_normalized(df) if df is not None else False
    return jsonify({'is_normalized': normalized})

@main.route('/get_dtypes')
def get_dtypes():
    # Adjust this to your actual logic for loading the user's CSV
    csv_path = session.get('uploaded_csv_path')  # Or however you store it
    if not csv_path or not os.path.exists(csv_path):
        return jsonify({'success': False, 'error': 'CSV not found'})
    df = pd.read_csv(csv_path)
    dtypes = df.dtypes.apply(str).to_dict()
    return jsonify({'success': True, 'dtypes': dtypes})

@main.route('/normalize', methods=['POST'])
def normalize():
    csv_path = session.get('uploaded_csv_path')
    if not csv_path or not os.path.exists(csv_path):
        return "No CSV uploaded", 400
    df = pd.read_csv(csv_path)
    # ... procesezi df ...
    processed_path = os.path.join('instance', 'processed.csv')
    df.to_csv(processed_path, index=False)
    session['uploaded_csv_path'] = processed_path
    session['uploaded_csv_name'] = os.path.basename(processed_path)
    return redirect(url_for('train'))

@main.context_processor
def inject_csv_status():
    csv_name = session.get('uploaded_csv_name')
    return dict(csv_name=csv_name)

@main.route('/prediction_view')
@login_required
def prediction_view():
    import os, json
    user_id = current_user.id
    log_path = os.path.join('app', 'user_models', f'user_{user_id}_predictions.json')
    predictions = []
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
        except Exception:
            predictions = []
    return render_template('prediction_view.html', predictions=predictions)




