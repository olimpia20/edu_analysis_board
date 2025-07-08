# Educational Analysis Board: Project Report

## Project Idea

The Educational Analysis Board is a web application that allows users (e.g., teachers, students) to upload, analyze, clean, and model educational data using AI. Users can perform regression, classification, and clustering tasks on their data, all through a user-friendly interface.

---

## Tools and Libraries Used

- **Flask**: Web framework for Python
- **Flask-Login**: User authentication/session management
- **Flask-SQLAlchemy**: Database ORM
- **Flask-WTF & WTForms**: Secure forms and CSRF protection
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **email_validator**: Email validation for registration
- **Bootstrap, jQuery, DataTables**: Frontend styling and interactivity

---

## Data Storage: Where and How

- **User Data**: Stored in a SQLite database (`instance/edu.db`) using SQLAlchemy models.
- **Uploaded CSV Files**: Saved in `app/uploads/` with filenames prefixed by user ID for privacy.
- **Trained AI Models**: Saved as `.joblib` files in `app/user_models/`, named by user and model type.
- **Session Data**: Managed by Flask sessions (e.g., which file is selected).

**User Model Example:**
```python
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    # ... password methods ...
```

---

## Step-by-Step Functionality

### 1. **User Registration & Login**
- Users register with name, email, and password.
- Passwords are hashed for security.
- Login uses email and password; session is managed by Flask-Login.
- CSRF protection is enabled for all forms.

**Login Route Example:**
```python
@main.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.tables'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)
            return redirect(url_for('main.tables'))
        else:
            flash('Invalid email or password', 'danger')
    return render_template('login.html', form=form)
```

### 2. **Uploading Data**
- Users upload CSV files via a modal form.
- Files are saved with a user-specific prefix.
- Uploaded files are listed for each user.

**Upload Form Example:**
```python
class UploadCSVForm(FlaskForm):
    file = FileField('Upload CSV', validators=[FileAllowed(['csv'], 'CSV files only!')])
    submit = SubmitField('Upload')
```

### 3. **Viewing and Cleaning Data**
- Uploaded tables are displayed using DataTables (sortable, searchable).
- Users can preview, drop columns, and clean missing data (drop NA rows).
- Data normalization and encoding are available for ML readiness.

### 4. **Statistics and Details**
- Users can view summary statistics (mean, std, min, max, etc.) for each column.
- Details page shows column names and data types.

### 5. **Training AI Models**
- Users select task type (regression, classification, clustering), algorithm, features, and (if needed) target column.
- Supported algorithms:
  - Regression: Linear Regression, Random Forest Regressor
  - Classification: Logistic Regression, Random Forest Classifier
  - Clustering: KMeans
- Model is trained on the selected data and saved for the user.

**Training Route Example:**
```python
@main.route('/train_model', methods=['POST'])
@login_required
def train_model():
    # Receives JSON with task_type, algorithm, features, target
    # Loads user data, encodes as needed, trains model, saves with joblib
    # Returns training metrics (accuracy, MSE, etc.)
```

### 6. **Prediction**
- After training, users can input new feature values to get predictions:
  - Regression: Predicts a number
  - Classification: Predicts a class label (and probabilities)
  - Clustering: Predicts a cluster number

**Prediction Route Example:**
```python
@main.route('/predict', methods=['POST'])
@login_required
def predict():
    # Receives JSON with model type and input features
    # Loads trained model, encodes input, returns prediction
```

### 7. **Logout and Security**
- Users can log out, which clears their session.
- All sensitive actions require login.
- CSRF protection is enforced for all POST requests.

---

## Concept Explanations

- **Regression**: Predicts continuous values (e.g., score, price).
- **Classification**: Predicts categories (e.g., pass/fail, A/B/C).
- **Clustering**: Groups data into clusters (no target column needed).
- **Normalization/Encoding**: Prepares data for ML by scaling/encoding values.
- **CSRF Token**: Prevents cross-site request forgery; included in all forms and AJAX requests.

---

## User Interface Pages

- **Login/Register/Forgot Password**: Secure user authentication.
- **Tables**: Upload, view, and clean data.
- **Details**: See column names and types.
- **Statistics**: View summary statistics.
- **Train**: Select task, algorithm, features, and train models.
- **Prediction**: Input new data and get predictions.

---

## Example: Login Form (HTML)
```html
<form method="POST" class="user" action="">
    {{ form.hidden_tag() }}
    <div class="form-group">
        {{ form.email(class="form-control form-control-user", placeholder="Enter Email...") }}
    </div>
    <div class="form-group">
        {{ form.password(class="form-control form-control-user", placeholder="Password") }}
    </div>
    <button type="submit" class="btn btn-primary btn-user btn-block">Login</button>
</form>
```

---

## Example: Model Training (Frontend JS)
```js
// When user clicks Train, send AJAX POST with selected options
fetch('/train_model', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': window.csrf_token
    },
    body: JSON.stringify({
        task_type: selectedTask,
        algorithm: selectedAlgorithm,
        features: selectedFeatures,
        target: selectedTarget
    })
})
.then(response => response.json())
.then(data => {
    // Show training results
});
```

---

## Security and Best Practices
- Passwords are hashed (never stored in plain text)
- CSRF protection is always enabled
- User data/files/models are separated by user ID
- Only logged-in users can access data and AI features

---

## Summary
This project demonstrates a full workflow for educational data analysis and AI modeling, from secure login to data upload, cleaning, analysis, model training, and prediction—all in a modern, user-friendly web app.

If you need more code examples or explanations for any part, just ask! 