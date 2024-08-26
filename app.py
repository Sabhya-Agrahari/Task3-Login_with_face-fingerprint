import os
from flask import Flask, render_template, request, redirect, url_for, session, flash,jsonify
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, SubmitField, FileField
from wtforms.validators import DataRequired, Email, EqualTo
import mysql.connector
import bcrypt
from werkzeug.utils import secure_filename
import cv2
import dlib
from twilio.rest import Client
import random
import shutil 
import base64
from flask import current_app as app
import numpy as np
from flask_cors import CORS
import face_recognition
import hashlib
import time
import tensorflow as tf
import h5py
from tensorflow.keras.models import load_model # type: ignore
from io import BytesIO

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("TensorFlow version:", tf.__version__)




app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store face images
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'} 
CORS(app)

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


cascade_path = r'C:/Users/sabhya/Desktop/NullClass/haarcascades_frontalface_default.xml'

# Initialize face recognition models
detector = dlib.get_frontal_face_detector()
shape_predictor_path = 'C:/Users/sabhya/Downloads/shape_predictor_68_face_landmarks.dat'
shape_predictor = dlib.shape_predictor(shape_predictor_path)

face_recognizer = dlib.face_recognition_model_v1('C:/Users/sabhya/Downloads/dlib_face_recognition_resnet_model_v1.dat')

saved_model_path ='C:\\Users\\sabhya\\Desktop\\NullClass\\dataset_FVC2000_DB4_B\\dataset'



# MySQL connection configuration
db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='root',
    database='amazon1'
)


# Initialize Twilio client
client = Client("ACe24596da06700c639e5df4f9b0fb4fbb", "dc7401079c8e5a1f0e78003c0aa1b623")
# Route for sending OTP (Twilio)
@app.route('/send_otp', methods=['POST'])
def send_otp_route():
    if request.method == 'POST':
        phone_number = request.form.get('phone_number')
        
        success, message = send_otp(phone_number)
        
        if success:
            flash('OTP sent successfully!', 'success')
            return redirect(url_for('verify_otp'))
        else:
            flash(f'Failed to send OTP: {message}', 'error')
            return redirect(url_for('profile'))
    
    flash('Invalid request method', 'error')
    return redirect(url_for('profile'))

# Function to send OTP via Twilio
def send_otp(phone_number):
    # Generate OTP
    otp = ''.join(random.choices('0123456789', k=6))

    # Save OTP in session or any other storage
    session['expected_otp'] = otp

    # Send OTP via SMS using Twilio
    try:
        message = client.messages.create(
            body=f'Your OTP is: {otp}',
            from_="+16082004392",  # Your Twilio phone number
            to="+919455956183"  # Recipient's phone number
        )
        return True, f'SMS sent successfully! Message ID: {message.sid}'
    except Exception as e:
        flash(f'Failed to send SMS: {str(e)}', 'error')
        return False, f'Failed to send SMS: {str(e)}'

# Route for OTP verification (Twilio)
@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'POST':
        entered_otp = request.form['otp']
        expected_otp = session.get('expected_otp')  # Retrieve expected OTP from session
        cursor = db.cursor()
        if entered_otp == expected_otp:
            # Update user record to mark phone number as verified
            
            
            cursor.execute("UPDATE users SET phone_number=%s, otp_secret=NULL WHERE email=%s", (session.get('phone_number'), session.get('user_email')))
            db.commit()
            cursor.close()

            flash('OTP verified successfully!', 'success')
            return redirect(url_for('profile'))
        else:
            flash('OTP verification failed. Please try again.', 'error')
            return redirect(url_for('verify_otp'))
    
    return render_template('verify_otp.html')

# Function to fetch user by email from MySQL
def fetch_user_by_email(email):
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    return user

# Function to fetch user by ID from MySQL
def fetch_user_by_id(user_id):
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    cursor.close()
    return user

# Login form
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# Registration form
class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    
    submit = SubmitField('Register')

# Profile form
class ProfileForm(FlaskForm):
    phone_number = StringField('Phone Number', validators=[DataRequired()])
    submit = SubmitField('Update Profile')

#Login Face form
class LoginFaceForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    photo = FileField('Upload Photo',validators=[DataRequired()])  # Added photo upload field
    fingerprint = FileField('Upload Fingerprint', validators=[DataRequired(), FileAllowed(['jpg', 'jpeg', 'png'], 'Images only!')])  # Added fingerprint field
    submit = SubmitField('LoginFace')


# Function to save face image to uploads folder
def save_face_image(photo, user_id):
    # Ensure the directory exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    try:
        # Secure the filename
        filename = f"user_{user_id}_{int(time.time())}.jpg"
        print(f"Secure filename: {filename}")

        # Save the photo to the upload folder
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        photo.save(photo_path)
        print(f"Photo saved to: {photo_path}")

        return filename  # Return the saved filename for further processing
    except Exception as e:

        print(f"Error saving file: {str(e)}")
        return None



# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        user = fetch_user_by_email(email)

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('profile'))
        else:
            flash('Login failed. Please check your email and password.', 'error')

    return render_template('login.html', form=form)

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data
       

        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Insert user into MySQL database
        cursor = db.cursor()
        try:
            cursor.execute("INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                           (username, email, hashed_password ))
            db.commit()
            cursor.close()
            flash('Registration successful. Please login.', 'success')
            return redirect(url_for('login'))
        except mysql.connector.Error as err:
            flash(f'MySQL Error: {err}', 'error')

    return render_template('register.html', form=form)

def enable_face_recognition():
    if 'user_id' not in session:
        flash("You must be logged in to access this page.", "error")
        return redirect(url_for('login'))

    # Placeholder: Capture face image from webcam using OpenCV
    filename = capture_face()

    # Check if filename is successfully returned from capture_face()
    if filename:
        # Example storing in file system (you can modify to store in database or cloud storage)
        save_face_image(filename, session['user_id'])
        store_face_descriptor(session['user_id'], filename)

        # Update database to indicate face recognition is enabled
        cursor = db.cursor()
        try:
            cursor.execute("UPDATE users SET face_enabled = TRUE, face_image_path = %s WHERE id = %s", (filename, session['user_id']))
            db.commit()
            cursor.close()
            flash("Face recognition enabled successfully!", "success")
            return redirect(url_for('profile'))
        except mysql.connector.Error as err:
            flash(f"MySQL Error: {err}", "error")
            return redirect(url_for('profile'))
    else:
        flash("Failed to capture face image. Please try again.", "error")
        return redirect(url_for('profile'))


def disable_face_recognition():
    if 'user_id' not in session:
        flash("You must be logged in to access this page.", "error")
        return redirect(url_for('login'))

    cursor = db.cursor()
                
    try:
        cursor.execute("UPDATE users SET face_enabled = FALSE, face_image_path = NULL WHERE id = %s", (session['user_id'],))
        db.commit()
        cursor.close()
        flash("Face recognition disabled successfully!", "info")
    except mysql.connector.Error as err:
        flash(f"MySQL Error: {err}", "error")

    return redirect(url_for('profile'))


def capture_face():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        flash("Error: Could not open webcam.", 'error')
        return None

    # Initialize the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            flash("Error: Failed to capture frame.", 'error')
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) where the face is detected
            face_region = gray[y:y+h, x:x+w]

            # Generate a filename for the captured image (you can use UUID or timestamp)
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'face_image.jpg')

            # Save the captured frame as an image file
            cv2.imwrite(filename, frame)

            # Display a rectangle around the detected face (optional)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the frame with rectangles around faces (optional)
        cv2.imshow('Face Detection', frame)

        # Break the loop on pressing 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return filename  # Return the filename after successful capture

    cap.release()
    cv2.destroyAllWindows()
    return None  # Return None if no filename is captured



# Profile route with face recognition enablement
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        flash('You must be logged in to access this page.', 'error')
        return redirect(url_for('login'))

    user = fetch_user_by_id(session['user_id'])
    form = ProfileForm()

    if form.validate_on_submit():
        phone_number = form.phone_number.data

        # Update phone number in the database
       
        cursor = db.cursor()
        try:
            cursor.execute("UPDATE users SET phone_number = %s WHERE id = %s", (phone_number, session['user_id']))
            db.commit()
            cursor.close()
            

        except mysql.connector.Error as err:
            flash(f'MySQL Error: {err}', 'error')

            flash('Profile updated successfully!', 'success')             
            return redirect(url_for('profile'))  # Redirect to refresh page after update
        except mysql.connector.Error as err:
            flash(f'MySQL Error: {err}', 'error')
        

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'enable_face_recognition':
            if enable_face_recognition():
                
                if 'user_id' not in session:
                    flash("You must be logged in to access this page.", "error")
                    return redirect(url_for('login'))
                
                filename = capture_face()
                print("sabhya 2 <>"+filename)

                if filename:
                # Example storing in file system (you can modify to store in database or cloud storage)
                    save_face_image(filename, session['user_id'])
                    print("sabhya 3 <>"+filename)
                    store_face_descriptor(session['user_id'], filename)

                # Update database to indicate face recognition is enabled
                    cursor = db.cursor()
                    
                    try:
                        cursor.execute("UPDATE users SET face_enabled = TRUE, face_image_path = %s WHERE id = %s", (filename, session['user_id']))
                        db.commit()
                        cursor.close()
                        send_otp_route()

                        # flash("Face recognition enabled successfully!", "success")
                        return redirect(url_for('verify_otp'))
                    except mysql.connector.Error as err:
                        flash(f"MySQL Error: {err}", "error")
                        # flash("Face recognition enabled successfully!", "success")
                        # return redirect(url_for('verify_otp'))
        
        elif action == 'disable_face_recognition':
            if disable_face_recognition():
            # Implement logic to disable face recognition
                
                if 'user_id' not in session:
                    flash("You must be logged in to access this page.", "error")
                    return redirect(url_for('login'))

                cursor = db.cursor()
                
                try:
                    cursor.execute("UPDATE users SET face_enabled = FALSE, face_image_path = NULL WHERE id = %s", (session['user_id'],))
                    db.commit()
                  
                    cursor.close()
                    send_otp_route()
                    flash("Face recognition disabled successfully!", "info")
                    return redirect(url_for('verify_otp'))
                except mysql.connector.Error as err:
                    flash(f"MySQL Error: {err}", "error")

                return redirect(url_for('profile'))
            
            if action == 'enable_fingerprint':
                print("Sabhya123")
                fingerprint_data = request.form.get('fingerprint_data')  # Ensure you have this data in the form
                print("Sabhya456")
                if fingerprint_data:
                    return enable_fingerprint(fingerprint_data)
                else:
                    flash("No fingerprint data provided.", "error")
                    return redirect(url_for('profile'))

            elif action == 'disable_fingerprint':
                cursor = db.cursor()
                try:
                    cursor.execute("UPDATE users SET finger_enabled = FALSE, fingerprint_data = NULL WHERE id = %s", (session['user_id'],))
                    db.commit()
                    cursor.close()
                    flash("Fingerprint recognition disabled successfully!", "info")
                except mysql.connector.Error as err:
                    flash(f"MySQL Error: {err}", "error")

    return render_template('profile.html', user=user, form=form)





@app.route('/login_with_face', methods=['GET', 'POST'])
def login_with_face():
    form = LoginFaceForm()
    if form.validate_on_submit():
        email = form.email.data
        
        # Handle the photo (Base64 encoded)
        photo_data = form.photo.data
        try:
            if not photo_data.startswith('data:image'):
                return jsonify({"success": False, "message": "Invalid file format for photo"}), 400

            # Extract and decode the Base64 data
            base64_header, base64_string = photo_data.split(",", 1)
            photo_content = base64.b64decode(base64_string)
            npimg = np.frombuffer(photo_content, np.uint8)
            captured_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if captured_image is None:
                return jsonify({"success": False, "message": "Failed to decode image"}), 400

        except Exception as e:
            return jsonify({"success": False, "message": f"Error processing photo file: {e}"}), 400

        # Handle the fingerprint (file upload)
        fingerprint_file = request.files.get('fingerprint')
        if not fingerprint_file:
            return jsonify({"success": False, "message": "No fingerprint file provided"}), 400

        if not allowed_file(fingerprint_file.filename):
            return jsonify({"success": False, "message": "Invalid file format for fingerprint"}), 400

        try:
            fingerprint_content = fingerprint_file.read()
            npfinger = np.frombuffer(fingerprint_content, np.uint8)
            captured_fingerprint = cv2.imdecode(npfinger, cv2.IMREAD_GRAYSCALE)
            if captured_fingerprint is None:
                return jsonify({"success": False, "message": "Failed to decode fingerprint image"}), 400

        except Exception as e:
            return jsonify({"success": False, "message": f"Error processing fingerprint file: {e}"}), 400

        # Fetch user by email
        user = fetch_user_by_email(email)
        if not user:
            flash('User not found', 'error')
            return jsonify({'success': False, 'message': 'User not found'}), 404

        # Load the fingerprint model (if not already loaded)
        fingerprint_model = load_fingerprint_model(file_path)

        # Get database connection
        cursor=db.cursor()

        # Recognize face
        face_success, recognized_user_id = recognize_face(captured_image)

        # Recognize fingerprint
        fingerprint_success = recognize_fingerprint(captured_fingerprint, user['id'], fingerprint_model, db)

        if face_success and recognized_user_id == user['id'] and fingerprint_success:
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful with face and fingerprint!', 'success')
            return redirect(url_for('profile'))
        else:
            flash('Face or fingerprint not recognized', 'error')
            return jsonify({'success': False, 'message': 'Face or fingerprint not recognized'}), 401

    return render_template('login_with_face.html', form=form)

# Function to recognize face in captured image
def recognize_face(image):
    # Detect faces
    detected_faces = detector(image, 1)
    
    if len(detected_faces) == 0:
        return False, None
    
    # Process the first detected face (if multiple faces are detected, adjust accordingly)
    shape = shape_predictor(image, detected_faces[0])
    
    # Compute the face descriptor
    face_descriptor = face_recognizer.compute_face_descriptor(image, shape)
    
    # Compare with known face descriptors from the database
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, face_descriptor FROM users WHERE face_descriptor IS NOT NULL")
    users = cursor.fetchall()
    cursor.close()

    for user in users:
        known_face_descriptor = np.frombuffer(user['face_descriptor'], dtype=np.float64)
        distance = np.linalg.norm(np.array(face_descriptor) - known_face_descriptor)
        
        if distance < 0.6:  # Threshold for face recognition
            recognized_user_id = user['id']
            return True, recognized_user_id
    
    return False, None
    
    

def store_face_descriptor(user_id, face_image_path):
    image = cv2.imread(face_image_path)
    detected_faces = detector(image, 1)
    
    if len(detected_faces) == 0:
        return False
    
    shape = shape_predictor(image, detected_faces[0])
    face_descriptor = face_recognizer.compute_face_descriptor(image, shape)
    
    # Serialize face descriptor to store in database
    serialized_descriptor = np.array(face_descriptor).tobytes()
    
    cursor = db.cursor()
    try:
        cursor.execute("UPDATE users SET face_descriptor = %s WHERE id = %s", (serialized_descriptor, user_id))
        db.commit()
    except mysql.connector.Error as err:
        print(f"Error storing face descriptor: {err}")
    finally:
        cursor.close()

    return True


# Function to load the pre-trained fingerprint model
def load_fingerprint_model(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    try:
        model = tf.keras.models.load_model(path)
        return model
    except OSError as e:
        raise RuntimeError(f"Error loading the model file: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

# Usage
file_path = 'C:/Users/sabhya/Desktop/NullClass/task3.1/fingerprint_model.keras'
try:
    model = load_fingerprint_model(file_path)
except RuntimeError as e:
    print(f"Model loading failed: {e}")
    model = None

if model is not None:
    model.save('C:/Users/sabhya/Desktop/NullClass/task3.1/fingerprint_model.h5', save_format='tf')
else:
    print("Model is not defined. Saving failed.")



# Function to load and preprocess the fingerprint image
def load_image(file_path):
   
    # Load and preprocess the fingerprint image.
   
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not load image. Please check the file path.")
    image = cv2.resize(image, (256, 256))  # Resize to a standard size
    return image

# Function to preprocess the image for the model
def preprocess_for_model(image):
    
    # Preprocess the image as required by the model (e.g., resize, normalize).
    
    image_resized = cv2.resize(image, (224, 224))  # Adjust size based on your model
    image_normalized = image_resized / 255.0  # Example normalization
    return np.expand_dims(image_normalized, axis=0)

# Function to check if the image is a fingerprint using the model
def is_fingerprint_image(image, model):
   
    # Check if the given image is likely a fingerprint using a pre-trained model.
   
    preprocessed_image = preprocess_for_model(image)
    prediction = model.predict(preprocessed_image)
    return prediction[0][0] > 0.5  # Adjust threshold based on your model

# Function to preprocess the fingerprint image for feature extraction
def preprocess_fingerprint(image):
    
    # Preprocess the fingerprint image for feature extraction.
    
    equalized_image = cv2.equalizeHist(image)  # Enhance contrast
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)  # Reduce noise
    return blurred_image

# Function to extract features from the fingerprint image
def extract_features(image):
   
    # Extract features from the fingerprint image using custom methods.
    
    features = np.mean(image, axis=0)  # Example: mean projection
    return features

# Function to compare two sets of features
def compare_features(features1, features2):
   
    # Compare two sets of features.
    
    distance = np.linalg.norm(features1 - features2)
    threshold = 100  # Set an appropriate threshold
    return distance < threshold

# Function to retrieve stored fingerprint features from the database
def get_stored_fingerprint_data(user_id, db):
    
    # Retrieve stored fingerprint features from the database.
    
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("SELECT fingerprint_data FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        if result:
            return np.frombuffer(result['fingerprint_data'], dtype=np.float32)
        else:
            print("No registered fingerprint found for this user.")
            return None
    except mysql.connector.Error as err:
        print(f"Error retrieving fingerprint features: {err}")
        return None
    finally:
        cursor.close()

# Route to register fingerprint
@app.route('/register', methods=['POST'])
def register_fingerprint():
   
    # Register a new fingerprint for a user.
    
    if 'user_id' not in session:
        flash("You must be logged in to register a fingerprint.")
        return redirect(url_for('login'))

    user_id = session['user_id']
    fingerprint_image = request.files['fingerprint_image']

    # Load and preprocess fingerprint image
    image = load_image(fingerprint_image)
    if not is_fingerprint_image(image, model):
        flash("The image does not appear to be a fingerprint.")
        return redirect(url_for('index'))

    processed_image = preprocess_fingerprint(image)
    features = extract_features(processed_image)

    # Convert features to bytes for storage
    features_bytes = features.astype(np.float32).tobytes()

    cursor = db.cursor()
    try:
        cursor.execute(
            "UPDATE users SET fingerprint_data = %s, finger_enabled = TRUE WHERE id = %s",
            (features_bytes, user_id)
        )
        db.commit()
        flash("Fingerprint registered successfully.")
    except mysql.connector.Error as err:
        print(f"Error storing fingerprint: {err}")
        flash("An error occurred while registering the fingerprint.")
    finally:
        cursor.close()
        db.close()

    return redirect(url_for('index'))

# Route to verify fingerprint
@app.route('/verify', methods=['POST'])
def verify_fingerprint():
   
   # Verify the fingerprint of a user.
  
    if 'user_id' not in session:
        flash("You must be logged in to verify a fingerprint.")
        return redirect(url_for('login'))

    user_id = session['user_id']
    fingerprint_image = request.files['fingerprint_image']

    # Load and preprocess fingerprint image
    image = load_image(fingerprint_image)
    if not is_fingerprint_image(image, model):
        flash("The image does not appear to be a fingerprint.")
        return redirect(url_for('index'))

    processed_image = preprocess_fingerprint(image)
    features = extract_features(processed_image)

    # Retrieve the stored fingerprint features
    stored_features = get_stored_fingerprint_data(user_id, db)

    if stored_features is None:
        flash("No registered fingerprint found for this user.")
        return redirect(url_for('index'))

    # Compare extracted features with stored features
    if compare_features(features, stored_features):
        flash("Fingerprint verification successful.")
    else:
        flash("Fingerprint verification failed.")

    db.close()
    return redirect(url_for('index'))

def recognize_fingerprint(fingerprint_image, user_id, model, db):
  
    try:
        # Load and preprocess fingerprint image
        image = load_image(fingerprint_image)
        if not is_fingerprint_image(image, model):
            return False, "The image does not appear to be a valid fingerprint."

        # Preprocess the fingerprint for feature extraction
        processed_image = preprocess_fingerprint(image)
        features = extract_features(processed_image)

        # Retrieve the stored fingerprint features from the database
        stored_features = get_stored_fingerprint_data(user_id, db)

        if stored_features is None:
            return False, "No registered fingerprint found for this user."

        # Compare the extracted features with the stored features
        if compare_features(features, stored_features):
            return True, "Fingerprint recognized successfully."
        else:
            return False, "Fingerprint not recognized."

    except ValueError as e:
        return False, f"Error processing fingerprint image: {e}"
    except Exception as e:
        return False, f"An unexpected error occurred: {e}"







@app.route('/enable_fingerprint', methods=['POST'])
def enable_fingerprint_route():
    print("Request received for enabling fingerprint")
    fingerprint_data = request.form.get('fingerprintData')
    
    if fingerprint_data:
        # Ensure the fingerprint_data is in bytes
        if isinstance(fingerprint_data, str):
            fingerprint_data = fingerprint_data.encode()
        return enable_fingerprint(fingerprint_data)
    else:
        flash("No fingerprint data provided.", "error")
        return jsonify({'message': 'No fingerprint data provided.'}), 400

def enable_fingerprint(fingerprint_data):
    if 'user_id' not in session:
        flash("You must be logged in to access this page.", "error")
        return jsonify({'message': 'User not logged in.'}), 401
    
    if not fingerprint_data:
        flash("No fingerprint data provided.", "error")
        return jsonify({'message': 'No fingerprint data provided.'}), 400

    # Hash the fingerprint data (assuming it's binary data)
    fingerprint_hash = hashlib.sha256(fingerprint_data).hexdigest()

    cursor = db.cursor()
    try:
        cursor.execute("UPDATE users SET finger_enabled = TRUE, fingerprint_data = %s WHERE id = %s", (fingerprint_hash, session['user_id']))
        db.commit()
        send_otp_route()
        
        flash("Fingerprint recognition enabled successfully!", "success")
        return redirect(url_for('verify_otp'))
        return jsonify({'message': 'Fingerprint recognition enabled successfully.'}), 200
    except mysql.connector.Error as err:
        db.rollback()  # Rollback in case of error
        flash(f"MySQL Error: {err}", "error")
        return jsonify({'message': f"MySQL Error: {err}"}), 500
    finally:
        cursor.close()



@app.route('/disable_fingerprint', methods=['POST'])
def disable_fingerprint_route():
    # Assuming user ID is stored in the session
    user_id = session.get('user_id')
    if user_id:
        disable_fingerprint(user_id)
        flash("Fingerprint disabled successfully.")
        return redirect(url_for('profile'))
    else:
        flash("User not logged in.", "error")
        return redirect(url_for('login'))
    
def disable_fingerprint(user_id):
    cursor = db.cursor()
    try:
        cursor.execute("UPDATE users SET fingerprint_data = NULL, finger_enabled = FALSE WHERE id = %s", (user_id,))
        db.commit()
        send_otp_route()
        print("Fingerprint disabled successfully.")
        return redirect(url_for('verify_otp'))
    except mysql.connector.Error as err:
        print(f"Error disabling fingerprint: {err}")
        db.rollback()  # Rollback in case of error
    finally:
        cursor.close()
    

# Logout route
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    
    app.run(debug=True)
