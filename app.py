import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory, session
import uuid
from werkzeug.utils import secure_filename
import cv2
from object_detection_with_huggingface import filepath_or_image_to_all_data
from object_detection_with_huggingface import text_to_embedding
from object_detection_with_huggingface import compute_similarity
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()
import io

app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = os.path.join(app.root_path, 'uploads')
app.secret_key = 'Reynaldo'  # required for Flask sessions
os.makedirs(app.config['UPLOAD_PATH'], exist_ok=True)


# color_name = None
# rgb_value = (0,0,0)
# orig_img = None
# cropped_image = None
# name = None
# similar = None
# display_index = 0


def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.route('/')
def index():
    print("redirected")
    files = session.get('files', [])
    
    
    color_name = session.get('color_name', None)
    name = session.get('name', None)
    similar = session.get('similar', None)
    display_index = session.get('display_index', 0)
    rgb_value = session.get('rgb_value', (0,0,0))
    points = session.get('points',None)
    error = session.get('error', False)
    guesses = session.get('guesses', 0)
    hint_used = session.get('hint_used', False)
    
    # orig_img = session.get('orig_img', [])
    # cropped_image = session.get('cropped_image', [])
    
    #rgb_string = f"rgb({rgb_value[0]}, {rgb_value[1]}, {rgb_value[2]})"
    return render_template('index.html', files=files, color=color_name, similar=similar, name=name,display_index=display_index,rgb_value=rgb_value,points=points,error=error,guesses=guesses,hint_used=hint_used)





@app.route('/', methods=['POST'])
def upload_files():
    # global color_name, rgb_value, orig_img, cropped_image, name, files, similar, display_index
    try:
        kill_current_session_images()
    except:
        pass

    # files = []
    hint_used = session.get('hint_used',False)
    points = session.get('points',0)
    if type(points) != int:
        points = 0
    session['points'] = points

    error = session.get('error', False)
    error = False
    session['error'] = error

    uploaded_file = request.files['file-upload']



    files = session.get('files', [])
    color_name = session.get('color_name', None)
    name = session.get('name', None)
    similar = session.get('similar', None)
    display_index = session.get('display_index', 0)
    rgb_value = session.get('rgb_value', (0,0,0))
    unique_name = session.get('unique_name', None)
    file_ext = session.get('file_ext', None)
    # orig_img = session.get('orig_img', None)
    # cropped_image = session.get('cropped_image', None)


    filename = secure_filename(uploaded_file.filename)

    print(filename)
    if filename != '':
        
        session['file_ext'] = os.path.splitext(filename)[1].lower()

        file_ext = session.get('file_ext', None)

        if file_ext == ".jpeg":
            file_ext = ".jpg"

        

        if file_ext == ".heic":
            file_bytes = uploaded_file.read()
            file_stream = io.BytesIO(file_bytes)

            img = Image.open(file_stream)
            img = img.convert("RGB")

            jpeg_stream = io.BytesIO()
            img.save(jpeg_stream,"JPEG",quality=95)
            jpeg_stream.seek(0)

           

            session['unique_name'] = f"{uuid.uuid4().hex}" 

            unique_name = session['unique_name']

            save_path = os.path.join(app.config['UPLOAD_PATH'], unique_name+".jpg")

            with open(save_path, 'wb') as f:
                f.write(jpeg_stream.read())

            files.append(unique_name+'.jpg')

            session['files'] = files
            file_ext = ".jpg"

        

        
        else:

            if file_ext not in app.config['UPLOAD_EXTENSIONS'] or file_ext != validate_image(uploaded_file.stream):
                abort(400)

            session['unique_name'] = f"{uuid.uuid4().hex}" 
            unique_name = session.get('unique_name', None)
            
            save_path = os.path.join(app.config['UPLOAD_PATH'], unique_name+file_ext)
            uploaded_file.save(save_path)

            files.append(unique_name+file_ext)
            session['files'] = files

        session['file_ext'] = file_ext

        prepare_round(save_path, unique_name)



    return redirect(url_for('index'))

@app.route('/again', methods=['POST'])
def go_again():
    unique_name = session.get('unique_name', None)
    file_ext = session.get('file_ext', None)
    if not unique_name or not file_ext:
        return redirect(url_for('index'))

    prepare_round(os.path.join(app.config['UPLOAD_PATH'], unique_name + file_ext), unique_name)
    return redirect(url_for('index'))


def prepare_round(save_path, unique_name):
    # Save to explicit paths instead of relying on YOLO's global output settings.
    session['files'] = [os.path.basename(save_path), f'{unique_name}_2.jpg', f'{unique_name}_3.jpg']
    try:
        color_name, rgb_value, _, _, name, result, arr_bgr = filepath_or_image_to_all_data(save_path)
        result.save(filename=os.path.join(app.config['UPLOAD_PATH'], f'{unique_name}_2.jpg'))
        crop_path = os.path.join(app.config['UPLOAD_PATH'], f'{unique_name}_3.jpg')
        if not cv2.imwrite(crop_path, arr_bgr):
            raise OSError(f'Could not save cropped image: {crop_path}')

        session.update(color_name=color_name, name=name, similar=None, display_index=0,
                       rgb_value=rgb_value.tolist(), hint_used=False, guesses=0, error=False)
    except Exception:
        app.logger.exception('Could not process uploaded image %s', save_path)
        kill_current_session_images()
        session['error'] = True


@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

@app.route('/guess', methods=['POST'])
def guess():
    print("guessing")
    # global similar, display_index
    # similarity  = session.get('similar', None)
    # if similarity != None:
    #     if similarity == True:
    #         upload_files()
    #         return redirect(url_for('index'))

    input_string = request.form.get("guess", "").strip()
    if not input_string or not session.get('name') or session.get('similar'):
        return redirect(url_for('index'))

    user_vector = text_to_embedding(input_string)
    program_vector = text_to_embedding(session.get('name', "default"))
    # try:
    session['similar'] = bool(compute_similarity(user_vector,program_vector, 0.5))
    session['guesses'] = session.get('guesses', 0) + 1
    # except:
    #     print(f"could not set session['similar'] to compute_similarity(user_vector,program_vector, 0.5), which returned {compute_similarity(user_vector,program_vector, 0.5)}")
    if session['similar']:
        if session.get('hint_used', False):
            session['points'] += 10
        else:
            session['points'] += 20
    elif session['points'] > 0:
        session['points'] -= 1
    return redirect(url_for('index'))
    
@app.route('/reset-session', methods=['POST'])
def reset_current_user():
    # global display_index
    try:
        kill_current_session_images()
    except:
        pass
    session['points'] = None
    session['color_name'] = None
    session['name'] = None
    session['similar'] = None
    session['display_index'] = 0
    session['rgb_value'] = (0,0,0)
    session['hint_used'] = False
    session['guesses'] = 0
    session['files'] = []
    session['error'] = False

    

    return redirect(url_for('index'))

@app.route('/boxes', methods=['POST'])
def boxes():
    # global display_index
    if not session.get('name') or session.get('similar'):
        return redirect(url_for('index'))
    session['hint_used'] = True
    session['display_index'] = 1 - session.get('display_index',0)

    return redirect(url_for('index'))

def kill_current_session_images():
    print(f"killing files in {session.get('files', [])}")
    for file in session.get('files',[]):
        print(f"file: {file}")
        if os.path.exists(os.path.join(app.config['UPLOAD_PATH'],file)):
            os.remove(os.path.join(app.config['UPLOAD_PATH'],file))
            print("removed (allegedly)")
    session['files'] = []
    session['color_name'] = None
    session['name'] = None
    session['similar'] = None
    session['display_index'] = 0
    session['rgb_value'] = (0,0,0)
    session['error'] = False
    session['hint_used'] = False
    session['guesses'] = 0
    # session['orig_img'] = orig_img


if __name__ == "__main__":
    app.run(debug=True)
