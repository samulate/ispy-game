import imghdr
import os
import shutil
from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
from werkzeug.utils import secure_filename
from object_detection_with_huggingface import filepath_or_image_to_all_data
from object_detection_with_huggingface import text_to_embedding
from object_detection_with_huggingface import compute_similarity



app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'
os.makedirs(app.config['UPLOAD_PATH'], exist_ok=True)
files = []
color_name = None
rgb_value = (0,0,0)
orig_img = None
cropped_image = None
name = None
similar = None
display_index = 0

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.route('/')
def index():
    #rgb_string = f"rgb({rgb_value[0]}, {rgb_value[1]}, {rgb_value[2]})"
    return render_template('index.html', files=files, color=color_name, similar=similar, name=name,display_index=display_index,rgb_value=rgb_value)

@app.route('/', methods=['POST'])
def upload_files():
    global color_name, rgb_value, orig_img, cropped_image, name, files, similar, display_index

    files = []
    color_name = None
    rgb_value = None
    orig_img = None
    cropped_image = None
    name = None
    similar = None
    display_index = 0
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    print(filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or file_ext != validate_image(uploaded_file.stream):
            abort(400)
            
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], 'img'+file_ext))
        color_name, rgb_value, orig_img, cropped_image, name, result = filepath_or_image_to_all_data(os.path.join(app.config['UPLOAD_PATH'], 'img'+file_ext))

        files.append('img'+file_ext)
        files.append('img2.jpg')
        files.append('img3.jpg')
        # print(files)
        img2 = os.path.join('uploads', 'predict', 'img.jpg')
        shutil.move(img2, os.path.join(app.config['UPLOAD_PATH'], 'img2.jpg'))
        
        os.removedirs(os.path.join('uploads', 'predict'))

        print(files)

    return redirect(url_for('index'))

@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

@app.route('/guess', methods=['POST'])
def guess():
    global similar, display_index
    input_string = request.form.get("guess")
    user_vector = text_to_embedding(input_string)
    program_vector = text_to_embedding(name)
    similar = compute_similarity(user_vector,program_vector, 0.5)

    
    return redirect(url_for('index'))
    

@app.route('/boxes', methods=['POST'])
def boxes():
    global display_index
    
    display_index = 1 - display_index

    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)