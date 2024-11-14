from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

dic = {0: 'bad', 1: 'good'}

model = load_model('mainmodel1.h5')

def predict_label(img_path):
    try:
        img = image.load_img(img_path, target_size=(150, 150))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return dic[predicted_class]
    except Exception as e:
        print(f"Error predicting label: {str(e)}")
        return None


@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index1.html")



@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)
        return render_template("index.html", prediction=p, img_path=img_path)
    return "Invalid request"


if __name__ == '__main__':
    app.run(debug=True)
