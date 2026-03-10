from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from PIL import Image
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def detect_skin_tone(image_path):

    img = cv2.imread(image_path)
    avg_color = np.mean(img, axis=(0,1))

    brightness = np.mean(avg_color)

    if brightness > 200:
        return "Fair"
    elif brightness > 150:
        return "Medium"
    elif brightness > 100:
        return "Olive"
    else:
        return "Deep"


def get_ai_recommendation(skin_tone, gender):

    prompt = f"""
    Give fashion styling advice for a {gender} with {skin_tone} skin tone.
    Include:
    - outfit ideas
    - color palette
    - hairstyle
    - accessories
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}],
        max_tokens=500
    )

    return response.choices[0].message.content


@app.route("/", methods=["GET","POST"])
def index():

    result = None
    skin_tone = None

    if request.method == "POST":

        file = request.files["image"]
        gender = request.form["gender"]

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        skin_tone = detect_skin_tone(filepath)

        result = get_ai_recommendation(skin_tone, gender)

    return render_template("index.html", result=result, skin_tone=skin_tone)


if __name__ == "__main__":
    app.run(debug=True)