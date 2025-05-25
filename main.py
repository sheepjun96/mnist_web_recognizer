# .venv\Scripts\activate
# .venv\Scripts\python.exe -m uvicorn main:app --reload

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import io
from dl_model import MNISTDeepLearning
from svm_model import MNISTLinearSVM

app = FastAPI()
dl_model = MNISTDeepLearning()
svm_model = MNISTLinearSVM()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def get_home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict_dl")
async def predict_dl(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    img.save("received_image_dl.png")

    img = np.array(img)
    img = img / 255.0
    prediction = dl_model.predict_digit(img)

    return JSONResponse(content={"prediction": int(prediction)})

@app.post("/predict_svm")
async def predict_svm(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    img.save("received_image_svm.png")

    img = np.array(img).flatten()
    prediction = svm_model.predict_digit(img)

    return JSONResponse(content={"prediction": int(prediction)})
