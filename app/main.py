import base64
import io
import re
from fastapi.templating import Jinja2Templates
import uvicorn
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import numpy as np

from PIL import Image
import torch
import torchvision.transforms as T

# Initialize API Server
app = FastAPI(
    title="Alzheimer Classification",
    description="EfficientNet-V2",
    version="0.0.1",
)

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Mount static folder, like demo pages, if any
app.mount("/static", StaticFiles(directory="static/"), name="static")
templates = Jinja2Templates(directory="templates")

model = torch.jit.load("model\predictor-torchscript.pt")
model.eval()


def perform_prediction(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = T.PILToTensor()(image).unsqueeze(0)
    softmax = model(image).data.cpu().numpy().squeeze()

    idxs = np.argsort(softmax)[::-1]

    # Create a list to store class probabilities and names
    class_probs = []

    # Loop over the classes with the largest softmax
    for i in range(len(idxs)):
        # Get softmax value
        p = softmax[idxs[i]]

        # Get class name
        class_name = model.class_names[idxs[i]]
        class_name = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', class_name) # don't ask James  how this works

        class_probs.append({"class_name": class_name, "probability": p})

    print(class_probs)
    return class_probs


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def home_predict(request: Request, file: UploadFile = File(...)):
    # Perform model prediction using the uploaded file
    image_data = await file.read()
    encoded_image = base64.b64encode(image_data).decode("utf-8")
    prediction = perform_prediction(image_data)

    # Render the result using the template
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction_results": prediction,
            "uploaded_image": encoded_image,
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )
