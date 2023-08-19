import base64
import io
import json
import re

import numpy as np
import plotly
import plotly.express as px
import torch
import torchvision.transforms as T
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image

from modules.model import EfficientNetV2
from modules.predictor import Predictor

# Initialize API Server
app = FastAPI(
    title="Alzheimer Classification",
    description="EfficientNet-V2",
    version="0.0.1",
)

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Mount static folder, like demo pages, if any
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

checkpoint = torch.load(
    "checkpoints\model-checkpoint (1).pt", map_location=torch.device("cpu")
)
new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
model = EfficientNetV2(num_classes=4, dropout=0.2)

# print(set(model.state_dict().keys()))
# print(set(new_state_dict.keys()))

model.load_state_dict(new_state_dict)
model.eval()

mean = [0.2951, 0.2955, 0.2957]
std = [0.3167, 0.3168, 0.3168]
class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

predictor = Predictor(
    model=model,
    class_names=class_names,
    mean=mean,
    std=std,
)


def get_prediction_chart(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = T.PILToTensor()(image).unsqueeze(0)
    softmax = predictor(image).data.cpu().numpy().squeeze()

    idxs = np.argsort(softmax)[::-1]

    label = []
    prob = []
    # Loop over the classes with the largest softmax
    for i in range(len(idxs)):
        # Get softmax value
        p = round(softmax[idxs[i]])
        # Get class name
        class_name = class_names[idxs[i]]
        class_name = re.sub(
            r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r" \1", class_name
        )  # don't ask James how this works

        label.append(class_name)
        prob.append(p)

    fig = px.bar(
        x=prob,
        y=label,
        labels={
            "x": "Probability",
            "y": "Alzheimer Stage",
        },
        title="Prediction Result",
    )

    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # print(label, prob)
    return graphJSON


def prep_img_for_gradcam(image_bytes):
    """A function that gets a URL of an image,
    and returns a numpy image and a preprocessed
    torch tensor ready to pass to the model"""

    img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    rgb_img_float = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img_float, mean=mean, std=std)
    return img, rgb_img_float, input_tensor


def get_gradcam_img(model, input_tensor, image_float):
    target_layers = [model.model.features[-1][0]]

    with GradCAM(model=model, target_layers=target_layers, use_cuda=False) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
        )[0, :]

    cam_image = show_cam_on_image(image_float, grayscale_cam, use_rgb=True)
    return cam_image


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def home_predict(request: Request, file: UploadFile = File(...)):
    # Perform model prediction using the uploaded file
    image_data = await file.read()
    encoded_image = base64.b64encode(image_data).decode("utf-8")
    prediction = get_prediction_chart(image_data)

    img, rgb_img_float, input_tensor = prep_img_for_gradcam(image_data)

    cam_image = Image.fromarray(
        get_gradcam_img(
            model=model, input_tensor=input_tensor, image_float=rgb_img_float
        )
    )
    buffer = io.BytesIO()
    cam_image.save(buffer, "PNG")
    cam_png = buffer.getvalue()
    encoded_cam_png = base64.b64encode(cam_png).decode("utf-8")

    # Render the result using the template
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction_results": prediction,
            "uploaded_image": f"data:image/png;base64,{encoded_image}",
            "cam_image": f"data:image/png;base64,{encoded_cam_png}",
        },
    )


@app.get("/home", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/contact", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )
