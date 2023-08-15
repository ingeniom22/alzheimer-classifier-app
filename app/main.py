import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.logger import logger
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import torch


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

# Load custom exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)


@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """

    logger.info("Running envirnoment: {}".format(CONFIG["ENV"]))
    logger.info("PyTorch using device: {}".format(CONFIG["DEVICE"]))

    # Initialize the pytorch model
    model = Model()
    model.load_state_dict(
        torch.load(CONFIG["MODEL_PATH"], map_location=torch.device(CONFIG["DEVICE"]))
    )
    model.eval()

    # add model and other preprocess tools too app state
    app.package = {"scaler": load(CONFIG["SCALAR_PATH"]), "model": model}  # joblib.load
