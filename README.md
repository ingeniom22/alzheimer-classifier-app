
# Alzheimer Classifier Web Application
@ingeniom22

## Overview
Welcome to the Alzheimer Classifier web application! This project is a work-in-progress submission for the EDE Challenge 2023. The aim of this application is to provide a simple and user-friendly interface to classify Alzheimer's disease stages from MRI scan images. The application is built using a combination of PyTorch, FastAPI, Jinja2, and Bootstrap.

## Features
- Upload an MRI scan image in JPG or PNG format.
- Perform Alzheimer's disease stage classification using a pre-trained, fine-tuned model.
- Display classification results with associated probabilities.
- Visualize the uploaded image and classification results in a clean and organized interface.
- Utilize the power of FastAPI to handle backend operations efficiently.
- Utilize Bootstrap for responsive and visually appealing design.

## Project Structure
The project structure is organized as follows:

- `app.py`: The main FastAPI application file that handles routing and prediction.
- `model/predictor-torchscript.pt`: The pre-trained PyTorch model for Alzheimer's disease classification.
- `static/`: Static files such as images and CSS for the application.
- `templates/`: Jinja2 templates for rendering HTML pages.
- `css/style.css`: Custom CSS styles for the application's appearance.

## Getting Started
To run the application locally:

1. Clone this repository.
2. Create a virtual environment using `python -m virtual env env`
3. Activate the virtual environment
4. Install the required dependencies using `pip install -r requirements.txt`.
5. Run the FastAPI application using `uvicorn app.main:app --host 0.0.0.0 --port 8080`.

## Future Enhancements
This project is a work in progress, and there are several enhancements planned:

- Improved classification model with higher accuracy.
- User authentication and management.
- Better visualization of classification results.
- Performance optimization for handling large images efficiently.
- Explainability with GradCAM
- Deployment to a production environment for wider access.

## Acknowledgments
This project was developed as a part of the EDE Challenge 2023 and is a collaborative effort by our team, RKF45. We are grateful for the guidance and support from the challenge organizers.

## Contact
For any inquiries or feedback, please contact [James Michael Fritz](jamesmichael0444@gmail.com).

> "The Destruction is crazy, and the Preservation is dumb. All the Aeons are stubborn, and Aha is embarrassed! Aha is embarrassed! Aha is embarrassed..."

---
