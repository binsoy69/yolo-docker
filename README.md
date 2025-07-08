# YOLOv11 Classifier API

A Dockerized FastAPI service that runs a YOLOv11 classification model and exposes it as a REST API.

## Usage

- **POST /predict**  
  Upload an image via `multipart/form-data`, receive JSON classification result.

## Example

```bash
curl -X POST https://<your-service>.onrender.com/predict \
  -H "accept: application/json" \
  -F "file=@image.jpg"
```
