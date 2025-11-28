from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import joblib
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# ensure house_model.pkl exists in the project; if not, try user's Downloads, else create fallback
PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "house_model.pkl"
USER_DOWNLOAD_PATH = Path(r"C:\Users\Athil S\Downloads\house_model.pkl")

is_fallback = False

if MODEL_PATH.exists():
    logging.info("Using model from project: %s", MODEL_PATH)
else:
    if USER_DOWNLOAD_PATH.exists():
        logging.info("Found model in Downloads (%s). Copying to project.", USER_DOWNLOAD_PATH)
        shutil.copy(USER_DOWNLOAD_PATH, MODEL_PATH)
    else:
        logging.warning("No model found; creating a lightweight fallback at %s", MODEL_PATH)
        class _FallbackModel:
            def predict(self, X):
                # deterministic fallback: return 0.0 for each input row
                return [0.0 for _ in X]
        # mark the instance so we can detect it after loading from disk
        fallback_instance = _FallbackModel()
        setattr(fallback_instance, "_is_fallback", True)
        joblib.dump(fallback_instance, MODEL_PATH)
        is_fallback = True

# load the model (now safe)
model = joblib.load(str(MODEL_PATH))
# detect if the loaded object is the fallback we created
is_fallback = bool(getattr(model, "_is_fallback", False))
logging.info("Loaded model type=%s, is_fallback=%s", type(model), is_fallback)

class Input(BaseModel):
    data: Optional[list] = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]

@app.post("/predict")
def predict(input: Input = Input()):
    # validate input
    if input.data is None or not isinstance(input.data, list):
        raise HTTPException(status_code=400, detail="Request JSON must include 'data' as a list of 8 numbers.")
    if len(input.data) != 8:
        raise HTTPException(status_code=400, detail=f"Expected 8 features but received {len(input.data)}.")
    try:
        pred = model.predict([input.data])
    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed; check server logs for details.")
    return {"prediction": pred[0]}

@app.get("/")
def root():
    return {
        "model_path": str(MODEL_PATH),
        "is_fallback": is_fallback,
        "model_type": str(type(model)),
        "has_predict": callable(getattr(model, "predict", None))
    }

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.get("/test")
def test_predict():
    sample = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]
    try:
        pred = model.predict([sample])
    except Exception:
        logging.exception("Test prediction failed")
        raise HTTPException(status_code=500, detail="Test prediction failed; check server logs.")
    return {"sample": sample, "prediction": pred[0], "is_fallback": is_fallback}

# lightweight inspection endpoint for the loaded model (safe for browser)
@app.get("/model_info")
def model_info():
    info = {
        "model_type": str(type(model)),
        "has_predict": callable(getattr(model, "predict", None)),
        "is_fallback": is_fallback,
    }
    # expose some numeric attributes if present (safe access)
    for attr in ("coef_", "intercept_", "n_features_in_"):
        if hasattr(model, attr):
            try:
                val = getattr(model, attr)
                # convert numpy arrays/scalars to python lists/values where reasonable
                if hasattr(val, "tolist"):
                    val = val.tolist()
                info[attr] = val
            except Exception:
                info[attr] = "unreadable"
    return info

@app.get("/ui")
def ui():
    # serve the separate static UI file
    ui_path = PROJECT_DIR / "static" / "ui.html"
    if ui_path.exists():
        return FileResponse(str(ui_path), media_type="text/html")
    raise HTTPException(status_code=404, detail="UI not found. Create static/ui.html in project.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=1000)