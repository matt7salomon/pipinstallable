from fastapi import File, UploadFile
from fastapi import FastAPI
from string import punctuation
import pickle
from fastapi.exceptions import HTTPException
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


app = FastAPI(title="Trellis document classifier", description="Matt Salomon's document classifier", version="1.0")

@app.on_event('startup')
def make_model():
    pass

# loadin the pickle files
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except:
    print("model.pkl not found")

try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except:
    print("vectorizer.pkl not found")
try:
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
except:
    print("label_encoder.pkl not found")

ml_dict = {'model': model, 'vectorizer': vectorizer, 'label_encoder': label_encoder}

@app.get("/")
def start():
    return {'Please type in 0.0.0.0:8000/docs or the localhost:port/docs you were given from terminal in the browser'}

def preprocess_text(text):
    """ Apply any preprocessing methods"""
    text = text.lower()
    text = ''.join(c for c in text if c not in punctuation)
    return text


def parse_csv(df):
    res = df.to_json(orient="records")
    parsed = json.loads(res)
    return parsed


@app.post("/upload_test_document")
def upload(file: UploadFile = File(...), format=[".txt"]):
    if file.content_type not in ["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise HTTPException(400, detail="Invalid document type. I only support .txt files at the moment!")
    try:
        contents = file.file.read()
        # with open(file.filename, 'wb') as f:
        #     f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file - Please upload a file that i can encode or change the encoder"}
    finally:
        file.file.close()
    if str(contents) == '':
        raise Exception('The file is empty.')
    contents = preprocess_text(str(contents))
    ml_dict['contents'] = str(contents)
    return {"message": f"Successfully uploaded {file.filename}"}


@app.post('/classify_document', tags=["predictions"])
async def get_prediction():
    if not ml_dict or not {k for k in ['contents']} <= set(ml_dict):
        raise HTTPException(status_code=404, detail=f"you have not uploaded a .text document yet!")
    else:
        logger.info("INFO:    I do have the text document to be working with! 200 OK")
    if not ml_dict or not {k for k in ['model','vectorizer','contents','label_encoder']} <= set(ml_dict):
        raise HTTPException(status_code=404, detail=f"The required dictionary ml_dict does not exist or does not have the required keys.")
    else:

        logger.info("INFO:    ml_dict dictionary has everything i need to score 200 OK")

    model = ml_dict['model']


    vectorizer = ml_dict['vectorizer']
    label_encoder = ml_dict['label_encoder']
    X_test = vectorizer.transform([ml_dict['contents']])

    prediction = model.predict(X_test).tolist()
    log_proba = model.predict_proba(X_test).tolist()
    prediction_encoded = str(label_encoder[prediction])
    log_proba_encoded = pd.DataFrame({'label': label_encoder, 'probability': model.predict_proba(X_test[0])[0]}).sort_values(
        'probability', ascending = False)
    return {"message": "Classification successful",
            "prediction": prediction_encoded,
            "All_labeles_with_probability": parse_csv(log_proba_encoded)}