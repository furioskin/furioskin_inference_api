from fastapi import FastAPI, UploadFile
import uvicorn

app = FastAPI()

@app.post("/infer")
def infer(file: UploadFile):
        
    return {"label": 1}

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=65501)