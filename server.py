import os
from fastapi import FastAPI
from gradio.routes import mount_gradio_app
from app import demo

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# Mount Gradio at root
app = mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")), workers=1)