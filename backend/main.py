from backend.app.main import app

if __name__ == "__main__":
    import uvicorn
    # Maintain compatibility with existing startup commands pointing to backend.main:app
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
