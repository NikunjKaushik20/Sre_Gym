"""FastAPI app for SRE-Gym environment."""
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server.http_server import create_app

from server.sre_environment import SREEnvironment
from models import SREAction, SREObservation

app = create_app(SREEnvironment, SREAction, SREObservation, env_name="sre_gym")

# ── Dashboard and static file serving ──
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    """Serve the incident command center dashboard."""
    html_path = STATIC_DIR / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>SRE-Gym API</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>")


@app.get("/health")
async def health():
    return {"status": "ok", "environment": "sre_gym"}


def main():
    """Entry point for the SRE-Gym server (used by [project.scripts])."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
