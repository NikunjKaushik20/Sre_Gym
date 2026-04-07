"""FastAPI app for SRE-Gym environment."""
from openenv.core.env_server.http_server import create_app

from server.sre_environment import SREEnvironment
from models import SREAction, SREObservation

app = create_app(SREEnvironment, SREAction, SREObservation, env_name="sre_gym")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
