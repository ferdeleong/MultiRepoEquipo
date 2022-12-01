# TC2008B. Sistemas Multiagentes y Gráficas Computacionales
# Python flask server to interact with Unity. Based on the code provided by Sergio Ruiz.
# Octavio Navarro. November 2022
from typing import Union
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import boids.streetModel as mod
import json
import os


app = FastAPI()

class AgentsModel(BaseModel):
    type: str


class CarItem(AgentsModel):
    type = "positions"
    boidId: str
    x: float
    y: float
    z: float
    direction: str
    
class TFItem(AgentsModel):
    type = "trafficLights"
    tfId : str
    green: str
    yellow: str
    red: str

M = 12
N = 12

# Definimos el número de agentes
NUM_AGENTS = 6

# Definimos tiempo máximo (segundos)
MAX_TIME = 0.6

# Registramos el tiempo de inicio y corremos el modelo
model = mod.StreetModel(NUM_AGENTS, M, N)


port=int(os.getenv('PORT',8000))

@app.get("/init")
def root():
    return json.dumps({"num_agents":NUM_AGENTS, "w": M, "h": N})
    
@app.get("/sim")
async def read_items():
    global model
    pos,states = mod.step_model(model)
    return mod.positionsToJSON(pos, states)     



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)




