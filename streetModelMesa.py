#!/usr/bin/env python
# coding: utf-8

# # Actividad Integradora 2
# 
# Realizado por 
# A01366768 Andrea Corona Arroyo 
# A01367480 Gabriel Crisóstomo Navidad 
# A01197340 Maria Fernanda De León Gomez 
# A01570679 Diana Laura Hernández Villarreal 	
# A01384237 Roberto Abraham Pérez Iga                 	
# A01412375 Jorge Claudio González Becerril          	


# ## Imports
# 
# Antes de empezar a crear el modelo de los robots de limpieza con multiagentes es necesario tener instalado los siguientes paquetes:
# - `python`
# - `mesa`: el framework de Python para el modelado de agentes.
# - `numpy`
# - `matplotlib`
# 
# Para poder modelar usando el framework de `mesa` es necesario importar dos clases: una para el modelo general, y otro para los agentes. 

# In[3]:


# La clase `Model` se hace cargo de los atributos a nivel del modelo, maneja los agentes. 
# Cada modelo puede contener múltiples agentes y todos ellos son instancias de la clase `Agent`.
from mesa import Agent, Model 

# Debido a que no se permitirá más de un agente por celda elegimos `SingleGrid`.
from mesa.space import SingleGrid

# Con `SimultaneousActivation` hacemos que todos los agentes se activen de manera simultanea.
from mesa.time import SimultaneousActivation

# Vamos a hacer uso de `DataCollector` para obtener el grid completo cada paso (o generación) y lo usaremos para graficarlo.
from mesa.datacollection import DataCollector

# mathplotlib lo usamos para graficar/visualizar como evoluciona el autómata celular.
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**128


# Definimos los siguientes paquetes para manejar valores númericos.
import numpy as np
import pandas as pd
import random
import json

# Definimos otros paquetes que vamos a usar para medir el tiempo de ejecución de nuestro algoritmo.
import time
import datetime


# # Creación del Modelo


def obtenerCalle(model):
    '''
    Esta es una función auxiliar que nos permite guardar el grid para cada uno de los agentes.
    param model: El modelo del cual optener el grid.
    return una matriz con la información del grid del agente.
    '''
    grid = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, x, y = cell
        if isinstance(cell_content, CarAgent):
            grid[x][y] = 0
        elif isinstance(cell_content, TrafficLight):
            grid[x][y] = cell_content.status
        else:
            grid[x][y] = 1
    return grid

class CarAgent(Agent):
    '''
    Representa a un coche.
    '''
    def __init__(self, unique_id, model,orientation, direction):
        '''
        Agente representa un automóvil, se inicializa con un id, con el numero de movimientos realizados en 0, 
        con su siguiente posición en None, su orientación (V - vertical y H - horizontal),
        su direccion (U - arriba y D - abajo), el tipo de movimiento que va a 
        realizar (F - adelante, TR - vuelta a la derecha y TL - vuelta a la izquierda), una lista vacía de pasos 
        siguientes, y con el color del coche
        '''
        
        super().__init__(unique_id, model)
        self.numMovimientos = 0
        self.nextPos = None
        self.orientation = orientation
        self.direction = direction
        self.nextSteps = []
        self.movement = "F"   


    def moveForward(self):
        '''
        Este método se utiliza para avanzar el agente
        '''
        nextX = self.pos[0]
        nextY = self.pos[1]
        if self.orientation == "V":
            if self.direction == "U":
                nextX -= 1
            else:
                nextX += 1
        else:
            if self.direction == "U":
                nextY += 1
            else:
                nextY -= 1
        self.nextPos = (nextX, nextY)
           
    def turnLeft(self): 
        '''
        Este método se utiliza para girar el agente a la izquierda
        '''
        if self.orientation == "V":
            self.orientation = "H"
            if self.direction == "U":
                self.direction = "D"
            else:
                self.direction = "U"
        else:
            self.orientation = "V"

    def turnRight(self):
        '''
        Este método se utiliza para girar el agente a la derecha
        '''
        if self.orientation == "V":
            self.orientation = "H"
        else:
            self.orientation = "V"
            if self.direction == "U":
                self.direction = "D"
            else:
                self.direction = "U"
       
    def findClosestTrafficLight(self):
        '''
        Este método se utiliza para encontrar el semáforo más cercano
        '''
        if self.orientation == "V":
            if self.direction == "U":
                return 3
            return 1
        else:
            if self.direction == "U":
                return 0
            return 2
        
    def closestTrafficLight(self):
        '''
        Este método se utiliza para calcular la distancia entre el auto y el semáforo
        '''
        idx = self.findClosestTrafficLight()
        if idx % 2 == 0: 
            return idx,  abs(self.pos[1] - self.model.trafficLights[idx].pos[1])
        return idx, abs(self.pos[0] - self.model.trafficLights[idx].pos[0])
    
    def sendSignal(self):
        '''
        Este método se utiliza para enviar la señal al semáforo.
        '''
        idx, dist = self.closestTrafficLight()
        if dist <= 2:
            if not self.unique_id in self.model.trafficLights[idx].signals and not self.model.carInIntersection(self):
                self.model.trafficLights[idx].signals.append(self.unique_id)
                
                return True
        return False
        
    def checkTrafficLight(self):
        '''
        Este método se utiliza para revisar si el semáforo más cercano está en rojo y si puede avanzar
        '''
        idx, dist = self.closestTrafficLight()
        if self. model.trafficLights[idx].status == self.model.trafficLights[idx].RED and dist == 0 and len(self.nextSteps) == 0:
            return False
        return True
                 
    def movementTurn(self, direc):
        '''
        Este método se utiliza para realizar la vuelta dependiendo si es a la izquierda o derecha
        '''
        if direc == "TL":
            self.nextSteps.extend(["F", "TL"])
        else:
            self.nextSteps.extend(["F", "F", "TR"])
            
    def step(self):
        '''
        En este método el agente realiza las acciones del agente por cada paso.
        '''
        
        move = "S"
        #Checa el siguiente movimiento del auto
        if self.checkTrafficLight():
            move = "F"
            if self.closestTrafficLight()[1] == 0 and self.movement == "F":
                self.movement = random.choice(["F", "TR", "TL"])
                
            if (self.movement == "TL" or self.movement == "TR") and len(self.nextSteps) == 0:
                self.movementTurn(self.movement)

            if len(self.nextSteps) > 0:
                move = self.nextSteps.pop(0)
                
        if move == "TR":
            self.turnRight()
            self.movement = "F"
        elif move == "TL":
            self.turnLeft()
            self.movement = "F"
            
        
        if (self.movement == "F" or move == "F") and move != "S":
            self.moveForward()
            
             
        #Actualizar la posición en la lista de posiciones de los autos
        
        carsPositions = [car.nextPos for car in self.model.cars if car != self]
        
        if self.nextPos in carsPositions:
            self.nextPos = self.pos
            if (self.movement == "TL" or self.movement == "TR") and move == "F":
                self.nextSteps.insert(0, "F")
        else:
            self.numMovimientos += 1
        
        self.sendSignal()
            

    def advance(self):
        '''
        Define el nuevo estado calculado del método step.
        '''
        self.model.grid.move_agent(self, self.nextPos)
   

 
class TrafficLight(Agent):
    '''
    Representa a un semáforo con los tres colores verde, amarillo y rojo.
    '''
    GREEN = 0.3
    YELLOW = 0.52
    RED = 0.7
    
    def __init__(self, unique_id, model):
        '''
        Agente representa un semáforo, se inicializa con un id, 
        con el status de amarillo y una lista de señales vacía.
        '''
        super().__init__(unique_id, model)
        self.status = self.YELLOW
        self.signals = []
        
    def addToQueue(self):
        '''
        Este método se utiliza para agregar el semáforo a la fila de espera en caso de que reciba señales,
        o quitarse en caso de que ya no tenga señales. Asimismo, si no tiene señales se agrega a la lista 
        del modelo que almacena los semáforos libres.
        '''
        idxSelf = self.model.findTrafficLightIndex(self.unique_id) 
        if len(self.signals) > 0:
            if not self.unique_id in self.model.queueTrafficLights:
                self.model.queueTrafficLights.append(self.unique_id)
            if self.unique_id in self.model.freeTrafficLights:
                self.model.freeTrafficLights.remove(self.unique_id)
        else:
            if idxSelf in self.model.trafficLightsOn:
                self.model.trafficLightsOn.remove(idxSelf)
            if self.unique_id in self.model.queueTrafficLights:
                self.model.queueTrafficLights.remove(self.unique_id)
            if not self.unique_id in self.model.freeTrafficLights:
                self.model.freeTrafficLights.append(self.unique_id)
    
    
    def updateSignals(self):
        '''
        Este método se utiliza para actualizar las señales de los semáforos.
        '''
        idxSelf = self.model.findTrafficLightIndex(self.unique_id) 

        for car in self.model.cars:
            idx, dist = car.closestTrafficLight()
            if car.unique_id in self.signals and not self.model.carInIntersection(car):
                if idx != idxSelf :
                    self.signals.remove(car.unique_id)

                if idx == idxSelf:
                    if car.direction == "U" and car.orientation == "H" and car.pos[1] > self.pos[1]:
                        self.signals.remove(car.unique_id)
                    elif car.direction == "D" and car.orientation == "H" and car.pos[1] < self.pos[1]:
                        self.signals.remove(car.unique_id)
                    elif car.direction == "U" and car.orientation == "V" and car.pos[0] < self.pos[0]:
                        self.signals.remove(car.unique_id)
                    elif car.direction == "D" and car.orientation == "V" and car.pos[0] > self.pos[0]:
                        self.signals.remove(car.unique_id)
                    
        
class StreetModel(Model):
    '''
    Define el modelo de la calle donde se encuentran los autos.
    '''
    def __init__(self, num, width, height):
        self.grid = SingleGrid(width, height, True)
        self.width = height
        self.height = height
        self.midWidth = self.width // 2
        self.midHeight = self.height // 2
        self.schedule = SimultaneousActivation(self)
        self.firstTrafficLightName = ""
        self.maxNumCars = num
        
        self.cars = []
        self.trafficLights = []
        self.queueTrafficLights = []
        self.freeTrafficLights = []
        self.trafficLightsOn = []
        
        
        #Posicionar semáforos 

        a = TrafficLight("TF1", self)
        self.grid.place_agent(a, (self.midWidth - 2, self.midHeight - 2))
        self.schedule.add(a)
        self.trafficLights.append(a)
        
        
        b = TrafficLight("TF2", self)
        self.grid.place_agent(b, (self.midWidth - 2, self.midHeight + 1))
        self.schedule.add(b)
        self.trafficLights.append(b)
        
        c = TrafficLight("TF3", self)
        self.grid.place_agent(c, (self.midWidth + 1, self.midHeight + 1))
        self.schedule.add(c)
        self.trafficLights.append(c)
        
        d = TrafficLight("TF4", self)
        self.grid.place_agent(d, (self.midWidth + 1, self.midHeight - 2))
        self.schedule.add(d)
        self.trafficLights.append(d) 
        
        
        #Posicionar autos hasta un máximo de cuatro a la vez
        maxNumCars = min(4, self.maxNumCars)
        numCars = random.randint(1, maxNumCars)
        self.addCars(numCars)
        
        
        
        # Aquí definimos con colector para obtener la calle y el número de movimientos totales.
        self.datacollector = DataCollector(
            model_reporters={'Calle': obtenerCalle},
            agent_reporters={'Movimientos': lambda a: getattr(a, 'numMovimientos', None)},
        )
            
        
    
    def step(self):
        '''
        En cada paso el colector tomará la información que se definió y almacenará el grid para luego graficarlo.
        '''
        #Se actualiza la luz de los semáforos
        self.changeTrafficLight()
        
        self.datacollector.collect(self)
        self.schedule.step()

        #Se agregan la cantidad de autos faltantes de forma paulatina
        carsLeft = self.maxNumCars - len(self.cars)
        if  carsLeft > 0 :
            carsLeft = min(4, carsLeft)
            numCars = random.randint(1, carsLeft)
            self.addCars(numCars)
        

    def addCars(self, n):
        '''
        Este método se utiliza para agregar n cantidad de autos.
        '''
        for n in range(n):
            approved = False
            while not approved:
                pos = random.randint(self.midWidth - 1, self.midWidth)
                orientation = random.choice(["H","V"])
                if pos < self.midWidth:
                    direction = "U" 
                    if orientation == "H":
                        coords = (pos, 0)
                    else: 
                        coords = (self.height - 1, pos)

                else:
                    direction = "D"
                    if orientation == "H":
                        coords = (pos, self.width - 1)
                    else: 
                        coords = (0, pos)
                
                carsPositions = [car.pos for car in self.cars]
                
                if not coords in carsPositions:
                    approved = True
            
            
            a = CarAgent(len(self.cars), self, orientation, direction)
            self.grid.place_agent(a, coords)
            self.schedule.add(a)
            self.cars.append(a)
    
    def intersectionEmpty(self):
        '''
        Este método se utiliza para verificar la intersección está libre.
        '''
        for i in range(self.midWidth - 1, self.midWidth + 1):
            for j in range(self.midHeight - 1, self.midHeight + 1):
                if not self.grid.is_cell_empty((i,j)):
                    return False
        return True
    
    def carInIntersection(self, car):
        '''
        Este método se utiliza para verificar si un coche se encuentra o no en la intersección.
        '''
        for i in range(self.midWidth - 1, self.midWidth + 1):
            for j in range(self.midHeight - 1, self.midHeight + 1):
                if car.pos == (i,j):
                    return True
        return False
                
    def findTrafficLightIndex(self, nameTrafficLight):
        '''
        Este método se utiliza para encontrar el indice de un semáforo en la lista de semáforos.
        '''
        for i in range(len(self.trafficLights)):
                if self.trafficLights[i].unique_id == nameTrafficLight:
                    return i

    def changeTrafficLight(self):
        '''
        Este método se utiliza para realizar el cambio de los semáforos.
        '''
            
        #Se actualiza la lista de señales del semáforo
        for idxTF in range(len(self.trafficLights)):
            
            self.trafficLights[idxTF].updateSignals()
            self.trafficLights[idxTF].addToQueue()
            
            if idxTF in self.trafficLightsOn:
                self.trafficLights[idxTF].status = self.trafficLights[idxTF].GREEN
            elif self.trafficLights[idxTF].unique_id in self.queueTrafficLights:
                self.trafficLights[idxTF].status = self.trafficLights[idxTF].RED
            elif self.trafficLights[idxTF].unique_id in self.freeTrafficLights:
                self.trafficLights[idxTF].status = self.trafficLights[idxTF].YELLOW
        
        #Se cambian a verde los semáforos correspondientes
        if len(self.queueTrafficLights) > 0:
            if len(self.trafficLightsOn) == 0:
                self.firstTrafficLightName = self.queueTrafficLights[0]
                idx = self.findTrafficLightIndex(self.firstTrafficLightName)
                self.trafficLights[idx].status = self.trafficLights[idx].GREEN
                self.trafficLightsOn.append(idx)

            else:
                idx = self.trafficLightsOn[0]

            modTF = idx % 2
            if modTF == idx:
                idxOther = idx + 2
            else:
                idxOther = idx - 2
            if self.trafficLights[idxOther].unique_id in self.queueTrafficLights:
                self.trafficLights[idxOther].status = self.trafficLights[idxOther].GREEN
                if not idxOther in self.trafficLightsOn:
                    self.trafficLightsOn.append(idxOther)
                    
            

# A continuación se corre el modelo:

# Definimos el tamaño del Grid
M = 12
N = 12

# Definimos el número de agentes
NUM_AGENTS = 4

# Definimos tiempo máximo (segundos)
MAX_TIME = 0.5

# Registramos el tiempo de inicio y corremos el modelo
num_generations = 0

start_time = time.time()
model = StreetModel(NUM_AGENTS, M, N)
while time.time() - start_time < MAX_TIME:
    num_generations += 1
    model.step()
    


# Obtenemos la información que almacenó el colector, este nos entregará un DataFrame de pandas que contiene toda la información.


all_grid = model.datacollector.get_model_vars_dataframe()


# Graficamos la información usando matplotlib


fig, axs = plt.subplots(figsize=(7,7))
axs.set_xticks([])
axs.set_yticks([])
patch = plt.imshow(all_grid.iloc[0][0], cmap=plt.colormaps["gist_ncar"])

def animate(i):
    patch.set_data(all_grid.iloc[i][0])
    
anim = animation.FuncAnimation(fig, animate, frames=num_generations)


plt.show()

