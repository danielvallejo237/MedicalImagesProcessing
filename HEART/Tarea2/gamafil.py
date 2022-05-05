'''
Implementación de los gaussian matched filters propuestos por chadhuri 
Implementación del método propuesto por 
'''

# Estos son los parámetros que se consideraron óptimos para el algoritmo de Chaudhuri
from array import array
from statistics import mean
import numpy as np
import math
import cv2

T = 13
L = 9
SIGMA = 2.0
κ = 12 
THETA = 15

def Equation32(x)-> float:
    return -np.exp(-1*(x**2)/(2*SIGMA**2))

def InitTemplate(shape=(15,15))-> array:
    Init=np.zeros(shape) #Plantilla inicial
    center=(shape[0]//2,shape[0]//2) #Se asume que la matriz tiene el centro en el origen
    Tvalues=np.arange(-1*(T//2),(T//2)+1,1,dtype=int)
    Ystart=center[1]-L//2
    #print(Ystart)
    Yvalues=np.arange(Ystart,L+Ystart,1,dtype=int)
    new_values=[]
    for t in Tvalues:
        new_values.append(Equation32(t))
    new_values=new_values-mean(new_values) #Paso de normalización
    new_values=np.asarray(new_values)
    new_values=new_values*10
    new_values=new_values.astype(int) #Necesitamos números enteros
    centerrow=len(new_values)//2
    centroPlantilla=center[0]
    new_value=np.asarray(new_values)
    for y in Yvalues:
        Init[y][centroPlantilla-centerrow:centroPlantilla+centerrow+1]=new_values
    return Init

def Rotate(iteration=0,angulo=THETA,shape=(15,15))->array:
    template=InitTemplate(shape=shape)
    degrees=iteration*angulo #Hay rotaciones de cada 15 grados
    (h, w) = template.shape[:2]
    (cX, cY) = (shape[0] //2 , shape[1]//2)
    M=cv2.getRotationMatrix2D((cX, cY),degrees, 1.0)
    PlantillaRotada =cv2.warpAffine(template, M, (w, h))
    return PlantillaRotada

def ApplyConv(image,kernel)->array:
    resulting_image = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    media=int(np.mean(resulting_image))
    ret,thresh1 = cv2.threshold(resulting_image,media,255,cv2.THRESH_BINARY)
    return thresh1

def GMF(image,rotations=12)->array:
    imagen=cv2.imread(image,0)
    imagenes=[]
    for i in range(rotations):
        Kernel=Rotate(i)
        resultante=ApplyConv(imagen,Kernel)
        imagenes.append(resultante)
    imagenes=np.asarray(imagenes)
    imagenes=np.max(imagenes,axis=0)
    return imagenes
