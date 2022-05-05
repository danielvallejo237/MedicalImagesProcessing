from email.mime import image
from pickletools import uint8
import numpy as np
import gamafil as ga
import StructElements as se
import utils
import cv2 
import math 
import random
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score


categorias=np.asarray(range(0,256)).reshape(1,-1)

OhEncoder=OneHotEncoder(categories=categorias,handle_unknown='ignore')

#Usamos la generación automática de templates con búsqueda iterada 

def recupera_ROI(template):
    '''
    Como la búsqueda iterada solo se hace sobre el template gaussiano,
    debemos de recuperar la zona en la que se encuentra
    '''
    image_center= (ga.ANCHO//2,ga.ANCHO//2)
    plantilla=np.zeros((ga.L,ga.T),dtype=int) #Tamaño de la plantilla
    plantilla=template[image_center[0]-ga.L//2:image_center[0] + ga.L//2 +1, image_center[1]-ga.T //2 :image_center[1]+ga.T//2+1].copy()
    return plantilla

def paste_ROI(core,shape=(15,15)):
    '''
    Una vez que hacemos la perturbación y la búsqueda iterada debemos de regresar el 
    nucleo perturbado a su template con valores de padding
    '''
    template=np.zeros(shape,dtype=int)
    image_center=(ga.ANCHO//2,ga.ANCHO//2)
    template[image_center[0]-ga.L//2:image_center[0] + ga.L//2 +1, image_center[1]-ga.T //2 :image_center[1]+ga.T//2+1]=core.copy()
    return template

def Perturb(core,perRate=0.1):
    '''
    Función de perturbación del core, en este caso tenemos una perturbación aleatoria
    '''
    new_core=core.copy()
    for i in range(core.shape[0]):
        for j in range(core.shape[1]):
            decider=np.random.uniform()
            if decider <=perRate:
                new_core[i][j]=random.randrange(-6,6)
    return new_core

def Variacion(core):
    var=core.copy()
    #Perturbamos solamente un pixel o un 1% de nuestro posible candidato
    var=Perturb(var,perRate=0.05)
    return var

def LocalSearch(core,image,ground_thruth,inner_iterations=12):
    '''
    Hay que calcular el area de ROC con la convolucion
    por simplicidad vamos a meter las imágenes concatenadas 
    '''
    localsol=core.copy()
    padded_core=paste_ROI(localsol)
    separadas=ga.CustomGMF(image,padded_core)
    pred=separadas.reshape(-1,1)
    real=ground_thruth.copy().reshape(-1,1)
    #pred=OhEncoder.fit_transform(pred).toarray()
    #real=OhEncoder.fit_transform(real).toarray()
    #print(pred.shape,real.shape)
    initroc=accuracy_score(real,pred)
    for it in range(inner_iterations):
        candiate=Variacion(localsol) #Ligeras variaciones de intensidad dentro del template
        padded_core=paste_ROI(candiate)
        separadas=ga.CustomGMF(image,padded_core)
        pred=separadas.reshape(-1,1)
        real=ground_thruth.copy().reshape(-1,1)
        #pred=OhEncoder.fit_transform(pred).toarray()
        #real=OhEncoder.fit_transform(real).toarray()
        roc=accuracy_score(real,pred)
        if roc>initroc:
            initroc=roc 
            localsol=candiate
            print("Local Improvement detected")
            break
    return initroc,localsol


def ILSAlgorithm(core,image,ground_truth,iterations=100,initer=5):
    solution=core.copy()
    init_roc,solution=LocalSearch(solution,image=image,ground_thruth=ground_truth,inner_iterations=initer)
    #print(init_roc)
    candidate=core.copy()
    loss=[]
    loss.append(init_roc)
    for i in tqdm(range(iterations)):
        candidate=Perturb(candidate,perRate=0.6)
        roc,candidate=LocalSearch(candidate,image=image,ground_thruth=ground_truth,inner_iterations=initer)
        if roc>init_roc:
            print("Global Improvement detected")
            init_roc=roc 
            loss.append(init_roc)
            solution=candidate
    return solution,init_roc,loss
