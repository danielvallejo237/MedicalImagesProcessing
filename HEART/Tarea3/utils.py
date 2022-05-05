'''
Funciones necesarias para procesar los archivos de la base de datos de imágenes coronarias
@Author danielvallejo237
'''
import os
import random
import cv2
import numpy as np


#Cambiar el path d acuerdo a donde se encuentren las imágenes en cada máquina
DBPATH='/home/danielvaal/Documents/FINAL_SEMESTER/Procesamiento de Imágenes Biomédicas/Parte 1/Tarea1/Database_134_Angiograms'

def ProcessPath(DBPATH):
    documents=os.listdir(DBPATH)
    inp=[d for d in documents if not d.endswith('gt.pgm')]
    gt=[''.join(d.split('.')[:-1])+'_gt.pgm' for d in inp]
    assert(len(gt)==len(inp))
    inp=[os.path.join(DBPATH,d) for d in inp]
    gt=[os.path.join(DBPATH,d) for d in gt]
    ALLFILES=tuple(zip(inp, gt))
    return list(ALLFILES) 

ALLFILES=ProcessPath(DBPATH)

def Query(index):
    if index>=0 and index<134:
        return ALLFILES[index]

def train_test_split(Ltr=100,return_valid=False):
    copia=ALLFILES.copy()
    random.shuffle(copia)
    train=copia[:Ltr]
    if return_valid:
        ln=Ltr-10
        aux=train[:ln]
        valid=train[ln:]
        train=aux
    else:
        valid=[]
    test=copia[Ltr:]
    return train,test,valid


def ConcatImages(dataset):
    firstreal=cv2.imread(dataset[0][0],0)
    firstgt=cv2.imread(dataset[0][1],0)
    for i in range(1,len(dataset)):
        are=cv2.imread(dataset[i][0],0) 
        agt=cv2.imread(dataset[i][1],0)
        #if len(set(agt.reshape(-1).tolist())) >2:
        #    _,agt=cv2.threshold(agt,127,255,cv2.THRESH_BINARY)
        firstreal=np.concatenate((firstreal,are),axis=1)
        firstgt=np.concatenate((firstgt,agt),axis=1)
    return firstreal,firstgt


if __name__=='__main__':
   pass