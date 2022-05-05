'''
Funciones necesarias para procesar los archivos de la base de datos de imágenes coronarias
@Author danielvallejo237
'''
import os
import random


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

def train_test_split(Ltr=100):
    copia=ALLFILES.copy()
    random.shuffle(copia)
    train=copia[:Ltr]
    test=copia[Ltr:]
    return train,test


if __name__=='__main__':
   pass