'''
Módulo para crear los elementos estructurantes para aplicar el tophat 
@Author danielvallejo237
'''

from array import array
from random import choices
from shutil import ExecError
import numpy as np
from scipy.spatial import distance
import cv2
from sklearn.metrics import classification_report,accuracy_score,precision_score,jaccard_score,confusion_matrix,roc_auc_score
import logging

def Disk(shape=(5,5),invert=True) -> array:
    '''
    Función para construir el elemento estructurante de tipo disco
    '''
    assert(shape[0] % 2 ==1 and shape[0] == shape[1])
    radio=shape[0] // 2 #División entera
    center=(radio,radio) 
    SE=np.zeros(shape,dtype=np.uint8)
    SE[center]=1
    center=(float(center[0]),float(center[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (float(i)-center[0])**2+(float(j)-center[0])**2 < radio**2:
                SE[i][j]=1
    SE[0,radio-1:SE.shape[1]-radio+1] = 1
    SE[SE.shape[0]-1,radio-1:SE.shape[1]-radio+1]= 1
    SE[radio-1:SE.shape[0]-radio+1,0] = 1
    SE[radio-1:SE.shape[0]-radio+1,SE.shape[1]-1] = 1
    if invert:
        for i in range(shape[0]):
            for j in range(shape[1]):
                SE[i,j]=(SE[i,j]+1) %2
    return SE 

def Diamond(shape=(5,5),invert=True) -> array:
    '''
    Función para construir un elemento estructurante de tipo Diamante
    Nuevamente como en el caso anterior consideraremos puros filtros de forma impar y de igual dimesión x,y
    '''
    assert(shape[0] %2 ==1 and shape[0]==shape[1])
    radio=shape[0] // 2
    center=(radio,radio)
    SE=np.zeros(shape,dtype=np.uint8)
    SE[center]=1
    SE[(0,radio)]=1
    SE[(shape[0]-1,radio)]=1
    SE[(radio,0)]=1
    SE[(radio,shape[1]-1)]=1
    a=0.5*distance.euclidean((radio,0),(radio,shape[1]-1))
    b=0.5*distance.euclidean((0,radio),(shape[0]-1,radio))
    U = (np.asarray([radio,shape[1]-1])-np.asarray([radio,0]))/(2*a);
    V = (np.asarray([shape[0]-1,radio])-np.asarray([0,radio]))/(2*b);
    Q=np.asarray(center)
    for i in range(SE.shape[0]):
        for j in range(SE.shape[1]):
            P=(i,j)
            W = P-Q;
            xabs = abs(np.dot(W,U))
            yabs = abs(np.dot(W,V))
            if xabs/a + yabs/b <= 1:
                SE[i][j]=1
    if invert:
        for i in range(shape[0]):
            for j in range(shape[1]):
                SE[i,j]=(SE[i,j]+1) %2
    return SE


def TopHat(img,se,iterations=1) -> array:
    nimg=cv2.erode(img, se, iterations=iterations)
    nimg=cv2.dilate(nimg,se,iterations=iterations)
    return img-nimg



def EQMethod(source,se,iterations=None,filtro=None,gt=None,read_binary=False) -> array:
    try:
        if read_binary == False:
            img=cv2.imread(source,cv2.IMREAD_GRAYSCALE) #Abrir en escala de grises
        else:
            img=cv2.imread(source,0)
    except:
        raise ExecError("Archivo no encontrado")
    if iterations is not None and type(iterations)==int:
        logging.info('Using iterated TopHat')
        #Usamos top hat iterado
        img=TopHat(img,se=se,iterations=iterations)
    elif iterations is None:
        logging.info('Using default TopHat')
        img=cv2.morphologyEx(img, cv2.MORPH_TOPHAT, se)
    
    local_avg=round(np.mean(img))
    #Los tres tipos de umbralización presentados en la tarea, el filtro de OTSU, el global threshold y el mean 
    #adaptative theshold
    if filtro is not None and filtro not in ['otsu','adapt']:
        #Se reestablece a los valores por defecto en este caso None
        filtro=None
    if filtro is None:
        ret2,th1 = cv2.threshold(img,local_avg,255,cv2.THRESH_BINARY_INV)
    elif filtro=='otsu':
        ret2,th1=cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    elif filtro == 'adapt':
        th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,15,2)
    
    if filtro is not None:
        logging.info('Using '+filtro+ ' as threshold')

    if gt is not None:
        logging.info('Using asisted selection for connected components')
        mask=cv2.imread(gt,0)
        numLabels,labelImage,stats,centroids=cv2.connectedComponentsWithStats(th1, 8, cv2.CV_32S)
        Candidates=np.nonzero(mask)
        choice=np.where(Candidates[0]==np.random.choice(Candidates[0]))[0][0]
        Choice=(Candidates[0][choice],Candidates[1][choice])
        Label=labelImage[Choice]
        try:
            assert(mask[Choice]==255 and th1[Choice]==255)
            Mask=np.zeros_like(th1)
            for i in range(Mask.shape[0]):
                for j in range(Mask.shape[1]):
                    if labelImage[i][j]==Label:
                        Mask[i][j]=255
            th1=Mask
        except:
            pass
    return th1

def EvaluateMetrics(y_pred,y,asdict=True) -> any:
    '''
    Calcular el Acc de predicción 
    '''
    pred=y_pred.reshape(-1,)
    real=y.reshape(-1,)
    acc=accuracy_score(pred,real)
    prec_pos=precision_score(pred,real,pos_label=255)
    tn, fp, fn, tp = confusion_matrix(pred,real).ravel()
    specificity = tn / (tn+fp)
    jaccard=jaccard_score(pred,real,pos_label=255)
    roccurve=roc_auc_score(pred,real)
    if asdict:
        Eval={}
        Eval['acc']=acc 
        Eval['prec']=prec_pos 
        Eval['spec']=specificity
        Eval['jaccard']=jaccard
        Eval['roc']=roccurve
        return Eval
    else:
        return np.round(np.asarray([acc,prec_pos,specificity,jaccard,roccurve]),decimals=3)
if __name__=='__main__':
    pass