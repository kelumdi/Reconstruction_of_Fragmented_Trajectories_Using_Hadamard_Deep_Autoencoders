
import numpy as np
import math
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer, BatchNormalization
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback

seed=1
np.random.seed(seed)

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

def twenty_201_TO_10_201_2_array(twoDim):  # inputSize=100,
    threeDim = []
    for i in range(0, (twoDim.shape[0] - 1)):  # i = 0,...,18
        if i % 2 == 0:
            twoCol = np.transpose(twoDim[i:i + 2, :])
            threeDim.append(twoCol)
    return (np.array(threeDim))

def ten_201_2_TO_20_201_numpy_long(arr):
    arr2 = np.zeros([2 * arr.shape[0], arr.shape[1]])
    for i, e in enumerate(arr):
        for i2, e2 in enumerate(e):
            # if i2 % 2 == 0:
            for i3, e3 in enumerate(e2):
                if i3 % 2 == 0:
                    arr2[2 * i][i2] = e3
                else:
                    arr2[2 * i + 1][i2] = e3
    return (arr2)

################################################################################
#https://stackabuse.com/autoencoders-for-image-reconstruction-in-python-and-keras/
#################################################################################
def build_autoencoder2(batchNor,img_shape, l1,l2,code_size):
    '''
    model = Sequential
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    '''
    if batchNor==True:
        # The encoder
        encoder = Sequential()
        encoder.add(InputLayer(img_shape))
        encoder.add(Flatten())
        #encoder.add(Dense(code_size))
        encoder.add(Dense(l1))
        encoder.add(BatchNormalization())
        encoder.add(Dense(l2))
        encoder.add(BatchNormalization())
        encoder.add(Dense(code_size))
        encoder.add(BatchNormalization())

        # The decoder
        decoder = Sequential()
        decoder.add(InputLayer((code_size,)))
        decoder.add(Dense(l2))
        decoder.add(BatchNormalization())
        decoder.add(Dense(l1))
        decoder.add(BatchNormalization())
        decoder.add(Dense(np.prod(img_shape))) # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
        decoder.add(BatchNormalization())
        decoder.add(Reshape(img_shape))
    else:
        # The encoder
        encoder = Sequential()
        encoder.add(InputLayer(img_shape))
        encoder.add(Flatten())
        # encoder.add(Dense(code_size))
        encoder.add(Dense(l1))
        encoder.add(Dense(l2))
        encoder.add(Dense(code_size))

        # The decoder
        decoder = Sequential()
        decoder.add(InputLayer((code_size,)))
        decoder.add(Dense(l2))
        decoder.add(Dense(l1))
        decoder.add(
            Dense(np.prod(img_shape)))  # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
        decoder.add(Reshape(img_shape))
    return encoder, decoder


def trainData_201_2_train(inputDF,labelDF,hadaDF):#inputSize=100,

    inputL = [twenty_201_TO_10_201_2_array(e) for i,e in enumerate(inputDF)]

    labelL = [twenty_201_TO_10_201_2_array(e) for i, e in enumerate(labelDF)]

    hadaL = [twenty_201_TO_10_201_2_array(e) for i, e in enumerate(hadaDF)]

    threeList = [(e, e2,e3) for i, e in enumerate(inputL) for j, e2 in enumerate(labelL) for k, e3 in enumerate(labelL) if i == j==k]

    return (inputL,labelL,hadaL,threeList)

def getDataMovingWin(inputDF,labelDF,hadaDF):
    #inputDF = labelDF = hadaDF = [8,500]
    inputDF_T = inputDF.copy().T
    inputL=[]
    for i,e in enumerate(inputDF_T):
        if i<400:
            input100 = inputDF_T[(1*i):(100+i),:].T #[8,100]
            inputL.append(input100)

    labelDF_T = labelDF.copy().T
    labelL=[]
    for i,e in enumerate(labelDF_T):
        if i<400:
            label100 = labelDF_T[(1*i):(100+i),:].T#[8,100]
            labelL.append(label100)

    hadaDF_T = hadaDF.copy().T
    hadaL=[]
    for i,e in enumerate(hadaDF_T):
        if i<400:
            hada100 = hadaDF_T[(1*i):(100+i),:].T#[8,100]
            hadaL.append(hada100)
    return(inputL,labelL,hadaL)

#(trainX, trainY) =getData(data, label,hadamard20Row,timeStepFirst)
def getDataTrain(inputDF,labelDF,hadaDF, timeStepFirst):
    (inputL, labelL, hadaL) = getDataMovingWin(inputDF,labelDF,hadaDF)

    if timeStepFirst == True: #THIS
        (y1, y2, y3,y123) = trainData_201_2_train(inputL, labelL,hadaL)

    Y = y2  # (1)12k
    X = y1
    Z=y3

    # #case1 :
    XYZ = y123  # (1)1200

    # (1)inputSize=1 : 1000 (2)inputSize=10 : 100 (3)inputSize=x :
    train_x = X[0:int(len(XYZ) * (5/5))]  # =20
    train_y = Y[0:int(len(XYZ) * (5 / 5))]  # =20
    train_z = Z[0:int(len(XYZ) * (5 / 5))]
    return (np.array(train_x),np.array(train_y),np.array(train_z))

def custom_mse(y_true, y_pred, indi):
    diff = y_pred - y_true
    KK = K.mean(K.square(diff*indi) , axis=-1)#[?,231]
    return KK

def calculateOverlap(output3,lengthOriginal,lengthMoving, numTraj):
    overlapK= np.array(output3).copy()
    AA=[]
    for i in range(0,lengthOriginal):
        #         ############################################
        #         # step1 : From ... To (i=99 & 100overlap),
        #                 From ... To (i=2 & 3overlap)
        #         #############################################
        if i < lengthMoving:
            l1 = np.linspace(0, i, i+1)
            l2 = reversed(l1)
            AA1 = []
            for p, q in zip(l1, l2):
                overlapped = overlapK[int(p),:,int(q)]
                AA1.append(overlapped)
            AA.append(AA1)
        #         ####################################################
        #         #step2 : From (i=100 & 100overlap) To (i=399 & 100overlap),
        #         ####################################################
        #elif (i >= lengthMoving) and (i < numTraj):
        elif (i >= lengthMoving) and (i < numTraj-1):
            l1 = np.linspace(i - lengthMoving+1, i, lengthMoving)
            l21=np.linspace(0,lengthMoving-1,lengthMoving)
            l2=reversed(l21)
            AA2 = []
            for p, q in zip(l1, l2):
                overlapped = overlapK[int(p),:,int(q)]#[8,]
                AA2.append(overlapped)
            AA.append(AA2)
        #         ##########################################################
        #         ##step3 : from (i=400 & 100overlap) to (i=499 & 1overlap)
        #         ##########################################################
        elif (i >= numTraj-1):
            if numTraj < 10:
                l1=np.linspace(i-lengthMoving+1,numTraj-1,lengthOriginal-i)#
                l21=np.linspace(i-numTraj+1,lengthMoving-1,lengthOriginal-i)
            else:
                l1=np.linspace(i-lengthMoving,numTraj-1,lengthOriginal-i)#
                l21=np.linspace(i-numTraj,lengthMoving-1,lengthOriginal-i)
            l2= reversed(l21)
            AA3 = []
            for p, q in zip(l1, l2):
                overlapped = overlapK[int(p),:,int(q)]
                AA3.append(overlapped)
            AA.append(AA3)
    meanL=[]
    for i,e in enumerate(AA):
        arrays = [np.array(x) for x in e]
        meanTime = [np.mean(k) for k in zip(*arrays)]
        meanL.append(meanTime)
    meanL = np.array(meanL)
    return(meanL)


def trainAEKerasMSE(batchNor,hadaTrue, l1,l2,codeSize,batchSizeTrain,tol,trainX, trainY, indiTrain,testX,testY,indiTest,n_epochs,lrAE):

    IMG_SHAPE_x = trainX.shape[1:]
    IMG_SHAPE_y = trainY.shape[1:]
    IMG_SHAPE_h = indiTrain.shape[1:]

    encoder, decoder = build_autoencoder2(batchNor,IMG_SHAPE_x, l1,l2,codeSize)

    input1 = Input(IMG_SHAPE_x)
    true = Input(IMG_SHAPE_y)
    hada = Input(IMG_SHAPE_h)
    code = encoder(input1)

    output1 = decoder(code)
    autoencoder = Model([input1,true, hada], output1)

    autoencoder.add_loss(custom_mse(true, output1, hada))

    opt = tensorflow.keras.optimizers.Adam(lr=lrAE)

    if hadaTrue == 'True':
        autoencoder.compile(optimizer=opt,loss=None)
    else:
        autoencoder.compile(optimizer=opt, loss='mse')

    callbacks = [
        EarlyStoppingByLossVal(monitor='loss', value=tol, verbose=1),
    ]

    history1 = autoencoder.fit(x=(trainX, trainY, indiTrain), y=trainY, epochs=n_epochs, batch_size=batchSizeTrain,
                               shuffle=True, callbacks=callbacks)

    output1 = autoencoder.predict((testX, testY, indiTest))
    output2=[]
    for i,e in enumerate(output1):
        e2 = ten_201_2_TO_20_201_numpy_long(e)
        output2.append(e2)

    output3 = calculateOverlap(output2,500,100,400)
    return (history1.history,output3.T)


def main():
    agent = 4
    corr= 75
    n_epochs = 1
    tol= 0

    batchNor=False
    batchSize = 1

    hadaTrue = 'True'
    (l1, l2, codeSize) = (128, 64, 32)

    timeStepFirst = True

    lrAE = 0.001

    pathCSV = './data/robot500_{}agent_{}missing.csv'.format(agent, corr)
    mat1 = pd.read_csv(pathCSV, sep=',',header=None).to_numpy()

    missingDFNoNaN  = np.nan_to_num(mat1)
    ones = np.ones(missingDFNoNaN.shape)
    hadamard = np.where(missingDFNoNaN == 0, 0, ones)

    missingDFNoNaN2 = missingDFNoNaN.copy()
    data = missingDFNoNaN2
    label = missingDFNoNaN2
    (trainX, trainY, hadaTrain) = getDataTrain(data, label,hadamard,timeStepFirst)
    testX = trainX
    testY =trainY
    hadaTest = hadaTrain
    history, output = trainAEKerasMSE(batchNor,hadaTrue, l1,l2,codeSize, batchSize, tol, trainX, trainY,hadaTrain, testX,testY,hadaTest,
                                       n_epochs, lrAE)

    np.savetxt("robot500_per{}_epoch{}_{}agent_{}batch_movingWin.csv".format(corr,n_epochs,agent,batchSize), output, delimiter=",")

    hist_df = pd.DataFrame(history)
    hist_csv_file = './history_robot500_per{}_epoch{}_tol_{}agent_{}batch_movingWin.csv'.format(corr, n_epochs,agent,batchSize) #
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    minLoss = min(hist_df['loss'])
    ee = hist_df.index[hist_df['loss'] == minLoss].tolist()
    print(-1)

if __name__ == "__main__":
    # execute only if run as a script
    main()
