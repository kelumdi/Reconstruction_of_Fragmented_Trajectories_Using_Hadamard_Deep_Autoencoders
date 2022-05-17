
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback

seed= 1
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
            for i3, e3 in enumerate(e2):
                if i3 % 2 == 0:
                    arr2[2 * i][i2] = e3
                else:
                    arr2[2 * i + 1][i2] = e3
    return (arr2)

def twenty_201_TO_10_201_2_list(twoDim):
    threeDim = []
    for i in range(0, (twoDim.shape[0] - 1)):
        if i % 2 == 0:
            twoCol = np.transpose(twoDim[i:i + 2, :])
            threeDim.append(twoCol)
    return (threeDim)

################################################################################
#https://stackabuse.com/autoencoders-for-image-reconstruction-in-python-and-keras/
################################################################################
def build_autoencoder2(img_shape, l1,l2,code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Flatten())
    #encoder.add(Dense(code_size))
    encoder.add(Dense(l1))
    encoder.add(Dense(l2))
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(l2))
    decoder.add(Dense(l1))
    decoder.add(Dense(np.prod(img_shape))) # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
    decoder.add(Reshape(img_shape))
    return encoder, decoder

def trainData_201_2(inputDF,labelDF,hadaDF):#inputSize=100,
    inputL = twenty_201_TO_10_201_2_list(inputDF)
    labelL = twenty_201_TO_10_201_2_list(labelDF)
    hadaL = twenty_201_TO_10_201_2_list(hadaDF)
    threeList = [(e, e2,e3) for i, e in enumerate(inputL) for j, e2 in enumerate(labelL) for k, e3 in enumerate(labelL) if i == j==k]
    return (inputL,labelL,hadaL,threeList)

def getData(inputDF,labelDF,hadaDF, timeStepFirst):
    (y1, y2, y3,y123) = trainData_201_2(inputDF, labelDF,hadaDF)

    Y = y2
    X = y1
    Z=y3
    XYZ = y123
    train_x = X[0:int(len(XYZ) * (5/5))]
    train_y = Y[0:int(len(XYZ) * (5 / 5))]
    train_z = Z[0:int(len(XYZ) * (5 / 5))]
    return (np.array(train_x),np.array(train_y),np.array(train_z))

def custom_mse(y_true, y_pred, indi):
    diff = y_pred - y_true
    KK = K.mean(K.square(diff*indi) , axis=-1)
    return KK

def trainAEKerasMSE_adam(hadaTrue, l1,l2,codeSize,batchSizeTrain,tol,trainX, trainY, indicator,n_epochs,lrAE):

    IMG_SHAPE_x = trainX.shape[1:]  # example [32,32,3] --> [231,2]
    IMG_SHAPE_y = trainY.shape[1:]
    IMG_SHAPE_h = indicator.shape[1:]

    encoder, decoder = build_autoencoder2(IMG_SHAPE_x, l1,l2,codeSize)
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

    history1 = autoencoder.fit(x=(trainX, trainY, indicator), y=trainY, epochs=n_epochs, batch_size=batchSizeTrain,
                               shuffle=True, callbacks=callbacks)

    output1 = autoencoder.predict((trainX, trainY, indicator))
    output2 = ten_201_2_TO_20_201_numpy_long(output1)
    return (history1.history,output2)

def main():
    columnDelete=2

    agent = 20
    corr= 90

    batchSize = 20

    hadaTrue = 'True'

    (l1, l2, codeSize) = (128,64,32)

    n_epochs = 1 #

    tol=0

    timeStepFirst = 'True'

    lrAE = 0.001
    pathCSV = './data/{}agent_{}missing_deleteColumn{}.csv'.format(agent, corr,columnDelete)
    mat1 = pd.read_csv(pathCSV, sep=',',header=None).to_numpy()

    missingDFNoNaN  = np.nan_to_num(mat1)
    ones = np.ones(missingDFNoNaN.shape)
    hadamard = np.where(missingDFNoNaN == 0, 0, ones)

    missingDFNoNaN2 = missingDFNoNaN.copy()

    data = missingDFNoNaN2
    label = missingDFNoNaN2

    (trainX, trainY, hada) = getData(data, label,hadamard,timeStepFirst)
    history, output = trainAEKerasMSE_adam(hadaTrue, l1,l2,codeSize, batchSize, tol, trainX, trainY, hada,
                                       n_epochs, lrAE)

    np.savetxt("vicsek_per{}_epoch{}_{}agent_{}batch.csv".format(corr,n_epochs,agent,batchSize), output, delimiter=",")
    hist_df = pd.DataFrame(history)
    hist_csv_file = './history_per{}_epoch{}_{}agent_{}batch.csv'.format(corr, n_epochs,agent,batchSize) #
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    minLoss = min(hist_df['loss'])
    ee = hist_df.index[hist_df['loss'] == minLoss].tolist()
    print('minLoss:,',minLoss)
    print('its index:',ee)
    print(-1)

if __name__ == "__main__":
    # execute only if run as a script
    main()