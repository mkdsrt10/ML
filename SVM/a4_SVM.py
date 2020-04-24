import numpy as np
import pandas as pd
import sys

trainfile = sys.argv[1]
testfile = sys.argv[2]
testpred = sys.argv[3]

data = np.array(pd.read_csv(trainfile, header=None))
test = np.array(pd.read_csv(testfile, header=None))

def randarr(arr, k):
    idx = np.random.choice(arr.shape[0], k, replace=False)
    return arr[idx, :]

def getPartialdataset(arr, n, k):
    idx = arr[np.in1d(arr[:,-1], [n, k])]
    for items in idx:
        if items[-1] == n:
            items[-1] = -1
        if items[-1] == k:
            items[-1] = 1
    return idx

def getposarr(arr, w):
    temp = (np.dot(arr[:, :-1], w))*(arr[:, -1])
    return arr[np.where(temp < 1)]

def pegasos(dataset, batch_size = 100, max_iter = 2000):
    w = np.zeros(784)
    b = 0
    # tem = 1 - ((np.dot(dataset[:, :-1], w) + b) * dataset[:, -1])
    # tem[tem < 0] = 0
    # loss = (1/batch_size)*np.sum(tem)
    # print("loss -> ",loss)
    for iterr in range(max_iter):
        batch = randarr(dataset, batch_size)
        s_w = np.zeros(w.size)
        s_b = 0
        for items in batch:
            x = items[:-1]
            y = items[-1]
            if y * (np.dot(w,x) + b) < 1:
                s_w += -y*x
                s_b += -y
        g_w = (1/(iterr + 1))*(w + (1/batch_size)*s_w)
        g_b = (1/(iterr + 1))*(1/batch_size)*s_b
        w = w - g_w
        b = b - g_b
        # tem = 1 - ((np.dot(dataset[:, :-1], w) + b) * dataset[:, -1])
        # tem[tem < 0] = 0
        # loss = (1/batch_size)*np.sum(tem)
        # print(iterr, " loss -> ",loss)
    err = np.matmul(w.T, w) / 2 - (1/batch_size)*(s_w.sum())
    print(iterr, err)
    # tem = 1 - ((np.dot(dataset[:, :-1], w) + b) * dataset[:, -1])
    # tem[tem < 0] = 0
    # loss = (1/batch_size)*np.sum(tem)
    # print("loss -> ",loss)
    return w,b

def train(trainarr):
    multiclass_weights = [[() for j in range(i+1, 10)] for i in range(10)]
    for i in range(10):
        for j in range(i+1, 10):
            # print(i, j, "----------------------------------------------------------")
            multiclass_weights[i][j-i-1] = pegasos(getPartialdataset(trainarr, i, j))
    return multiclass_weights
np.random.seed(1)
s = train(data)

def predict(data, w_arr):
    win_arr = np.zeros(10)
    for i in range(10):
        for j in range(i+1, 10):
            w,b = w_arr[i][j-i-1]
            if np.dot(w, data[:-1]) + b < 0:
                win_arr[i] += 1
            else:
                win_arr[j] += 1
    return np.argmax(win_arr)

predict_arr = np.zeros(test.shape[0])
for ind in range(test.shape[0]):
    predict_arr[ind] = predict(test[ind], s)

np.savetxt(testpred, predict_arr)
