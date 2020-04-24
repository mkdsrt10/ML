import math
import sys

import pandas as pd
import numpy as np
from scipy import stats

output_class = 10

def getXY(xyf = "train.csv"):
    df = pd.read_csv(xyf, sep=',', header=None)
    xy = df.values
    y = xy[:, -1]
    x = xy[:, :-1]
    return x, y

def pegaosBinary(Ax, Ay, K, T, lmd):
    w = np.ones(Ax.shape[1])
    b = 0
    C=1
    for t in range(1,T+1):
        ranCh = np.random.choice(Ax.shape[0], K, replace=True)
        Axt = Ax[ranCh]
        Ayt = np.reshape(Ay[ranCh], (ranCh.size,))
        Bp = np.matmul(Axt, w)+b
        B = Bp*Ayt
        Axtp = Axt[B<1]
        Aytp = Ayt[B<1]
        n = 1/(lmd*t)
        wp = (1-n)*w
        Assd = Aytp[:,None]
        wpp = np.sum((Assd*Axtp), axis=0)
        w = wp + C*(n/K)*wpp
        b = b + (n*C/K)*np.sum(Aytp)
        err = np.matmul(w.T, w)/2 + C*(wpp.sum())
        print(t, err)
    print("END\n\n")
    sys.argv[6]
    return w, b

def main():
    np.random.seed(1)
    # trainf = sys.argv[1]
    # testf = sys.argv[2]
    # testpred = sys.argv[3]
    trainf = "train.csv"
    testf = "test_public.csv"
    testpred = "pred.txt"
    x, y = getXY(trainf)
    K = 100
    lmd = 1
    T = 2000
    # W = []
    B = [[() for j in range(i + 1, output_class)] for i in range(output_class)]
    W = [[() for j in range(i + 1, output_class)] for i in range(output_class)]
    for i in range(output_class):
        for j in range(i+1, output_class):
            j = 5
            yu = y[(y==i) + (y==j)]
            yi = yu==i
            yj = yi-1
            Ay = yi+yj
            Ay = Ay *(-1)
            # Ay = y[y==i or y==j]
            Ax = x[(y==i) + (y==j)]
            wij, bij = pegaosBinary(Ax, Ay, K, T, lmd)
            W[i][j-i-1] = wij
            B[i][j - i - 1] = bij

    xt, yt = getXY(testf)
    i = 0
    j = 0
    Y = []
    for i in range(output_class):
        for j in range(i+1, output_class):
            yp = np.matmul(xt, W[i][j-i-1]) + B[i][j-i-1]
            ypti = (yp < 0) * i
            yptj = (yp >= 0) * j
            ypt = ypti + yptj
            Y.append(ypt)
    # for k in range(len(B)):
    #     if j < output_class:
    #         j += 1
    #     else:
    #         i = i+1
    #         j = i+1

    yp = stats.mode(Y)
    np.savetxt(testpred, yp[0].reshape(yp[0].size, 1))

if __name__ == "__main__":
    main()