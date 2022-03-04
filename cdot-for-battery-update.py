from asyncore import write
import numpy as np
import pandas as pd
import matplotlib.pylab as pl
import ot
import ot.plot
from scipy.misc import derivative
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn import neighbors
import os

# [10 15 25 20 30 35 40 45 50]

method = 'dot'
dir = '/Users/liuhanbing/Desktop/code'
lambd = 0.01
max_epoch = 10
step_size = 0.1
mut = 50
cnt = 9
n0 = 0

if method == 'cdot':
    fo = open(os.path.join(dir, 'cdot/seq-log.txt'), "w")
else:
    fo = open(os.path.join(dir, 'cdot/log.txt'), "w")

for i in range(cnt):

    # load data
    path2 = dir + '/out_SOC_005-075_excel/out-SOC-0{}.xlsx'.format(5*(i+2))
    soc10 = pd.read_excel(path2, engine='openpyxl').values
    label = pd.read_excel(os.path.join(dir, 'cdot/source.xlsx'), engine='openpyxl').values

    soc10_xt = soc10[:, 2 : 23] # (67, 21)
    soc10_yt = soc10[:, 1] # (67, )

    if method == 'cdot':
        print("This is the {}th sequential transport".format(i+1))
        if i == 0:
            path1 = dir + '/out_SOC_005-075_excel/out-SOC-00{}.xlsx'.format(5*(i+1))
            soc5 = pd.read_excel(path1, engine='openpyxl').values

            soc5_xs = soc5[:, 2 : 23] # (67, 21)
            soc5_ys = soc5[:, 1] # (67, )
        
        else:
            path1 = dir + '/out_SOC_005-075_excel/out-SOC-0{}.xlsx'.format(5*(i+1))
            soc5 = pd.read_excel(path1, engine='openpyxl').values

            soc5_xs = soc5[:, 2 : 23] # (67, 21)
            soc5_ys = label[:, 1]
    
    else:
        print("This is the {}th transport".format(i+1))

        path1 = dir + '/out_SOC_005-075_excel/out-SOC-005.xlsx'
        soc5 = pd.read_excel(path1, engine='openpyxl').values

        soc5_xs = soc5[:, 2 : 23] # (67, 21)
        soc5_ys = soc5[:, 1] # (67, )
    

    # loss matrix
    M = ot.dist(soc5_xs, soc10_xt, metric='euclidean') 
    M /= M.max() # (67, 67)
    n1 = soc5_xs.shape[0]
    n2 = soc10_xt.shape[0]
    a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2  # uniform distribution on samples

    if i == 0:

        gamma = ot.sinkhorn(a, b, M, lambd) # (67, 67)

    else:
        
        # transport matrix
        gamma0 = np.random.rand(n1, n2)

        def derivation(gamma):
            tmp1 = 2 * (n1 ** 2) * soc5_xs @ soc5_xs.T @ gamma
            
            tmp2 = (n0 * n1) * soc5_xs @ soc0_x.T @ gamma_t0
            return tmp1 - 2 * tmp2

        gamma = gamma0
        for i in range(max_epoch):
            c = step_size * M + step_size * mut * derivation(gamma) - np.log(gamma)
            c /= c.max()
            gamma = ot.sinkhorn(a, b, c, 1+step_size*lambd)
        
    xt = n1 * gamma.dot(soc10_xt)

    soc0_x = soc5_xs
    gamma_t0 = gamma
    n0 = n1

    # pl.figure(5)

    # pl.imshow(gamma, interpolation='nearest')
    # pl.colorbar()

    # pl.title('OT matrix sinkhorn')


    model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=4)

    score = 0
    def try_different_method(model):
        model.fit(xt, soc5_ys)
        # score = model.score(soc10_xt, soc10_yt)
        result = model.predict(soc10_xt)
        score = r2_score(result, soc10_yt)
        loss = np.mean(np.square(result - soc10_yt))
        # print("score ", score)
        # print("loss ", loss)

        return result, loss, score
        
    result, loss, score = try_different_method(model_RandomForestRegressor)
    minloss = loss
    bestresult = result
    ascore = score

    for i in range(9):  
        result, loss, score = try_different_method(model_RandomForestRegressor)
        if(loss < minloss):
            minloss = loss
            bestresult = result
            ascore = score
            
    print("model score ", ascore)
    print("min loss ", minloss)
    fo.write('{}\t{}\n'.format(minloss, ascore))

    data = pd.DataFrame(bestresult)

    writer = pd.ExcelWriter(os.path.join(dir, 'cdot/source.xlsx'))
    data.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()

    writer.close()

# fo.close()

pl.figure()
pl.imshow(gamma, interpolation='nearest')
pl.colorbar()
pl.title('OT matrix sinkhorn')

y = np.zeros((67,1))
y[:,0] = soc10_yt[:]
plt.figure()
plt.plot(np.arange(67), soc10_yt,'o--',label='true value')
plt.plot(np.arange(67), bestresult,'o--',label='predict value')

plt.grid()
plt.title('score: %f'%score)
plt.title('CDOT 1-{}'.format(cnt+1))
plt.xlabel('Sample')
plt.ylabel('Capacity')
plt.legend()
plt.fill_between(np.arange(67), bestresult.squeeze(), y.squeeze(), alpha=0.2)
plt.ylim([1.25, 2])
plt.show()
