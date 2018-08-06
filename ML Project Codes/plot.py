
import matplotlib.pyplot as plt
import numpy as np

plt.figure()
lw = 2
list=[]
lamda=[10,30,50,100,150,175,200]
one=[0.9242,0.9228,0.9219,0.9207,0.9203,0.9204,0.9205]
two=[0.9173,0.9136,0.9106,0.9059,0.904,0.904,0.9043]

g=[0.1,0.01]
list.append(one)
list.append(two)



#colors = ['aqua', 'darkorange', 'cornflowerblue','black', 'blue','green']
colors = ['darkorange','blue']
for i,col in zip(range(0,len(list)),colors):
    plt.plot(lamda,list[i],lw=lw,label='gamma {0}'
             ''.format(g[i]))

hist=np.arange(3)
plt.ylim([.894,.93])

plt.xlabel('No. of factors')
plt.ylabel('RMSE')
plt.title('SVD')
plt.legend(loc="lower right")
plt.show()


