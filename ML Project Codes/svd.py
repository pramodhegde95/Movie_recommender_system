import pandas as pd
import numpy as np
import math


ratings=pd.read_csv('D:/MS/ML/ml-latest-small/ratings.csv')
del ratings['timestamp']
movies=pd.read_csv('D:/MS/ML/ml-latest-small/movies.csv')
final_dataset=pd.merge(ratings,movies,on='movieId')
utility_matrix=pd.pivot_table(final_dataset, values='rating', index='movieId', columns='userId', aggfunc='mean', fill_value=0, margins=False, dropna=True, margins_name='All')
no_users=len(ratings['userId'].unique())
no_movies=len(ratings['movieId'].unique())
no_features=len(final_dataset['genres'].unique())

train_rows=3400
train_col=240
um=np.array(utility_matrix)
meu=0
count=0
for i in range(um.shape[0]):
    for j in range(um.shape[1]):
        if not (i>train_rows and j>train_col):
            if(um[i][j]!=0):
                meu+=um[i][j]
                count+=1
                      
meu/=count          
lamda=0.5
gamma=0.01
#.01,.005
no_factors=175
p=np.full((um.shape[0],no_factors),0.1)
#p=np.random.rand(um.shape[0],no_factors)
q=np.full((um.shape[1],no_factors),0.1)
#q=np.random.rand(um.shape[1],no_factors)
bi=np.zeros(um.shape[0])
bu=np.zeros(um.shape[1])

for k in range(10):
    for i in range(um.shape[0]):
        for j in range(um.shape[1]):
            if not (i>train_rows and j>train_col):
                if(um[i][j]!=0):
                    rui_hat=meu+bi[i]+bu[j]+np.dot(p[i],q[j])
                    e=um[i][j]-rui_hat
                    bi[i]+=gamma*(e-(lamda*bi[i]))
                    bu[j]+=gamma*(e-(lamda*bu[j]))
                    #del_q=gamma*((e*p[i])-(lamda*q[j]))
                    #del_p=gamma*((e*q[j])-(lamda*p[i]))
                    q[j]+=gamma*((e*p[i])-(lamda*q[j]))
                    p[i]+=gamma*((e*q[j])-(lamda*p[i]))
                    
                    
                    
                    
err=0
counter=0
for i in range(train_rows,um.shape[0]):  
  for j in range(train_col,um.shape[1]):
      if(um[i][j]!=0):
          pred=um[i][j]-(meu+bi[i]+bu[j]+np.dot(p[i],q[j]))
          err+=math.pow(pred,2)
          #err+=abs(pred)
          counter+=1
                    
rmse=math.sqrt(err/counter)  
#print('rmse: ',round(err/counter,2))
print('svd with factors:',no_factors,round(rmse,4))                    