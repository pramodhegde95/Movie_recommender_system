import pandas as pd
import numpy as np
import math
import heapq

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
            
#non_zeros=np.count_nonzero(um)            
meu/=count          
lamda=0.05
gamma=0.001


bi=np.zeros(um.shape[0])
bu=np.zeros(um.shape[1])

for k in range(10):
    for i in range(um.shape[0]):
        for j in range(um.shape[1]):
            if not (i>train_rows and j>train_col):
                if(um[i][j]!=0):
                    rui_hat=meu+bi[i]+bu[j]
                    e=um[i][j]-rui_hat
                    bi[i]+=gamma*(e-(lamda*bi[i]))
                    bu[j]+=gamma*(e-(lamda*bu[j]))
                    

def movieavg(um):
    avg=np.zeros(um.shape[0])
    for i in range(um.shape[0]):
        if i<train_rows:
            count=np.count_nonzero(um[i])
            avg[i]=sum(um[i])/count
        else:
            count=np.count_nonzero(um[i,0:train_col])
            total=sum(um[i,0:train_col])
            if(total==0 or count==0):
                avg[i]=0
            else:
                avg[i]=total/count
    return avg    

           
def correlation(um,i,j,r):
    numerator=0
    denomi_1=0
    denomi_2=0
    if i>=train_rows or j>=train_rows:
        col_len=train_col
    else:
        col_len=um.shape[1]
    for k in range(col_len):
        if um[i][k]!=0 and um[j][k]!=0:
            x=(um[i][k]-r[i])
            y=(um[j][k]-r[j])
            numerator+=x*y 
            denomi_1+=math.pow(x,2)
            denomi_2+=math.pow(y,2)
    return round(numerator/math.sqrt(denomi_1*denomi_2),4) if denomi_1*denomi_2!=0 else 0        
            
    
def similarity(um,r):
    sim=np.ones((um.shape[0],um.shape[0]))
    for i in range(um.shape[0]):
        for j in range(i+1,um.shape[0]):
                sim[i][j]=correlation(um,i,j,r)
    
    for i in range(1,um.shape[0]): 
        for j in range(0,i):
           sim[i][j]=sim[j][i]        
    return sim

def predict(um,r,x):
    c=0
    error=0
    for i in range(train_rows,um.shape[0]):
        for j in range(train_col,um.shape[1]):
            if um[i][j]!=0:
                #pred=r[i]+KNN(um,x,r,i,j)
                pred=meu+bi[i]+bu[j]+KNN(um,x,r,i,j)
                diff=um[i][j]-pred
                error+=math.pow(diff,2)
                c+=1
    return math.sqrt(error/c)            

                
                
def KNN(um,x,r,i,j):
    a=x[i]
    count=0
    s=heapq.nlargest(2000, range(len(a)), a.take)
    numerator=0
    denominator=0
    for k in range(len(s)):
        if s[k]!=i and count<40 and um[s[k]][j]!=0:
            #numerator+=a[s[k]]*(um[j][s[k]]-r[s[k]])
            numerator+=a[s[k]]*(um[s[k]][j]-(meu+bu[j]+bi[s[k]]))
            denominator+=abs(a[s[k]])
            count+=1
        if count==40:
            break
    return numerator/denominator if denominator!=0 else 0        
            
    
    

# =============================================================================
# r=movieavg(um) 
# x=similarity(um,r)
# np.save('item_item_movie_avg',r)
# np.save('item_item_movie_sim',x)
# =============================================================================
# =============================================================================
r=np.load('item_item_movie_avg.npy')
x=np.load('item_item_movie_sim.npy')
# =============================================================================
print(" 100 neighbours ",round(predict(um,r,x),4))    

   
