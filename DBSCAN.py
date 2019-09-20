import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt
 
# 数据集：TestData.txt，目的：分析学生上网时间并绘直方图
# 参数讲解地址：https://www.jianshu.com/p/b004861105f4 
mac2id=dict()
onlinetimes=[]
f=open('d:\hello,world!\Project\Clustering\TestData.txt',encoding='utf-8')
for line in f:
    mac=line.split(',')[2]
    onlinetime=int(line.split(',')[6])
    starttime=int(line.split(',')[4].split(' ')[1].split(':')[0])
    # mac2id 是一个字典，对应的是每个mac地址的id编号，即0,1,2....
    if mac not in mac2id:
        mac2id[mac]=len(onlinetimes)
        onlinetimes.append((starttime,onlinetime))
    else:
        onlinetimes[mac2id[mac]]=[(starttime,onlinetime)]
# reshape 能够按给出参数重新构造矩阵，其中一个参数为-1时代表以第2个参数为准构造
real_X=np.array(onlinetimes).reshape((-1,2))
 
X=real_X[:,0:1]

# fit函数用来调用训练的数据集 
db=skc.DBSCAN(eps=0.01,min_samples=20).fit(X)
labels = db.labels_
 
print('Labels:')
print(labels)
raito=len(labels[labels[:] == -1]) / len(labels)
print('Noise raito:',format(raito, '.2%'))
 
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
 
print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels))
 
for i in range(n_clusters_):
    print('Cluster ',i,':')
    print(list(X[labels == i].flatten()))

# 绘制直方图，同时显示该图     
plt.hist(X,24)
plt.show()