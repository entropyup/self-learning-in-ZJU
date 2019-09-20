
import numpy as np
from sklearn.cluster import KMeans

# 一个基本的k-means算法的实现，数据集为1999年中国31个省各项消费情况，旨在通过聚类
# 找出消费等级相似的省份，类数目为4  --->数据集名称：city.txt
# 参数讲解地址：https://blog.csdn.net/xiaoQL520/article/details/78269539

def loadData(filePath):
    fr = open(filePath,'r+')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(',')
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1,len(items))])
    return retData,retCityName

if __name__ == '__main__':
    data,cityName = loadData('d:\hello,world!\Project\Clustering\city.txt')
    # 调用KMeans函数
    km = KMeans(n_clusters=4)
    # fit_predict方法用来进行k-means聚类，输出每个数据对应的类别标签（关于标签的列表）
    label = km.fit_predict(data)
    # cluster_centers_ 为每一个类的质心坐标
    expenses = np.sum(km.cluster_centers_,axis=1)
    CityCluster = [[],[],[],[]]
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):
        print('Expenses:%.2f' % expenses[i])
        print(CityCluster[i])
