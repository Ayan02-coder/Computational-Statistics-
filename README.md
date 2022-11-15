# Computational-Statistics-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pylab


iris = pd.read_csv("iris.csv")
iris.head()
str(iris)
str(iris)
print(iris.sample(20))
type(iris)

iris.columns
iris.size
iris.shape
iris.describe()
np.mean(iris["sepal.length"])

s = np.std(iris['sepal.width'])
v= np.var(iris["sepal.width"])
print(s,v)

print(iris["variety"].value_counts())

print("data  type:->",type(iris))
print("*********")

print("columns:->",iris.columns)
print("*********")
print("shape:",iris.shape)
print("*********")
print("Size:",iris.size)
print("*********")
print(iris["variety"].value_counts())

iris.mean()
iris.median()
iris.std()
iris.describe()

print("no of samples available for each type") 
print(iris["class"].value_counts())
print("*********")
print("Mean value\n",iris.mean())
print("*********")
print("Mode value \n",iris.mode())
print("*********")
print("Median value \n",iris.median())
print("*********")
print("standard deviation\n",iris.std())
print("*********")
print("Variance value \n",iris.var())

print(iris.describe())

minval= iris["petal.length"].min()
minval
maxval =iris["petal.length"].max()
print(minval,maxval)

col=['sepal_length','sepal_width','petal_length','petal_width','type']

sum_data = iris["petal.length"].sum() 
mean_data = iris["petal.length"].mean() 
median_data = iris["petal.length"].median() 
std_data = iris["petal.length"].std()

  
print("Sum:",sum_data, "\nMean:", mean_data, "\nMedian:",median_data,"\n std", std_data)

min_data=iris["sepallength"].min() 
max_data=iris["sepallength"].max() 
  
print("Minimum:",min_data, "\nMaximum:", max_data)

iris_setosa=iris.loc[iris["variety"]=="Iris-setosa"]
iris_virginica=iris.loc[iris["variety"]=="Iris-virginica"]
iris_versicolor=iris.loc[iris["variety"]=="Iris-versicolor"]

sum_data = iris["sepal.width"].sum() 
mean_data = iris["sepal.width"].mean() 
median_data = iris["sepal.width"].median() 
  
print("Sum:",sum_data, "\nMean:", mean_data, "\nMedian:",median_data)


import matplotlib.pyplot as plt
plt.hist(iris["sepal.length"])

sum_data = iris["sepallength"].sum() 
mean_data = iris["sepallength"].mean() 
median_data = iris["sepallength"].median() 
std_dev = iris["sepallength"].std()
  
print("Sum:",sum_data, "\nMean:", mean_data, "\nMedian:",median_data, "\n Standard deviation", std_dev)

import seaborn as sns
sns.distplot(iris["petal.length"])

import seaborn as sns
x= iris["sepal.length"]
sns.set_style('darkgrid')
sns.distplot(x)

import seaborn as sns

sns.set_style('darkgrid')
sns.distplot(iris["sepal.length"])

plt.hist(iris["sepal.width"])
plt.axvline(iris["sepal.width"].mean(),color= 'r')
plt.axvline(iris["sepal.width"].median(),color= 'g')
plt.xlabel("sepal.width")
plt.ylabel("freq")
plt.title("histogram")

x= iris.sepallength
plt.hist(x)

plt.axvline(x.mean(), color='g', linestyle='dashed', linewidth=2)
plt.axvline(x.median(), color='r', linestyle='dashed', linewidth=1)
plt.axvline(x.std(), color='g', linestyle='dotted', linewidth=1)
plt.title('Histogram for sepal width')
plt.xlabel('sepal width')
plt.ylabel('frequency')
plt.show()

x1= iris.sepallength
plt.hist(x1,range=(x1.min(),x1.max()))
plt.axvline(x1.mean(), color='b', linestyle='dashed', linewidth=2)
plt.axvline(x1.median(), color='r', linestyle='dashed', linewidth=1)
plt.axvline(x1.std(), color='g', linestyle='dotted', linewidth=1)
plt.title('Histogram for sepal length')
plt.xlabel('sepal length')
plt.ylabel('frequency')
plt.show()


x2= iris.petallength
plt.hist(x2,range=(x2.min(),x2.max()))
plt.axvline(x2.mean(), color='b', linestyle='dashed', linewidth=2)
plt.axvline(x2.median(), color='r', linestyle='dashed', linewidth=1)
plt.axvline(x2.std(), color='g', linestyle='dotted', linewidth=1)
plt.title('Histogram for petal length')
plt.xlabel('petal length')
plt.ylabel('frequency')
plt.show()
plt.show()

print("sentosa",np.percentile(iris_setosa["sepal.length"],np.arange(0,100,25)))

print("Quantiles")
print("setosa",np.percentile(iris_setosa["petallength"],np.arange(0,100,25)))
print("virginica",np.percentile(iris_virginica["petallength"],np.arange(0,100,25)))
print("versicolor",np.percentile(iris_versicolor["petallength"],np.arange(0,100,25)))
print("With some extra outlier")
print("setosa",np.percentile(np.append(iris_setosa["petallength"],500),np.arange(0,100,25)))

print("90th percentiles")
print("seatosa",np.percentile(iris_setosa["petallength"],90))
print("virginca",np.percentile(iris_virginica["petallength"],90))
print("versicolor",np.percentile(iris_versicolor["petallength"],90))

import seaborn as sns
sns.boxplot(x= iris["sepal.width"],y= iris["variety"],data =iris)

sns.boxplot(x="variety",y="sepal.width",data=iris)
plt.show()

sns.boxplot(x="class",y="sepallength",data=iris)
plt.show()



sns.violinplot(x="variety",y="petal.length",data=iris)
plt.show()

sns.scatterplot(x= iris["petal.length"],y=iris["petal.width"])
plt.show()

sns.scatterplot(x=iris.sepallength,y=iris.sepalwidth)
plt.show()

sns.FacetGrid(iris,hue = "variety",size=3).map(sns.distplot,"petal.length")
sns.FacetGrid(iris,hue = "variety",size=3).map(sns.distplot,"sepal.length")
plt.show()

['sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'class']
sns.FacetGrid(iris,hue="class",size=3).map(sns.distplot,"petallength").add_legend()
sns.FacetGrid(iris,hue="class",size=3).map(sns.distplot,"petalwidth").add_legend()
sns.FacetGrid(iris,hue="class",size=3).map(sns.distplot,"sepallength").add_legend()
sns.FacetGrid(iris,hue="class",size=3).map(sns.distplot,"sepalwidth").add_legend()
plt.show()



sns.scatterplot(x=iris["petal.length"],y=iris["petal.width"])
plt.show()

sns.set_style("whitegrid") 
sns.FacetGrid(iris, hue ="class", height = 6).map(plt.scatter, 'sepallength',  'petallength').add_legend()



sns.pairplot(iris,hue="variety");
#Scatter plot of the dataset
sns.pairplot(iris,hue="class")
plt.show()



iris.hist(
    column=["sepal.length", "sepal.width", "petal.length", "petal.width", "variety"],
    figsize=(10, 10)
    #,sharey=True, sharex=True
)
pylab.suptitle("Analyzing distribution for the series", fontsize="xx-large")



t = sns.PairGrid(iris,vars= ["sepallength","petalwidth"],hue="class")
t.map(plt.scatter)

import seaborn as sns
g = sns.PairGrid(iris, vars=["sepallength", "sepalwidth"], hue="class")
g.map(plt.scatter);

g = sns.PairGrid(iris)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter);
plt.show()

g = sns.PairGrid(iris)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot, cmap = "Blues_d")
g.map_diag(sns.kdeplot, lw = 3, legend = False);
plt.show()

iris_setosa = iris[iris["class"] == "setosa"]
iris_virginica = iris[iris["class"] == "virginica"]
iris_versicolor = iris[iris["class"] == "versicolor"]

import scipy.stats as st
c = st.pearsonr(iris['petallength'],iris['sepallength'])
c

import scipy.stats as st

covdata = np.cov(iris['petallength'],iris['sepalwidth'])
covdata

import scipy.stats as st
#Linear correlation coefficients
ccl1 = st.pearsonr(iris["petal.width"],iris["petal.length"])[0]
print("Linear correlation coefficient (petalwidth,petallength):", ccl1)
ccl2 = st.pearsonr(iris["sepal.width"],iris["petal.width"])[0]
print("Linear correlation coefficient (sepalwidth,petalwidth):", ccl2)

#Pearson correlation coefficient by hand
def r_pearson(X,Y):
    S_XY = (1/len(X))*(sum((X-np.mean(X))*(Y-np.mean(Y))))
    S_X = np.sqrt((1/len(X))*(sum((X-np.mean(X))**2)))
    S_Y = np.sqrt((1/len(Y))*(sum((Y-np.mean(Y))**2)))
    return (S_XY/(S_X*S_Y))

print('\nPearson correlation coefficient (petal_width,petal_length): %.3f' \
 %r_pearson(iris['petal.width'].values,iris['petal.length'].values ))
print('Pearson correlation coefficient (sepal.width,petal.width): %.3f' \
 %r_pearson(iris['sepal.width'].values,iris['petal.width'].values ))

#Correlation coefficient with Python module
iris.corr()

"""## **Multivariate analysis**
**correlation matrix**
"""

cmat = iris.corr(method = 'pearson')
f, ax =plt.subplots(figsize=(5,5))
sns.heatmap(cmat,ax =ax,  cmap="BuGn_r")

corrmat = iris.corr(method='pearson')
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidths=0.1)

#correlation matrix
corrmat = iris.corr(method='spearman')
cg = sns.clustermap(corrmat, cmap="YlGnBu", linewidths=0.1);
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
cg



mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])
sns.jointplot(x="x", y="y", data=df);

mp = iris.petallength.mean()
ms = iris.sepallength.mean()
meanvec = [mp,ms]
meanvec
covd = np.cov(iris.petallength,iris.sepallength)
covd
a= np.random.multivariate_normal(meanvec,covd,200)

df =pd.DataFrame(a,columns = ["x","y"])
df = pd.DataFrame(a, columns=[iris.petallength, iris.sepallength])
sns.jointplot(x="petallength", y="sepallength", data=df);
sns.jointplot(x="x",y="y",data =df)



mean_data = iris["sepallength"].mean() 
median_data = iris["sepallength"].median() 
std_dev = iris["sepallength"].std()
print(mean_data,median_data,std_dev)

"""# **Bivaraite distribution for sepal length and sepal width**"""

iris_setosa = iris[iris["class"] == "setosa"]
iris_virginica = iris[iris["class"] == "virginica"]
mean_l = iris["sepallength"].mean() 
mean_w = iris["sepalwidth"].mean()
mean_data = [mean_l,mean_w]
cov_data = np.cov(iris["sepallength"],iris["sepalwidth"])
print(mean_data)
print(cov_data)

data = np.random.multivariate_normal(mean_data, cov_data, 200)
df = pd.DataFrame(data, columns=["x", "y"])
sns.jointplot(x="x", y="y", data=df);

"""Without library function"""

def multivariate_gaussian(x, d, mean ,covariance):
  x_m = x -mean
  return((1.0)/(np.sqrt((2*np.pi)**d*np.linalg.det(covariance)))*np.exp(-(np.linalg.solve(covariance,x_m).T.dot(x_m))/2))

iris.max()

mean_ls = iris["sepallength"].mean() 
mean_ws = iris["sepalwidth"].mean()
mean_lp = iris["petallength"].mean() 
mean_wp = iris["petalwidth"].mean()

mean_data = [mean_ls,mean_ws,mean_lp,mean_wp]
cov_data = np.cov(iris["sepallength"],iris["sepalwidth"],iris["petallength"],iris["petalwidth"])
print(mean_data)
print(cov_data)

x = np.linspace(-3,5,num=150)
plt.plot(iris,multivariate_gaussian(iris,5,mean = mean_data,covariance = cov_data),label="$\mathcal{N}(0, 1)$")
plt.xlabel('x')
plt.ylabel("density: p(x)")
plt.title('Univariate normal distributions')
plt.ylim([0,1])
plt.xlim([-3,5])
plt.legend(loc=1)
fig.subplots_adjust(bottom=0.15)
plt.show()

mean_data = iris.mean()
mean_data

cov_data = iris.cov()
cov_data

mean =mean_data
cov = cov_data
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["sepallength", "sepalwidth","petallength","petalwidth"])
sns.jointplot(x="sepallength", y="sepalwidth", data=df);
sns.jointplot(x="sepallength", y="petallength", data=df);



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#Our 2-dimensional distribution will be over variables X and Y
N = 60
X = np.linspace(-3, 3, N)
Y = np.linspace(-3, 4, N)
X, Y = np.meshgrid(X, Y)

#Mean vector and covariance matrix
mu = np.array([0., 1.])
Sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]])


pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    
fac = np.einsum('...k,kl,...l->...', pos-mu,  Sigma_inv, pos-mu)

   return np.exp(-fac / 2) / N

#The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)

cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)


ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(27, -21)

plt.show()



x, y = np.random.multivariate_normal(mean, cov, 1000).T
with sns.axes_style("white"):
    sns.jointplot(x=x, y=y, kind="hex", color="k");
