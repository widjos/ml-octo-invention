from sklearn import preprocessing
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, plot_tree
le = preprocessing.LabelEncoder()
import pandas as pd


class Functions():

    def __init__(self) -> None:
        self.paths = ['.\data\Cellphones data.csv']
        self.data = []
        self.loadData = False
        self.resultPredict = ''
        


    def LoadData(self,path):
        self.data = pd.read_csv(path)
        loadData = True

    def lineal(self,xAxis , yAxis, data, valuePredict):

    #    if self.loadData == False:
    #       return
       #
        x = data[xAxis].values.reshape(-1,1)
        y  = data[yAxis].values.reshape(-1,1)

        rreg = linear_model.LinearRegression()
        rreg.fit(x,y)

        y_pred = rreg.predict(x)

        plt.scatter(x,y, color="black")
        plt.xlabel(xAxis)
        plt.ylabel(yAxis)
        plt.plot(x,y_pred,color ="r", linewidth=3)

        plt.savefig('..\\img\\linear.png')

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y,y_pred)
        prediction = rreg.predict(np.array([[valuePredict]]))[0]
        temp = []
        temp.append(f'rmse: {rmse}')
        temp.append(f'r: {r2}')
        temp.append(f'prediction for value {xAxis}: {prediction}')

        return "Resultado:\n".join(str(x) for x in temp)
        
    def polinomial(self,xAxis,yAxis, data,pediction, degree = 2):
        x = data[xAxis].values.reshape(-1,1)
        y = data[yAxis].values.reshape(-1,1)
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        x_poly = poly.fit_transform(x)
        model = linear_model.LinearRegression()
        model.fit(x_poly,y)
        y_pred = model.predict(x_poly)
        plt.scatter(x,y)
        plt.plot(x,y_pred,color='g')
        plt.xlabel(xAxis)
        plt.ylabel(yAxis)
        plt.savefig('..\\img\\polinomial.png')
        
        rmse = np.sqrt(mean_squared_error(y,y_pred))
        r2 = r2_score(y,y_pred)
        
        
        print ('RMSE: ' + str(rmse))
        print ('R2: ' + str(r2))

    def GaussianNB(self,xAxis, yAxis, data , predict): 
        X = data[xAxis].values.reshape(-1,1)
        Y  = data[yAxis]
        clf = GaussianNB()
        #Training 
        clf.fit(X,Y)
        # x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=17)
        #Testing
        vPred = clf.predict([[predict]]) 
        print(f'Resultado para {predict} : ', vPred, clf.score(X,Y))


    def Three(self, dataset):
       
        var_column = [c for c in dataset.columns if c not in ['id','diagnosis']]

        X = dataset.loc[:,var_column]
        y = dataset.loc[:,'diagnosis']

        #Split de data in test and validator 
        X_train , X_valid , y_train, y_valid = train_test_split(X,y, test_size=0.2, random_state=42)

        #Model
        clf = DecisionTreeClassifier(max_leaf_nodes=8, class_weight='balanced')
        clf.fit(X_train,y_train)
    

        #Create figure 
        plt.figure(figsize=(20,10))
        #Create the tree plot

        plot_tree(clf,
           feature_names = var_column, #Feature names
           class_names = ["B","M"], #Class names
           rounded = True,
           filled = True)

        plt.show()

        #predictions = clf.predict(X_test)
        #print(accuracy_score(y_test,predictions))


    
    def MdModel(self,data):
        pass        


paths = ['..\data\Cellphones data.csv', '..\data\CancerBreast.csv']
a: Functions = Functions()


#print(a.lineal('screen size','battery size', paths[0],9.0))
dat = pd.read_csv(paths[0])
#a.GaussianNB('screen size','brand',dat,4500) #Use a level fro easy find 
#a.polinomial('screen size','battery size',dat,'')

b = pd.read_csv(paths[1])
a.Three(b)
#print(b.groupby('diagnosis').size()) 
#print(b.head())

