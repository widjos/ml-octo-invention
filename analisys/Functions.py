from sklearn import preprocessing
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import neural_network
le = preprocessing.LabelEncoder()
import pandas as pd


class Functions():


    def __init__(self) -> None:
        self.paths = ['.\data\Cellphones data.csv', '..\data\CancerBreast.csv', '..\data\Best_movies_netflix.csv', '..\data\heart.csv', '..\data\WineQt.csv' ]
        self.paths2 = {
            'Cellphones': '.\data\Cellphones data.csv',
            'CancerBreast': '.\data\CancerBreast.csv',
            'NetflixMovies': '.\data\Best_movies_netflix.csv',
            'Heart': '.\data\heart.csv',
            'WineQT': '.\data\WineQt.csv'
        }
        self.data = []
        self.loadData = False
        self.resultPredict = ''

    def createResult(self, value1='', value2='' , value3='', value4=''):
        return f'Resultado {value1}'
        
    def cleanData(self):
        self.data = []

    def LoadData(self,fileName):
        self.data = pd.read_csv(self.paths2[fileName])
        columnas = []
        for col in self.data.columns:
            columnas.append(col)
        loadData = True
        return columnas

    #Lineal
    def lineal(self,xAxis , yAxis, data, valuePredict):

        x = data[xAxis].values.reshape(-1,1)
        y  = data[yAxis].values.reshape(-1,1)

        rreg = linear_model.LinearRegression()
        rreg.fit(x,y)

        y_pred = rreg.predict(x)

        plt.scatter(x,y, color="black")
        plt.xlabel(xAxis)
        plt.ylabel(yAxis)
        plt.plot(x,y_pred,color ="r", linewidth=3)

        plt.savefig('..\\static\\img\\linear.png')

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y,y_pred)
        prediction = rreg.predict(np.array([[valuePredict]]))[0]
        
        #Preparation for output
        temp = f'rmse: {rmse}\n'
        temp += f'r: {r2}\n'
        temp += f'Prediction for value {xAxis}: {prediction}'

        return temp

    #Polinomail    
    def polinomial(self,xAxis,yAxis, data,prediction, degree = 2):
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
        plt.savefig('..\\static\\img\\polinomial.png')
        
        rmse = np.sqrt(mean_squared_error(y,y_pred))
        r2 = r2_score(y,y_pred)
        
    
        modPre = model.predict([[prediction,0.85]])
        
        #Output 
        temp = f'RMSE: {rmse}\n'
        temp += f'R2: {r2}\n'
        temp += f'Prediction : {modPre}'
        return temp

    #Modelo Gaussiano
    def GaussianNB(self,xAxis, yAxis, data , predict): 
        X = data[xAxis].values.reshape(-1,1)
        Y  = data[yAxis]
        clf = GaussianNB()
        #Training 
        clf.fit(X,Y)
        # x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=17)
        #Testing
        vPred = clf.predict([[predict]]) 
        plt.plot(X,Y)
        plt.savefig('..\\static\\img\\gaussian.png')
        
        return f'Predict para {predict}: {vPred} \n SCORE: {clf.score(X,Y)}'  


    #Modelo del Arbol
    def Three(self, dataset):
        
        #Para cargar todas las columnas del data set
        #var_column = [c for c in dataset.columns if c not in ['id','diagnosis']]
      
        #X = dataset.loc[:,var_column]
        X = pd.DataFrame({
            'radMean': dataset['radius_mean'],
            'perMean': dataset['perimeter_mean'],
            'areaMean' : dataset['area_mean']   
        })
        
        y = dataset.loc[:,'diagnosis']

        #Split de data in test and validator 
        X_train , X_valid , y_train, y_valid = train_test_split(X,y, test_size=0.2, random_state=42)

        #Model
        clf = DecisionTreeClassifier(max_leaf_nodes=5, class_weight='balanced')
        clf.fit(X_train,y_train)
    

        #Create figure 
        plt.figure(figsize=(10,8))
        #Create the tree plot

        plot_tree(clf,
           feature_names = ['radMean','perMean','areaMean'], #Feature names
           class_names = ["B","M"], #Class names
           rounded = True,
           filled = True)

        plt.savefig('..\\static\\img\\tree-model.png')

        #Se obtiene los datos de
        predictions = clf.predict(X_valid)
        
        #print(accuracy_score(y_valid,predictions))

        pred = clf.predict([(11.5,12.5,14)])
        temp = f'Acuraccy : {accuracy_score(y_valid,predictions)}\n'
        temp += f'Prediction : {pred}'

        return temp


    #Peliculas utiliza  el genero 
    def MdModel(self,data):
        #Conver label into number 
        lb_genere = LabelEncoder()
        lb_contry = LabelEncoder()
        Y= pd.array(lb_genere.fit_transform(data['main_genre']))

        X = pd.DataFrame({
            'year': data['release_year'],
            'score': data['score'],
            'votes' : data['number_of_votes'],
            'production':  lb_contry.fit_transform(data['main_production']) 
        })



        lr = 0.01 #Learning rate
        nn = [2,16,8,1]

        #Creamos el obejto del modelo 
        model = neural_network.MLPRegressor(
                solver='sgd',
                learning_rate_init= lr,
                hidden_layer_sizes=nn[1:],
                verbose=True,
                n_iter_no_change=1000
                )

        #Entrenamos el modelo 
        model.fit(X,Y)

        x_predict = pd.DataFrame({
            'year': [2015],
            'score': [8.1],
            'votes' : [20595],
            'production':  [5] 
        })

        #Pediccion 
        print("Preiccion : ", model.predict(x_predict))


a: Functions = Functions()


#print(a.lineal('screen size','battery size', paths[0],9.0))
#dat = pd.read_csv(paths[2])


#b = pd.read_csv(paths[1])
#a.Three(b)


#print(b.groupby('diagnosis').size())
paths = ['..\data\Cellphones data.csv', '..\data\CancerBreast.csv', '..\data\Best_movies_netflix.csv', '..\data\heart.csv', '..\data\WineQt.csv' ] 
c = pd.read_csv(paths[1])
#print(a.lineal('chol','sex', c,310))
#print(a.polinomial('fixed acidity','density',c,6.8))
#print(a.GaussianNB('screen size','brand',c,4500))
#print(a.Three(c))


