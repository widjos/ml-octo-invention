
from flask import Flask, render_template ,request
from analisys.Functions import Functions


app = Flask(__name__ , template_folder="template" )
paths = ['..\data\Cellphones data.csv', '..\data\CancerBreast.csv', '..\data\Best_movies_netflix.csv', '..\data\heart.csv', '..\data\WineQt.csv' ]

#Instancia de mi clase principal con funciones ML
functMl = Functions()

@app.route('/result-analysis', methods=['POST'])
def resultAnalisis():
    xAxys = request.form.get('xAxys')
    yAxys = request.form.get('yAxys')
    prediction = request.form.get('prediction')
    print(xAxys,yAxys)
    fileName = request.form.get('currentFile')
    outputValue = functMl.selectFunction(fileName,str(xAxys),str(yAxys),prediction)
    imgPath = functMl.getImage(fileName)
    
    currentFile = [fileName]
    columnsData =  functMl.LoadData(fileName=fileName) 
    return render_template('index.html',
                            myList=currentFile, 
                            xColumn = columnsData, 
                            yColumn = columnsData,
                            outputValue = outputValue,
                            imgPath = imgPath,
                            nameLabel = fileName
                            )

#Resultado del arbol
@app.route('/result-arbol', methods=['POST'])
def loadTree():
    xAxys = request.form.get('xAxys')
    yAxys = request.form.get('yAxys')
    prediction = request.form.get('prediction')
    print(xAxys,yAxys)
    fileName = request.form.get('currentFile')
    outputValue = functMl.selectFunction(fileName,xAxys,yAxys,prediction)
    imgPath = functMl.getImage(fileName)
    
    currentFile = [fileName]
    columnsData =  functMl.LoadData(fileName=fileName) 
    return render_template('index.html',
                            myList=currentFile, 
                            xColumn = columnsData, 
                            yColumn = columnsData,
                            outputValue = outputValue,
                            imgPath = imgPath,
                            nameLabel = fileName
                            )    

#Resultado  Neuronal 
@app.route('/result-nn', methods=['POST'])
def loadNN():
    xAxys = request.form.get('xAxys')
    yAxys = request.form.get('yAxys')
    votes = request.form.get('votes')
    prod = request.form.get('production')
    print(xAxys,yAxys)
    fileName = request.form.get('currentFile')
    outputValue = functMl.selectFunction(fileName,xAxys,yAxys,votes,prod)
    
    currentFile = [fileName]
    columnsData =  functMl.LoadData(fileName=fileName) 
    return render_template('index.html',
                            myList=currentFile, 
                            xColumn = columnsData, 
                            yColumn = columnsData,
                            outputValue = outputValue,
                            nameLabel = fileName
                            )    

#LLenar las nuevas listas 
@app.route('/load-data', methods=['POST'])
def loadData():
    fileName = request.form.get('file-analisis')
    currentFile = [fileName] 
    columnsData =  functMl.LoadData(fileName=fileName)
    pageName = 'index.html'
    if fileName == 'CancerBreast':
        pageName = 'tree.html'
    elif fileName == 'NetflixMovies': 
        pageName = 'neuronal_network.html'
    return render_template(pageName,myList=currentFile, xColumn = columnsData, yColumn = columnsData, nameLabel = fileName)


#Home page
@app.route('/' , methods=['GET'])
def home():
    imgPath = ''
    tempo =  ['Cellphones','CancerBreast','NetflixMovies','Heart','WineQT']
    return render_template('index.html' , myList = tempo, imgPath=imgPath)




if __name__ == "__main__":
    app.run(port=4200,debug=True)