 
from flask import Flask, render_template ,request
from analisys.Functions import Functions


app = Flask(__name__ , template_folder="template" )

paths = ['..\data\Cellphones data.csv', '..\data\CancerBreast.csv', '..\data\Best_movies_netflix.csv', '..\data\heart.csv', '..\data\WineQt.csv' ]

functMl = Functions()

@app.route('/result-analysis', methods=['POST'])
def resultAnalisis():
    fileName = request.form.get('file-analisis')
    outputValue = ''
    imgPath = '' 
    pass



#LLenar las nuevas listas 
@app.route('/load-data', methods=['POST'])
def loadData():
    fileName = request.form.get('file-analisis')
    currentFile = [fileName] 
    columnsData =  functMl.LoadData(fileName=fileName)
    return render_template('index.html',myList=currentFile, xColumn = columnsData, yColumn = columnsData)


#Home page
@app.route('/')
def home():
    imgPath = ''
    tempo =  ['Cellphones','CancerBreast','NetflixMovies','Heart','WineQT']
    return render_template('index.html' , myList = tempo, imgPath=imgPath)




if __name__ == "__main__":
    app.run(port=4200,debug=True)