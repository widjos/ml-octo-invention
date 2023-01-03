 
from flask import Flask, render_template



app = Flask(__name__ , template_folder="template" )


tempo =  ['Cellphones','Animales','Casa']

@app.route('/')
def home():

    return render_template('index.html' , myList = tempo)



if __name__ == "__main__":
    app.run(port=4200,debug=True)