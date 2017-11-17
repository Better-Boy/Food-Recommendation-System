from flask import Flask, render_template
 
app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template("home.html")

@app.route('/service')
def servicepage():
    return render_template('service.html')

if __name__ == '__main__':
    app.run()