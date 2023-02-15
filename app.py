from flask import Flask, render_template, request

app = Flask(__name__)
chat_history = []

@app.route('/', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        message = request.form['message']
        chat_history.append(message)
        response = "Hello"
        chat_history.append(response)
        return render_template('index.html', messages=chat_history)
    else:
        return render_template('index.html', messages=chat_history)

if __name__ == '__main__':
    app.run(debug=True)