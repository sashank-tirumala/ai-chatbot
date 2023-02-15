from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Get user input
    user_input = request.form['user_input']

    # Generate chatbot response
    chatbot_response = 'Hi'

    return chatbot_response

if __name__ == '__main__':
    app.run(debug=True)