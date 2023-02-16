from flask import Flask, render_template, request
from chatbot import ChatBot
import openai
import sys
import logging

app = Flask(__name__)
bot = ChatBot()
logging.basicConfig(level=logging.DEBUG)
# openai.api_key = "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"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Get user input
    user_input = request.form['user_input']

    # Generate chatbot response
    chatbot_response = bot.get_response(user_input) 

    return chatbot_response

@app.route('/upvote', methods=['POST'])
def upvote():
    bot.commit_response()
    return "Successfully committed response"
    pass

if __name__ == '__main__':
    app.run(debug=True)