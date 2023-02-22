from flask import Flask, render_template, request
from chatbot import ChatBot
import openai
import sys
import logging

app = Flask(__name__)
bot = ChatBot()
bot.generate_embeddings()
logging.basicConfig(level=logging.DEBUG)

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

@app.route('/downvote', methods=['POST'])
def downvote():
    bot.commit_negative_response()
    return "Successfully committed response"
if __name__ == '__main__':
    app.run(debug=True)