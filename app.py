
import openai
import os
import json
from flask import Flask, redirect, render_template, request, url_for, jsonify
from models.preprocessing import *

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/chat", methods=("GET", "POST"))
def chat():
    user_input = request.args.get('user_input') if request.method == 'GET' else request.form['user_input']
    # preprocess(user_input)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=generate_messages(user_input),
            temperature=0.6,
            stop=None,
            max_tokens=256,
        )
        response = response.choices[0].message["content"]

    except openai.error.RateLimitError:
        # Handles rate limit error
        response = "The server is experiencing a high volume of requests. Please try again later."
    except openai.error.InvalidRequestError:
        # Handles request error
        response = "The request was wrong. It could be because of a missing parameter, or a parameter that is not " \
                   "valid. "
    except openai.error.APIConnectionError as e:
        # Handles connection error
        response = "Failed to connect to OpenAI API: {}".format(e)

    return jsonify(content=response)

def generate_messages(prompt):
    messages = [
        {"role": "system", "content": "You are a shakespearean swearword AI. "}
    ]
    if prompt:
        messages.append({"role": "user", "content": prompt})
    return messages


if __name__ == '__main__':
    app.run(debug=True)
