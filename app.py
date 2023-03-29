import openai
import os
import json
from flask import Flask, redirect, render_template, request, url_for, jsonify
from ml_models.preprocessing import preprocess
from ml_models.linearsvc import linear_svc

# flask app
app = Flask(__name__)

# openai api key
openai.api_key = os.getenv("OPENAI_API_KEY")


# routes
@app.route('/')
def index():
    return render_template('index.html')


# chat route
@app.route("/chat", methods=("GET", "POST"))
def chat():
    # Get user input from the form
    user_input = request.args.get('user_input') if request.method == 'GET' else request.form['user_input']
    # Preprocess user input to get the predicted class. The predicted class is then used to generate the prompt for the
    # GPT-3 model

    # call ML model to get prediction for the input. disabled in this version because of heroku conflicts
    # user_input = preprocess(user_input)
    # predicted_user_input = linear_svc(user_input)

    # Call the GPT-3 API to get a response
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=generate_messages(user_input),
            temperature=0.6,
            stop=None,
            max_tokens=256,
        )
        response = response.choices[0].message["content"]
        # Extract the response errors from the API result
    except openai.error.AuthenticationError:
        # Handles authentication error
        response = "Authentication failed. Please ask for API key."
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


# helper function to generate messages for the GPT-3 model when the user inputs a message to the chatbot
# the messages can be changed with supplying additional roles and content
def generate_messages(prompt):
    messages = [
        {"role": "system", "content": "You are an education AI tool for the company named Valearnis. "}
    ]
    if prompt:
        messages.append({"role": "user", "content": prompt})
    return messages


def display_image():
    user_input = request.form['user_input']

    image_response = openai.Image.create(
        prompt=user_input,
        n=1,
        size="512x512",
        response_format="b64_json",
    )
    image_response = image_response["data"][0]["b64_json"][:50]
    return image_response


if __name__ == '__main__':
    # port = int(os.environ.get("PORT", 5000))
    # app.run(host='0.0.0.0', port=port)
    app.run(debug=True)
