import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from ml_models.preprocessing import preprocess


def linear_svc(user_input):
    # define example user inputs and corresponding GPT-3 responses
    training_inputs = ['What is the weather like today?', 'Tell me a joke', 'What is the capital of France?']
    training_responses = ['The weather is sunny and warm!',
                          'Why did the tomato turn red? Because it saw the salad dressing!',
                          'The capital of France is Paris.']

    # preprocess the input data the same way
    for i in range(len(training_inputs)):
        training_inputs[i] = preprocess(training_inputs[i])
        training_responses[i] = preprocess(training_responses[i])

    # Load the data in a pandas dataframe
    data = pd.DataFrame({'text': training_inputs, 'response': training_responses})

    # vectorize the inputs using TfidfVectorizer
    vectorizer = TfidfVectorizer()
    fitted_data = vectorizer.fit_transform(data['text'])

    # create a LinearSVC model and fit it to the vectorized inputs
    model = LinearSVC()
    model.fit(fitted_data, data['response'])

    # vectorize user input
    user_input_vectorized = vectorizer.transform([user_input])

    # predict the response
    prediction = model.predict(user_input_vectorized)
    return prediction
