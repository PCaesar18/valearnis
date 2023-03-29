from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from ml_models.preprocessing import preprocess


# Define the model and fit to the data
def lstm_model(user_input):
    # Load the data, should be past input and outputs given to Valearnis
    past_inputs = "input"  # load_data("X.npy")
    past_response = "responses"  # load_data("Y.npy")

    # Preprocess the data
    past_inputs = preprocess(past_inputs)
    vectorizer = CountVectorizer()
    past_inputs = vectorizer.fit_transform(past_inputs)

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(past_inputs, past_response, test_size=0.2)

    # Define the model and train it
    model = Sequential()
    model.add(Embedding(input_dim=1000, output_dim=64))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))

    # if necessary, evaluate the model
    # loss, accuracy = model.evaluate(X_test, y_test)

    # Preprocess the user input amd predict the response on the new input
    new_input = preprocess(user_input).vectorizer.transform([user_input])
    prediction = model.predict(new_input)

    return prediction
