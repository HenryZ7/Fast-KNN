import numpy as np
import fastknn as fast


def get_data():
    x_train = (
        np.load("x_train.npy") / 1.0
    )  # turns data into float # shape=(60000,28,28)
    x_test = np.load("x_test.npy")  # shape=(10000,28,28)
    y_train = np.load("y_train.npy") / 1.0
    y_test = np.load("y_test.npy")

    train_row_number = x_train.shape[0]
    test_row_number = x_test.shape[0]

    x_train = x_train.reshape(train_row_number, -1)
    x_test = x_test.reshape(test_row_number, -1)
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = get_data()

x_train, x_test = fast.random_compress(x_train, x_test, 64)

prediction = np.zeros((len(y_test),))
correct = 0
for i in range(len(y_test)):
    if i % 10 == 0:
        print(i, correct)
    prediction[i] = fast.single_predict(x_train, y_train, x_test[i])
    if prediction[i] == y_test[i]:
        correct += 1

prediction_accuracy = (correct / len(y_test)) * 100
print(prediction_accuracy)
