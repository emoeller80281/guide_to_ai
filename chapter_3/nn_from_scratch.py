from math import e, log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class Model():
    def __init__(self, m1=-1, m2=1, b1=0, b2=0, alpha=0.1):
        self.m1 = m1
        self.m2 = m2
        self.b1 = b1
        self.b2 = b2

        self.alpha = alpha

        self.h1 = None
        self.h2 = None

        self.y_hat1 = None
        self.y_hat2 = None

        self.dl_dm1 = None
        self.dl_dm2 = None
        self.dl_db1 = None
        self.dl_db2 = None

    def forward(self, x, y1, y2):
        self.h1 = self.compute_hidden_state(self.m1, x, self.b1)
        self.h2 = self.compute_hidden_state(self.m2, x, self.b2)

        self.y_hat1 = self.softmax(self.h1, self.h1, self.h2)
        self.y_hat2 = self.softmax(self.h2, self.h1, self.h2)

        if y1 == 1:
            loss = self.cross_entropy_loss(self.y_hat1)
        elif y2 == 1:
            loss = self.cross_entropy_loss(self.y_hat2)

        return loss

    def backward(self, x, y1, y2):
        self.dl_dm1 = self.calculate_gradients(x, self.y_hat1, y1)
        self.dl_dm2 = self.calculate_gradients(x, self.y_hat2, y2)
        self.dl_db1 = self.calculate_gradients(1, self.y_hat1, y1)
        self.dl_db2 = self.calculate_gradients(1, self.y_hat2, y2)

        self.update_params()

    def update_params(self):
        self.m1 = self.update_param(self.m1, self.alpha, self.dl_dm1)
        self.m2 = self.update_param(self.m2, self.alpha, self.dl_dm2)
        self.b1 = self.update_param(self.b1, self.alpha, self.dl_db1)
        self.b2 = self.update_param(self.b2, self.alpha, self.dl_db2)

    def compute_hidden_state(self, m, x, b):
        return m*x + b

    def softmax(self, h, h1, h2):
        return (e**h/(e**h1 + e**h2))

    def cross_entropy_loss(self, y_hat):
        return -log(y_hat)

    def calculate_gradients(self, x, y_hat, y):
        return x*(y_hat - y)

    def update_param(self, param, alpha, param_grad):
        return param - (alpha * param_grad)

    def make_prediction(self, y1, y2):
        model_prediction = None
        if self.y_hat1 > self.y_hat2:
            model_prediction = 1
        else:
            model_prediction = 2

        correct_answer = None
        if y1 == 1 and y2 == 0:
            correct_answer = 1
        elif y1 == 0 and y2 == 1:
            correct_answer = 2

        assert model_prediction != None, "Model prediction is None, check if statements"
        assert correct_answer != None, "Correct Answer is None, check if statements"

        result = None
        if model_prediction == correct_answer:
            result = 1
        else:
            result = 0
        
        return result

training_examples = [2.351, 2.295, 13.405, 13.378]
one_hot_y = [[1, 0], [1, 0], [0, 1], [0, 1]]

m1 = -1
m2 = 1
b1 = 0
b2 = 0

alpha = 0.1
num_epochs = 20

model = Model(m1, m2, b1, b2, alpha)

epoch_loss = []
for epoch in range(num_epochs):
    training_dict = {}
    for i in range(len(training_examples)):
        training_dict[i] = {}
        sample_num = i % 4

        x = training_examples[sample_num]
        y1 = one_hot_y[sample_num][0]
        y2 = one_hot_y[sample_num][1]

        training_dict[i]["x"] = x
        training_dict[i]["y1"] = y1
        training_dict[i]["y2"] = y2
        training_dict[i]["m1"] = model.m1
        training_dict[i]["m2"] = model.m2
        training_dict[i]["b1"] = model.b1
        training_dict[i]["b2"] = model.b2

        loss = model.forward(x, y1, y2)
        training_dict[i]["h1"] = model.h1
        training_dict[i]["h2"] = model.h2
        training_dict[i]["y_hat1"] = model.y_hat1
        training_dict[i]["y_hat2"] = model.y_hat2

        acc_model = Model(model.m1, model.m2, model.b1, model.b2, model.alpha)
        num_correct = 0
        for c in range(len(training_examples)):
            x_acc = training_examples[c]
            y1_acc = one_hot_y[c][0]
            y2_acc = one_hot_y[c][1]

            _ = acc_model.forward(x_acc, y1_acc, y2_acc)

            num_correct += acc_model.make_prediction(y1_acc, y2_acc)
        accuracy = num_correct / len(training_examples)

        model.backward(x, y1, y2)

        training_dict[i]["dL/dm1"] = model.dl_dm1
        training_dict[i]["dL/dm2"] = model.dl_dm2
        training_dict[i]["dL/db1"] = model.dl_db1
        training_dict[i]["dL/db2"] = model.dl_db2
        training_dict[i]["Loss"] = loss
        training_dict[i]["Accuracy"] = accuracy

    df = pd.DataFrame(training_dict).T
    df = df[["x", "m1", "m2", "dL/dm1", "dL/dm2", "b1", "b2", "dL/db1", "dL/db2", "h1", "h2", "y_hat1", "y_hat2", "y1", "y2", "Loss", "Accuracy"]]
    df = df.T
    df_rounded = df.round(3)

    if epoch == 0:
        print(df_rounded)
    mean_loss = np.mean(df.loc["Loss"])

    epoch_loss.append(mean_loss)

plt.figure(figsize=(7,5))
plt.plot(epoch_loss)
plt.title("Average Loss Per Epoch Across Training")
plt.xlabel("Training Epoch")
plt.ylabel("Mean Loss")
plt.savefig("chapter_3/training_loss_over_training.png", dpi=200)
