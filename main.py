import numpy as np
import matplotlib.pyplot as plt
import math

x_train = np.loadtxt('./ex1data1.txt', usecols=0, delimiter=',')
y_train = np.loadtxt('./ex1data1.txt', usecols=1, delimiter=',')


class linear_regression_ml:
    def __init__(self, X_train, y_train, w_init, b_init, alpha, no_iters):
        self.features = X_train
        self.targets = y_train
        self.w = w_init
        self.b = b_init
        self.learning_rate = alpha
        self.total_iters = no_iters
        self.m = len(self.features)  # Calculating m
        self.predictions = 0

    def find_cost(self):
        f_x = (self.w * self.features) + self.b  # Calculating f_x
        sq_error = (f_x - self.targets) ** 2
        cost = (sq_error / (2 * self.m)).sum()
        return cost

    def find_gradient(self):
        f_x = (self.w * self.features) + self.b  # Calculating f_x
        error = (f_x - self.targets)
        dj_dw = ((error * self.features) / self.m).sum()
        dj_db = (error / self.m).sum()
        return dj_dw, dj_db

    def run_gradient_descent(self):
        j_his = []
        w_his = []
        for i in range(self.total_iters):
            dj_dw, dj_db = self.find_gradient()
            self.w = self.w - (self.learning_rate * dj_dw)
            self.b = self.b - (self.learning_rate * dj_db)
            if i < 10000:
                j_his.append(self.find_cost())
            if i % math.ceil(self.total_iters / 10) == 0:
                w_his.append(self.w)
                print("Interation number: {}, The current w value is {}".format(i, self.w))
        return self.w, self.b, j_his, w_his

    @staticmethod
    def make_prediction(w, b, feature):
        prediction = (w * feature) + b
        print("A city with {} population may give {:.2f}$ profit".format(feature * 10_000, prediction * 10_000))

    def accuracy_visualization(self, w, b):
        predictions = (w * self.features) + b
        plt.scatter(self.features, self.targets, color='#03C04A', label='Actual')
        plt.plot(self.features, predictions, color='#234F1E', label='Predicted')
        plt.xlabel("Population (in 10,000s)")
        plt.ylabel("Profit (in 10,000$)")
        plt.show()


# Instance of an Object
AI = linear_regression_ml(x_train, y_train, 0, 0, 0.01, 1500)

# Running gradient descent
final_w, final_b, _, _ = AI.run_gradient_descent()
print("\n\nThe final w is {}".format(final_w))
print("The final b is {}".format(final_b))

# AI.accuracy_visualization(final_w, final_b)
AI.make_prediction(final_w, final_b, 3.5)
AI.make_prediction(final_w, final_b, 7.0)

# Check Accuracy
AI.accuracy_visualization(final_w, final_b)