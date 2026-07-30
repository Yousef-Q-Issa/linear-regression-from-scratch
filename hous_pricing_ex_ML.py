import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

print(f"x_train: {x_train}")
print(f"y_train: {y_train}")

m = x_train.shape[0]
print(f"number of training exampel: {m}")
# print(len(x_train))

i = 1

x_i = x_train[i]
y_i = y_train[i]

print(f"x^({i}), y^({i}) = ({x_i}, {y_i})")

plt.scatter(x_train, y_train , marker ='x' , c = 'r')
plt.title("Housing Prices")
plt.ylabel("PricePrice (in 1000s of dollars)")
plt.xlabel("Size (1000 sqft)")
# plt.show()

w = 200
b = 100
print(f"w = {w}")
print(f"b = {b}")

def compute_model_output(x, w ,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

tmp_f_wb = compute_model_output(x_train, w ,b)

plt.plot(x_train, tmp_f_wb, c='b' , label = "ourprediction")

plt.scatter(x_train, y_train , marker = 'x' , c= 'r', label = "actual values")

plt.title("Housing paice")

plt.ylabel("Price (in 1000s of dollars")
plt.xlabel("Size (1000 sqft)")

plt.legend()
plt.savefig("output.png", dpi=300, bbox_inches="tight")
plt.show()

b = 100
w = 200
x_i = 1.2
cost_1200sqft = x_i*w +b
print(f"${cost_1200sqft:.0f} thousand dollars")
