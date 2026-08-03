import numpy as np
import matplotlib.pyplot as plt
import math

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
plt.show()

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

def compute_cost(x_train , y_train , w, b):
    m = x_train.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w*x_train[i] + b
        cost = (f_wb - y_train[i])**2
        cost_sum += cost
    total_sum = (1/(2*m))*cost_sum        
    return total_sum


def compute_gradient(x_train , y_train ,w , b):
    m = x_train.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w*x_train[i] + b
        dj_w_i = (f_wb - y_train[i]) * x_train[i]
        dj_b_i = (f_wb - y_train[i])
        dj_db += dj_b_i
        dj_dw += dj_w_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    
    return dj_dw, dj_db


def gradient_descent(x_train , y_train, w_in, b_in, alpha, num_iter, compute_cost, compute_gradient):
    
    j_history = []
    p_history = []
    w  = w_in
    b = b_in
    
    for i in range(num_iter):
        dj_dw, dj_db = compute_gradient(x_train , y_train ,w , b)
        
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
        
        if i < 100000:
            j_history.append(compute_cost(x_train , y_train , w, b))
            p_history.append([w,b])
            
        if i % math.ceil(num_iter/10) == 0:
            print(f"Iteration {i:4}: Cost {j_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, j_history, p_history
    
    
    
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, 0, 0, 1.0e-2, 
                                                    10000, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
    
    
    
    
    
    
    
    
             
