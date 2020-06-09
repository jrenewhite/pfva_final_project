import numpy as np

#---------------------------------------------------------------------------------------------
def sigmoid(z):
    s = 1/(1+np.exp(-1*z))
    
    return s

#---------------------------------------------------------------------------------------------
def initialize_with_zeros(dim):
    w = np.zeros([dim,1])
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

#---------------------------------------------------------------------------------------------
def propagate(w, b, X, Y):   
    m = X.shape[1]
    
    A = sigmoid(np.dot(w.T,X)+b)                          # compute activation
    cost = (1/m)*np.sum(-Y*np.log(A)-(1-Y)*np.log(1-A))   # compute cost

    dw = (1/m)*(np.dot(X,(A-Y).T))
    db = (1/m)*np.sum(A-Y)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

#---------------------------------------------------------------------------------------------
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)        

        w = w-learning_rate*grads["dw"]
        # b = b-learning_rate*grads["db"]        
        
        if i % 100 == 0:
            costs.append(cost)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": grads["dw"],
             "db": grads["db"]}
    
    return params, grads, costs

#---------------------------------------------------------------------------------------------
def predict(w, b, X):   
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T,X)+b)
    Y_prediction = np.where(A > 0.5,1,0)
  
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

#---------------------------------------------------------------------------------------------
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = np.zeros([X_train.shape[0],1]),0

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
#---------------------------------------------------------------------------------------------