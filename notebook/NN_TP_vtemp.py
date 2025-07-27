# this is a script python copying the notebook for easier development (debugging)

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib  # maybe useful Ubuntu rodo ISIMA
# matplotlib.use('Qt5Agg') # maybe useful Ubuntu rodo ISIMA

from typing import Callable, List

from optimcourse.gradient_descent import gradient_descent, gradient_finite_diff
from optimcourse.optim_utilities import print_rec
from optimcourse.forward_propagation import (
    forward_propagation,
    create_weights,
    vector_to_weights,
    weights_to_vector)
from optimcourse.activation_functions import (
    relu,
    sigmoid,
    linear,
    leaky_relu
)
from optimcourse.test_functions import (
    linear_function,
    ackley,
    sphere,
    quadratic,
    rosen,
    L1norm,
    sphereL1
)
from optimcourse.restarted_gradient_descent import restarted_gradient_descent


# ###########################################################
# # Example with an analytic function
# dim = 2
# LB = [-5] * dim
# UB = [5] * dim
# zfun = rosen
# printlevel = 4
#
# # res = restarted_gradient_descent(func=zfun, LB=LB,UB=UB,start_x=np.random.uniform(low=LB,high=UB),budget=500,nb_restarts=4,
# #                                  printlevel=printlevel)
# res = restarted_gradient_descent(func=zfun, LB=LB,UB=UB,budget=500,nb_restarts=4,
#                                  printlevel=printlevel)
# print_rec(res=res, fun=zfun, dim=len(res["x_best"]),
#            LB=LB, UB=UB , printlevel=printlevel, logscale = True)

###########################################################




def simulate_data_target(fun: Callable,
                       n_features: int,
                       n_obs: int,
                       LB: List[float],
                       UB: List[float]) -> dict:
    
    entry_data = np.random.uniform(low= LB,high=UB,
                                   size=(n_obs, n_features))
    target = np.apply_along_axis(fun, 1, entry_data)
    
    return {"data": entry_data, "target": target}




# ### Error functions 
# 
# A utility function to transform a vector into weight matrices. You will probably not need it, but this is used in the calculation of the error function (the vector is transformed into NN weights, ...).

# We define 2 error functions, one for regression is the mean square error, the other is the cross-entropy error for classification.


# mean squared error
def cost_function_mse(y_predicted: np.ndarray,y_observed: np.ndarray):
    error = 0.5 * np.mean((y_predicted - y_observed)**2)
    return error


# entropy
# TODO : make it more robust by testing when y_predicted is equal or less than 0, or equal or larger than 1
def cost_function_entropy(y_predicted: np.ndarray,y_observed: np.ndarray):

    n = len(y_observed)
    
    term_A = np.multiply(np.log(y_predicted),y_observed)
    term_B = np.multiply(1-y_observed,np.log(1-y_predicted))
    
    error = - (1/n)*(np.sum(term_A)+np.sum(term_B))

    return(error)


# TODO: I think this function would only work for 1 output because of the reshape(-1) at the end that is not applied to the data.
# --> make it multi-dimensional
def error_with_parameters(vector_weights: np.ndarray,
                          network_structure: List[int],
                          activation_function: Callable,
                          data: dict,
                          cost_function: Callable,
                          regularization: float = 0) -> float:
    
    weights = vector_to_weights(vector_weights,used_network_structure)
    predicted_output = forward_propagation(data["data"],weights,activation_function)
    predicted_output = predicted_output.reshape(-1,)
    
    error = cost_function(predicted_output,data["target"]) + regularization * np.sum(np.abs(vector_weights))
    
    return error


def neural_network_cost(vector_weights):
    cost = error_with_parameters(vector_weights,
                                 network_structure=used_network_structure,
                                 activation_function=used_activation,
                                 data=used_data,
                                 cost_function=used_cost_function)

    return cost


#################################

# ## Question : Make your own network
# 
# 3. Generate 100 data points with the quadratic function in 2 dimensions.
# 4. Create a network with 2 inputs, 5 ReLU neurons in the hidden layer, and 1 output.
# 5. Learn it on the quadratic data points you generated. Plot some results, discuss them.

# ### Generate the data

# In[ ]:


used_function = quadratic
n_features = 2
n_obs = 100
LBfeatures = [-5] * n_features
UBfeatures = [5] * n_features
simulated_data = simulate_data_target(fun = used_function,n_features = n_features,n_obs=n_obs,
                                      LB=LBfeatures,UB=UBfeatures)


# ### Create the network
# and calculate the cost function of the first, randomly initialized, network.

# In[ ]:

used_network_structure = [2,5,1]
used_activation = [[sigmoid,sigmoid,sigmoid,leaky_relu,leaky_relu],[leaky_relu]]#[sigmoid,leaky_relu] #sigmoid # leaky_relu, sigmoid
used_data = simulated_data
used_cost_function = cost_function_mse
weights = create_weights(used_network_structure)
weights_as_vector,_ = weights_to_vector(weights)
dim = len(weights_as_vector) 

print("Number of weights to learn : ",dim)
print("Initial cost of the NN : ",neural_network_cost(weights_as_vector))


# ### Learn the network

# In[ ]:


LB = [-8] * dim
UB = [8] * dim
printlevel = 1
# res = gradient_descent(func = neural_network_cost,
#                  start_x = weights_as_vector,
#                  LB = LB, UB = UB,budget = 100,printlevel=printlevel,
#                  min_step_size = 1e-13, min_grad_size = 1e-13, do_linesearch=True,step_factor=0.01, direction_type="momentum"
#             )
#

res = restarted_gradient_descent(func=neural_network_cost, start_x=weights_as_vector,LB=LB,UB=UB,budget=50000,nb_restarts=10,
                                 printlevel=printlevel)
print_rec(res=res, fun=neural_network_cost, dim=len(res["x_best"]),
           LB=LB, UB=UB , printlevel=printlevel, logscale = True)

weights_best = vector_to_weights(res["x_best"],used_network_structure)

# a small study about the gradients ...
# initial gradient
init_grad = gradient_finite_diff(func=neural_network_cost , x=weights_as_vector , f_x=neural_network_cost(weights_as_vector))
final_grad = gradient_finite_diff(func=neural_network_cost , x=res["x_best"] , f_x=neural_network_cost(res["x_best"]))
#


print("Best NN weights:",weights_best)


######### randopt comparison ######################
# from optimcourse.random_search import random_opt
# res = random_opt(func=neural_network_cost,LB=[-5]*dim,UB=[5]*dim,budget=10000)



######### scipy comparison ######################
# from scipy.optimize import minimize
# from scipy.optimize import show_options
# show_options(solver='minimize',method='powell')  # get info on specific modules
# res = minimize(neural_network_cost, weights_as_vector, method='nelder-mead',options={'xatol': 1e-12, 'disp': True,'maxfev': 100000,'adaptive': True})
# res = minimize(neural_network_cost, weights_as_vector, method='BFGS',options={'maxiter': 100, 'disp': True, 'gtol': 1e-6})
# res = minimize(neural_network_cost, weights_as_vector, method='powell',options={'maxfev': 10000, 'disp': True, 'ftol': 1e-9, 'xtol': 1e-9})

#########  cma-es comparison ######################
### from the terminal : pip install cma
# import cma
# res, es = cma.fmin2(objective_function=neural_network_cost,x0=weights_as_vector,sigma0=5,options={'popsize':300})
## es.result.fbest has best objective, res is xbest

# Compare the network prediction to the true function
#

# function definition
dimFunc = 2
LBfunc = [-5,-5]
UBfunc = [5,5]
fun = quadratic
  
# start drawing the function (necessarily dim==2)
no_grid = 100
# 
x1 = np.linspace(start=LBfunc[0], stop=UBfunc[0],num=no_grid)
x2 = np.linspace(start=LBfunc[1], stop=UBfunc[1],num=no_grid)
x, y = np.meshgrid(x1, x2)
xy = np.array([x,y])
z = np.apply_along_axis(fun,0,xy)
zNN = np.apply_along_axis(forward_propagation,0,xy,weights_best,used_activation)
zNN = np.squeeze(zNN)

# Create a figure with two 3D subplots side by side
fig = plt.figure(figsize=(12, 6))
# First 3D subplot
ax1 = fig.add_subplot(121, projection='3d')  # 1 row, 2 columns, first subplot
ax1.set_zlim(0,100)
ax1.plot_surface(x, y, z, cmap='jet', shade= "false")
# ax1.plot_surface(x, y, z, cmap='viridis')
ax1.set_title('target function')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f')
# Second 3D subplot
ax2 = fig.add_subplot(122, projection='3d')  # 1 row, 2 columns, second subplot
ax2.set_zlim(0,100)
ax2.plot_surface(x, y, zNN, cmap='jet', shade= "false")
# ax2.plot_surface(x, y, zNN, cmap='plasma')
ax2.set_title('NN prediction')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f_{NN}')
plt.tight_layout() # Adjust layout
plt.show()

# # **FIN DU NOTEBOOK**
