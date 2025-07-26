"""
    A collection of methods for forward propagation in a neural network
    
    @author: Rodolphe Le Riche, Kevin Kpakpo, Brian Dedji Whannou
"""
import numpy as np
from typing import Union, Callable, List
from copy import deepcopy

# TODO: there might be simpler expressions for these operations avoiding all the transpositions
# by making the products from the left z1 = x*W1 , z2 = z1*W2 etc
def forward_propagation(
    inputs: np.ndarray,
    weights: List[np.ndarray],
    activation_functions: Union[Callable, List[Callable], List[List[Callable]]],
) -> np.ndarray:
    """
    Returns the output of the network.

            Parameters:
                    inputs: is an array of shape n,p
                    (with p being the number of input features and n the number of observations)
                    weights: List of matrices.
                             Each matrix represents a layer.
                             The number of rows is the number of nodes.
                             The number of columns is the number inputs of the layer (with the bias)
                    activation_functions: the user can provide either
                        * one activation function for the whole network (one callable)
                        * one activation function for each layer (List of callable)
                        * one activation function per nodes (List of List of callable)
            Outputs:
                    layer_output: is an array of shape n,q
                    (with q being the number of nodes in the last layer and n the number of observations)

    """

    # number of nodes per layers
    network_structure = []
    for layer_weights in weights:
        network_structure.append(layer_weights.shape[0])

    # all activation functions
    activation_functions_parsed = parse_activation_function(
        activation_functions, network_structure
    )

    if inputs.ndim == 1:
        inputs = inputs.reshape(1,inputs.shape[0])
    bias = np.repeat(np.array([[1]]), inputs.shape[0], axis=0)

    layer_input = np.append(inputs, bias, axis=1)

    for func, layer_weights in zip(activation_functions_parsed, weights):
        layer_combinaison = layer_weights.dot(layer_input.T)
        layer_output = apply_activation_functions(func, layer_combinaison.T)
        layer_input = np.append(layer_output.T, bias, axis=1)

    return layer_output.T


def parse_activation_function(
    activation_functions, network_structure
) -> List[List[Callable]]:
    activation_functions_parsed = deepcopy(activation_functions)
    if isinstance(activation_functions, Callable):
        activation_functions_parsed = []
        for number_nodes in network_structure:
            layer_funcs = [activation_functions for n in range(0, number_nodes)]
            activation_functions_parsed.append(layer_funcs)

    if isinstance(activation_functions, List):
        if isinstance(activation_functions[0], Callable):
            activation_functions_parsed = []
            for func, number_nodes in zip(activation_functions, network_structure):
                activation_functions_parsed.append(
                    [func for n in range(0, number_nodes)]
                )
    return activation_functions_parsed


def apply_activation_functions(
    functions: List[object], values: np.ndarray
) -> np.ndarray:
    all_activated_values = np.zeros(values.shape)
    for i in range(0, values.shape[0]):
        row_values = values[i, :]
        all_activated_values[i, :] = [
            node_func(value) for node_func, value in zip(functions, row_values)
        ]
    return np.array(all_activated_values).T


def create_layer_weights(inputs_size: int, outputs_size: int) -> np.ndarray:
    avg = 0
    std = np.sqrt(1 / inputs_size)
    layer_weights = np.random.normal(avg, std, (outputs_size, inputs_size))

    return layer_weights


def create_weights(network_structure: List[int]) -> List[np.ndarray]:
    # np.random.seed(42)
    weights = []
    for layer in range(0, len(network_structure) - 1):
        weights.append(
            create_layer_weights(
                inputs_size=network_structure[layer] + 1,
                outputs_size=network_structure[layer + 1],
            )
        )
    return weights


def vector_to_weights(
    vector: np.ndarray, network_structure: List[int]
) -> List[np.ndarray]:
    weights = []
    for layer in range(0, len(network_structure) - 1):
        input_size = network_structure[layer] + 1
        output_size = network_structure[layer + 1]
        selected_parameters = np.array(vector[: (input_size * output_size)])
        weights.append(selected_parameters.reshape((output_size, input_size)))
        vector = vector[(input_size * output_size) :]

    return weights


def weights_to_vector(weights: List[np.ndarray]):
    vector = []
    network_structure = []
    for layer_weight in weights:
        vector = vector + layer_weight.reshape((1, -1))[0].tolist()
        if len(network_structure) == 0:
            network_structure = network_structure + [layer_weight.shape[1] - 1]
        network_structure = network_structure + [layer_weight.shape[0]]

    return np.array(vector), network_structure
