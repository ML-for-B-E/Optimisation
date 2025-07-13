#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:18:27 2022

scripts for unit tests

@author: Rodolphe Le Riche
"""

import numpy as np
import test_functions as tf
from gradient_descent import get_gradient
import pytest
import forward_propagation as fwd
from activation_functions import sigmoid

#%% check get_gradient
# playing with the following code, I got convinced that epsilon=1.e-7 is a reasonable setting
fun = tf.sphere


def sphere_grad(x):
    """analytical gradient of the sphere"""
    dim = len(x)
    df = np.zeros(shape=(dim,), dtype=float)
    for i in range(dim):
        df[i] = 2 * (x[i] - (i + 1))

    return df


dim = 4
LB = -5 * np.ones(dim)
UB = 5 * np.ones(dim)
x = np.random.uniform(low=LB, high=UB)
fd_grad = get_gradient(func=fun, x=x, epsilon=1.0e-7)
exact_grad = sphere_grad(x)
print("exact gradient =", exact_grad)
print("exact - finite_diff grads =", exact_grad - fd_grad)


########## forward_propagation
# TODO: make sure all the tests are running
@pytest.mark.ut
def test_forward_nn_returns_correct_value():
    # given
    inputs = np.array([[1, 2, 5, 4]])
    weights = [
        np.array([[1, 0.2, 0.5, 1, -1], [2, 1, 3, 5, 0], [0.2, 0.1, 0.6, 0.78, 1]])
    ]
    activation = sigmoid
    expected_output = np.array([[0.99899323, 1, 0.99945816]])
    # when
    output = fwd.forward_propagation(inputs, weights, activation)
    # then
    np.testing.assert_allclose(output, expected_output)


@pytest.mark.ut
@pytest.mark.parametrize(
    "activation", [(sigmoid), ([sigmoid]), ([[sigmoid, sigmoid, sigmoid]])]
)
def test_forward_nn_returns_correct_value(activation):
    # given
    inputs = np.array([[1, 2, 5, 4]])
    weights = [
        np.array([[1, 0.2, 0.5, 1, -1], [2, 1, 3, 5, 0], [0.2, 0.1, 0.6, 0.78, 1]])
    ]

    expected_output = np.array([[0.99899323, 1, 0.99945816]])
    # when
    output = fwd.forward_propagation(inputs, weights, activation)
    # then
    np.testing.assert_allclose(output, expected_output)


@pytest.mark.ut
def test_vector_to_weights_returns_correct_value():
    # given
    vector = [
        0.2027827,
        -0.05644616,
        0.26441774,
        0.62177434,
        -0.09559271,
        -0.09558601,
        0.64471093,
        0.31330392,
        -0.19166212,
        0.22149921,
        -0.18918948,
        -0.19013338,
        0.09878068,
        -0.78109339,
        -0.70419476,
        -0.22955292,
        -0.41348657,
        0.12829094,
        -0.45401204,
        -0.70615185,
        0.73282438,
        -0.11288815,
    ]
    network_structure = [5, 3, 1]
    expected_weights = [
        np.array(
            [
                [
                    0.2027827,
                    -0.05644616,
                    0.26441774,
                    0.62177434,
                    -0.09559271,
                    -0.09558601,
                ],
                [
                    0.64471093,
                    0.31330392,
                    -0.19166212,
                    0.22149921,
                    -0.18918948,
                    -0.19013338,
                ],
                [
                    0.09878068,
                    -0.78109339,
                    -0.70419476,
                    -0.22955292,
                    -0.41348657,
                    0.12829094,
                ],
            ]
        ),
        np.array([[-0.45401204, -0.70615185, 0.73282438, -0.11288815]]),
    ]
    # when
    weights = fwd.vector_to_weights(vector, network_structure)
    # then
    for layer_weight, expected_layer_weight in zip(weights, expected_weights):
        np.testing.assert_allclose(layer_weight, expected_layer_weight)
