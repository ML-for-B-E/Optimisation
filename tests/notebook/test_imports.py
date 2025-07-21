import pytest

@pytest.mark.ut
def test_notebook_correction_project_import_cell_succeeds():
    # Given

    import numpy as np
    import matplotlib.pyplot as plt
    from typing import Callable, List

    from optimcourse.gradient_descent import gradient_descent
    from optimcourse.optim_utilities import print_rec
    from optimcourse.forward_propagation import (
        forward_propagation, 
        create_weights, 
        vector_to_weights,
        weights_to_vector)
    from optimcourse.activation_functions import (
        relu,
        sigmoid
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

    