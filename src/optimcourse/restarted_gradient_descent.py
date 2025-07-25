##########################################
########  WORK IN PROGRESS DOES NOT WORK
##########################################

import math
from gradient_descent import gradient_descent

def restarted_gradient_descent(
    func: object,
    start_x: np.array = None,
    LB: np.array,
    UB: np.array,
    budget: int = 1e3,
    nb_restarts: int = 4,
    step_factor: float = 1e-1,
    direction_type: str = "momentum",
    do_linesearch: bool = True,
    min_step_size: float = 1e-11,
    min_grad_size: float = 1e-6,
    inertia: float = 0.9,
    printlevel: int = 1,
) -> dict:
    """
    an optimizer made of repeated gradient_descent searches restarted at random

    Parameters :
    those of gradient_descent plus
    nb_restarts (int) : number of restarts

    Returns:
    res (dict) : a dictionary with the search results

    """
    one_budget= math.ceil(budget/nb_restarts)
    for iter in range(1,(nb_restarts+1)):
        # get the starting point
        if iter==1 and start_x != None:
            one_start_x = start_x
        else:
            one_start_x = np.random.uniform(low=LB, high=UB)

        # do one search
        res = gradient_descent(func=func,
                start_x=one_start_x,
                LB=LB, UB=UB,
                budget=one_budget,
                step_factor=step_factor,
                direction_type=direction_type,
                do_linesearch=do_linesearch,
                min_step_size=min_step_size, min_grad_size=min_grad_size,
                inertia=inertia,
                printlevel=printlevel
              )

