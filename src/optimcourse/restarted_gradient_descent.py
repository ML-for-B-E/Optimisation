import math

import numpy as np

from optimcourse.gradient_descent import gradient_descent

def restarted_gradient_descent(
    func: object,
    start_x: np.array = None,
    LB: np.array = None,
    UB: np.array = None,
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
    res_total (dict) : a dictionary with the search results

    """
    one_budget= math.ceil(budget/nb_restarts)
    res_total = {}
    cum_time=0
    for iter in range(1,(nb_restarts+1)):
        # get the starting point
        if iter==1 and start_x.any()!=False:
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


        if iter==1:
            res_total = res
        else:
            if res['f_best']<res_total['f_best']:
                res_total['f_best']=res['f_best']
                res_total['x_best'] = res['x_best']

            if printlevel>0:
                res_total['hist_time_best'] = np.append(res_total['hist_time_best'],np.array(res['hist_time_best'])+cum_time)
                res_total['hist_f_best'] = np.append(res_total['hist_f_best'],res['hist_f_best'])
                res_total['hist_x_best']= np.append(res_total['hist_x_best'], res['hist_x_best'], axis=0)

            if printlevel>1:
                res_total['hist_time'] = np.append(res_total['hist_time'],
                                                        np.array(res['hist_time']) + cum_time)
                res_total['hist_f'] = np.append(res_total['hist_f'], res['hist_f'])
                res_total['hist_x'] = np.append(res_total['hist_x'], res['hist_x'], axis=0)

                # res.keys()
                # Out[1]: dict_keys(['time_used', 'hist_f', 'hist_time', 'hist_x',
                #         'x_best', 'f_best', 'hist_f_best', 'hist_time_best', 'hist_x_best', 'stop_condition'])



        cum_time+=res['time_used']
        res_total['time_used']=cum_time

    return res_total