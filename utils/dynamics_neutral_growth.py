import torch
from torch import Tensor as T
from torch.nn.parameter import Parameter


def dynamics_neutral_mitosis(W: T, n: int):
    """
    Equations (4) and (5)
    XXX: check that halving columsn for W is still the right answer!
    XXX: make sure E,I types stay in blocks here!!!
    """
    new_W = W.clone()

    new_column = new_W[:,n] * 0.5
    new_W = torch.concat([new_W[:,:n], new_column, new_column, new_W[:,n+1:]], axis=1)

    new_row = new_W[n]
    new_W = torch.concat([new_W[:n], new_row, new_row, new_W[n+1:]], axis=0)

    return new_W

def select_top_params(init_costs, init_ws, init_thetas, num_initialisation_iterations):
    # TODO: shouldn't this be 10 not 20??
    top_costs, top_ws, top_thetas = [], [], []
    for i in range(num_initialisation_iterations):
        idx = init_costs[i].argmin()
        top_costs.append(init_costs[i,idx])
        top_ws.append(init_ws[i,idx])
        top_thetas.append(init_thetas[i,idx])
    top_costs, top_ws, top_thetas = [torch.stack(l) for l in (top_costs, top_ws, top_thetas)]
    ranking = torch.argsort(top_costs)
    return top_costs, top_ws, top_thetas, ranking
