import numpy as np
import torch
from sklearn.metrics import mean_absolute_percentage_error


def compare_to_analytic(x_space,y_space,psy_trial,analytic_solution,pde):
    nx = x_space.shape[0]
    ny = y_space.shape[0]
    surface = np.zeros((nx,ny))
    an_surface = np.zeros((nx,ny))
    for i, x in enumerate(x_space):
        for j, y in enumerate(y_space):
            input_point = torch.Tensor([x, y])
            input_point.requires_grad_()
            net_out = pde.forward(input_point)

            psy_t = psy_trial(input_point, net_out)
            surface[i][j] = psy_t
            an_surface[i][j] = analytic_solution([x, y])
    diff = np.max(np.abs(surface-an_surface))
    mape = mean_absolute_percentage_error(an_surface, surface)
    return mape

if __name__ == '__main__':
    x_space = np.linspace(0,1.0,10)
    y_space = x_space
    X,Y = np.meshgrid(x_space,y_space)
    an_sol = np.sin(X)*Y**2
    from auxiliary import psy_trial
    from analytics import analytic_solution
    from PDE import PDEnet
    pde = PDEnet(x_space.shape[0])
    df = compare_to_analytic(x_space, y_space, psy_trial, analytic_solution, pde)
    qq = 0
