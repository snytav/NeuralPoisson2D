# -*- coding: utf-8 -*-
"""Копия блокнота "PDE-Poisson.ipynb"

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1aL6xz_T-tstK6WQ9NtoeJBBR5Q6oDQ3t
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#from my_optimizer import optimizer_step

from matplotlib import pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

print("CUDA GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
   x = torch.ones(20)
   print(x.device)
   x = x.to("cuda:0")
   # or x=x.to("cuda")
print(x)
print(x.device)

def f(x):
    return 0.



from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian
from torch.autograd import grad

def loss_function(x, y,pde,psy_trial,f):
    loss_sum = 0.

    for xi in x:
        for yi in y:
            input_point = torch.Tensor([xi, yi])
            input_point.requires_grad_()

            net_out = pde.forward(input_point)
            net_out_w = grad(outputs=net_out, inputs=pde.fc1.weight, grad_outputs=torch.ones_like(net_out),
                       retain_graph=True,create_graph=True)


            net_out_jacobian = jacobian(pde.forward,input_point,create_graph=True)
            # jac1  = get_jacobian(pde.forward,input_point,2)
            net_out_hessian = hessian(pde.forward,input_point,create_graph=True)
            psy_t = psy_trial(input_point, net_out)

            inputs = (input_point, net_out)
            psy_t_jacobian = jacobian(psy_trial, inputs,create_graph=True)[0]
            psy_t_hessian  = hessian(psy_trial,inputs,create_graph=True)
            psy_t_hessian = psy_t_hessian[0][0]
            # acobian(jacobian(psy_trial))(input_point, net_out

            gradient_of_trial_d2x = psy_t_hessian[0][0]
            gradient_of_trial_d2y = psy_t_hessian[1][1]

            # D_gradient_of_trial_d2x_D_W0 = grad(outputs=gradient_of_trial_d2x, inputs=pde.fc1.weight, grad_outputs=torch.ones_like(gradient_of_trial_d2x), retain_graph=True)
            # D_gradient_of_trial_d2y_D_W0 = grad(outputs=gradient_of_trial_d2y, inputs=pde.fc1.weight, grad_outputs=torch.ones_like(gradient_of_trial_d2y), retain_graph=True)
            # D_func_D_W0 = grad(outputs=func,iputs=pde.fc1.weight,grad_outputs=torch.ones_like(func))
            func = f(input_point)
            func_t = torch.Tensor([func])
            func_t.requires_grad_()

            err_sqr = ((gradient_of_trial_d2x + gradient_of_trial_d2y) - func_t) ** 2
            # D_err_sqr_D_W0 = 2*((gradient_of_trial_d2x + gradient_of_trial_d2y) - func)*(
            #                     (D_gradient_of_trial_d2x_D_W0 + D_gradient_of_trial_d2y_D_W0) -D_func_D_W0
            #                     )

            loss_sum += err_sqr
            qq = 0

    return loss_sum







class PDEnet(nn.Module):
    def __init__(self,N):
        super(PDEnet,self).__init__()
        self.N = N
        fc1 = nn.Linear(2,self.N) # первый слой
        fc2 = nn.Linear(self.N, 1) # второй слой
        self.fc1 = fc1
        self.fc2 = fc2

    def forward(self,x):
        x = x.reshape(1, 2)
        y = self.fc1(x)
        y = torch.sigmoid(y)
        y = self.fc2(y.reshape(1, self.N))
        return y



nx = 10
ny = nx
pde = PDEnet(nx)
dx = 1. / nx
dy = 1. / ny

x_space = torch.linspace(0, 1, nx)
y_space = torch.linspace(0, 1, ny)
print("CUDA GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
  print(torch.device)
  print(x_space.device)
  x_space.to("cuda:0")
  #y_space.to("cuda:0")
  #print(x_space)
  print(x_space.device)

input_point = torch.zeros(2)
net_out = pde.forward(input_point)

psy_t = psy_trial(input_point, net_out)
print(psy_t)

loss = loss_function(x_space, y_space,pde,psy_trial,f)
loss.backward()
print(x_space.grad)

def train(lmb,x_space,y_space,N_epoch,pde,psy_trial,f):
    # lmb = 0.001
    optimizer = torch.optim.SGD(pde.parameters(), lr=lmb)
    import time
    t1 = time.time()
    for i in range(N_epoch):
        #print('begin ',i,loss.item())
        optimizer.zero_grad()
        #print('zero grad ',i,loss.item())
        #print(x_space.device,y_space.device)
        loss = loss_function(x_space, y_space,pde,psy_trial,f)
        #print(loss.device)
        print(i,loss.item())
        loss.backward(retain_graph=True)
        #print('loop end ',i,loss.item())
        optimizer.step()

        print('step ',i,loss.item())
t2 = time.time()
print('computation time ',t2-t1)


import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(x_space.numpy(), y_space.numpy())
surf = ax.plot_surface(X, Y, surface, rstride=1, cstride=1, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
plt.title('Neural solution')

fig = plt.figure()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(x_space.numpy(), y_space.numpy())
surf = ax.plot_surface(X, Y, an_surface, rstride=1, cstride=1, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
plt.title('Analytic solution')

from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(an_surface, surface)
mape

import numpy as np
diff = np.abs(an_surface-surface)
md = np.max(diff)
md = np.where(diff == md)

print(surface[5][7],
      an_surface[5][7],
      np.abs(surface[5][7]-an_surface[5][7]),
      np.abs(surface[5][7]-an_surface[5][7])/np.abs(an_surface[5][7])
      )