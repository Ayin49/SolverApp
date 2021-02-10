# driver
import tkinter as tk
from app import Wiz
from integrators import LorenzEquation, TaylorIntegrator, RungeKutta4th

lor_eq = LorenzEquation()
integ_dict = {'Taylor': TaylorIntegrator(lor_eq), "Runge-Kutta": RungeKutta4th(lor_eq)}
app = Wiz(integ_dict, (1, 1, 1))
tk.mainloop()

# note to self you can zoom by moving mouse while the right mouse button is pressed
