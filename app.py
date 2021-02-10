"""
application for visualising custom integrators for Lorenz equation
"""
import tkinter as tk
from tkinter import Frame, Button
from tkinter import LEFT, TOP, X, FLAT, RAISED, RIGHT, BOTTOM, DISABLED, NORMAL, YES
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class Wiz(tk.Tk):
    """
    window for visualisation
    """

    def __init__(self, integrators, begin):
        tk.Tk.__init__(self)
        self.wm_title("Lorenz equation animation")
        self.begin = np.asarray(begin)
        self.integrators = integrators
        self.equation = integrators['Taylor'].lorenz
        self.integ = integrators['Taylor']
        self.reset_plot()
        self.init_plot()
        #       setting the plot and axes limits
        fig = Figure(figsize=(5, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(fig, master=self)  # A tk.DrawingArea.
        self.canvas.draw()
        self.axes = fig.add_subplot(111, projection='3d')
        self.axes.set_xlim3d(-10, 15)
        self.axes.set_ylim3d(-10, 20)
        self.axes.set_zlim3d(0, 50)
        self.main_plot_line, = self.axes.plot(self.xs, self.ys, self.zs, color="green")
        self.current_plot_position = self.axes.scatter3D([self.xs[-1]], [self.ys[-1]], [self.zs[-1]], color="red")
        self.axes.set_axis_off()
        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.ani = FuncAnimation(fig, self.animate_one, interval=200)
        self.paused = False
        plt.tight_layout()
        plt.show()
        self.canvas.mpl_connect("key_press_event", self.on_key_press)
        toolbar2 = Frame(master=self, bd=1, relief=FLAT)
        toolbar2.pack(side=BOTTOM, fill=X)
        quit_but = tk.Button(master=toolbar2, text="Quit", command=self._quit)
        quit_but.pack(side=tk.RIGHT)
        self.lorenz = tk.Button(master=toolbar2, text="Lorentz", command=self.equation_show)
        self.lorenz.pack(side=tk.LEFT)
        toolbar = Frame(master=self, bd=1, relief=RAISED)
        toolbar.pack(side=TOP, fill=X)
        self.taylint = tk.Button(master=toolbar, relief=RAISED, text="Taylor", command=self.taylor)
        self.taylint.pack(side=LEFT)
        self.rungeint = tk.Button(master=toolbar, relief=RAISED,
                                  text="Runge-Kutta 4", command=self.runge)
        self.rungeint.pack(side=LEFT)

        step_ten = tk.Button(master=toolbar, relief=FLAT, text="Step x10", command=self.step_ten)
        step_ten.pack(side=RIGHT)
        step = tk.Button(master=toolbar, relief=FLAT, text="Step", command=self.step)
        step.pack(side=RIGHT)
        self.btn_text = tk.StringVar()
        self.btn_text.set("Pause")
        step_pause = tk.Button(master=toolbar, relief=FLAT,
                               textvariable=self.btn_text, command=self.step_pause)
        step_pause.pack(side=RIGHT)

    def reset_plot(self):
        """
        resets plot to blank state
        """
        self.wyniki = np.asarray([self.begin])

    def init_plot(self):
        """
        initializes plot to starting settings
        """
        temp = self.integ.step(self.begin)
        self.wyniki = np.asarray([temp])
        self.xs, self.ys, self.zs = self.wyniki[:, 0], self.wyniki[:, 1], self.wyniki[:, 2]

    def animate_one(self, _):
        """
        animates 1 frame
        :param _: not used
        """
        self.wyniki = np.vstack([self.wyniki, self.integ.step(self.wyniki[-1])])
        self.xs, self.ys, self.zs = self.wyniki[:, 0], self.wyniki[:, 1], self.wyniki[:, 2]
        self.main_plot_line.set_data(self.xs, self.ys)
        self.main_plot_line.set_3d_properties(self.zs)
        self.current_plot_position._offsets3d = ([self.xs[-1]], [self.ys[-1]], [self.zs[-1]])

    def animate_ten(self, _):
        """
        animates 10 frames
        :param _: not used
        """
        for _ in range(10):
            self.wyniki = np.vstack([self.wyniki, self.integ.step(self.wyniki[-1])])
        self.xs, self.ys, self.zs = self.wyniki[:, 0], self.wyniki[:, 1], self.wyniki[:, 2]
        self.main_plot_line.set_data(self.xs, self.ys)
        self.main_plot_line.set_3d_properties(self.zs)
        self.current_plot_position._offsets3d = ([self.xs[-1]], [self.ys[-1]], [self.zs[-1]])

    def on_key_press(self, event):
        """
        matplotlib toolbar event handler
        :param event:
        """
        print("you pressed {}".format(event.key))
        key_press_handler(event, self.canvas, self.toolbar)

    def _quit(self):
        """
        shuts app down
        :return:
        """
        self.quit()  # stops mainloop
        self.destroy()  # this is necessary on Windows to prevent
        # Fatal Python Error: PyEval_RestoreThread: NULL tstate

    def submit_lorenz_para(self, widget, ents):
        """
        changes Lorenz equation parameters on button click
        :param widget:
        :param ents:
        :return:
        """
        params = {e: float(ents[e].get()) for e in ents}
        self.lorenz['state'] = NORMAL
        self.rungeint['state'] = NORMAL
        self.taylint['state'] = NORMAL
        self.equation.set_params(params)
        widget.destroy()

    def submit_taylor_para(self, widget, ents):
        """
        changes Taylor solver parameters on button click
        :param widget:
        :param ents:
        :return:
        """
        params = {e: float(ents[e].get()) for e in ents}
        params['order'] = int(params['order'])
        self.lorenz['state'] = NORMAL
        self.rungeint['state'] = NORMAL
        self.taylint['state'] = NORMAL
        self.integrators['Taylor'].set_solver_params(params)
        self.begin[0] = params['x0']
        self.begin[1] = params['y0']
        self.begin[2] = params['z0']
        self.reset_plot()
        self.integ = self.integrators['Taylor']
        self.init_plot()
        widget.destroy()

    def submit_runge_para(self, widget, ents):
        """
        changes Taylor solver parameters on button click
        :param widget:
        :param ents:
        :return:
        """
        params = {e: float(ents[e].get()) for e in ents}
        self.lorenz['state'] = NORMAL
        self.rungeint['state'] = NORMAL
        self.taylint['state'] = NORMAL
        self.integrators['Runge-Kutta'].set_solver_params(params)
        self.begin[0] = params['x0']
        self.begin[1] = params['y0']
        self.begin[2] = params['z0']
        self.reset_plot()
        self.integ = self.integrators['Runge-Kutta']
        self.init_plot()
        widget.destroy()

    def cancel_window(self, widget):
        """
        closes form window without updating parameters
        :param widget:
        :return:
        """
        self.lorenz['state'] = NORMAL
        self.rungeint['state'] = NORMAL
        self.taylint['state'] = NORMAL
        widget.destroy()

    def makeform(self, widget, fields):
        """
        generic form making for changing parameters windows
        :param widget:
        :param fields:
        :return:
        """
        entries = {}
        for field in fields:
            row = Frame(widget)
            lab = tk.Label(row, width=22, text=field + ": ", anchor='w')
            ent = tk.Entry(row)
            ent.insert(0, self.params[field])
            row.pack(side=TOP, fill=X, padx=5, pady=5)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT, expand=YES, fill=X)
            entries[field] = ent
        return entries

    def equation_show(self):
        """
        window with Lorenz equation parameters
        :return:
        """
        self.lorenz['state'] = DISABLED
        self.rungeint['state'] = DISABLED
        self.taylint['state'] = DISABLED
        lorenz_form = tk.Toplevel(self)
        lorenz_form.title("Lorenz equation parameters")
        lorenz_form.geometry("400x150")
        lorenz_form_submit = Button(lorenz_form, text="Submit",
                                    command=lambda: self.submit_lorenz_para(lorenz_form, ents))
        lorenz_form_submit.pack(side=tk.BOTTOM)
        lorenz_form_cancel = Button(lorenz_form, text="Cancel",
                                    command=lambda: self.cancel_window(lorenz_form))
        lorenz_form_cancel.pack(side=tk.TOP)
        fields = ('sigma', 'rho', 'beta')
        self.params = self.equation.get_params()
        ents = self.makeform(lorenz_form, fields)

    def taylor(self):
        """
        window with Taylor solver parameters
        :return:
        """
        self.lorenz['state'] = DISABLED
        self.rungeint['state'] = DISABLED
        self.taylint['state'] = DISABLED
        taylor_form = tk.Toplevel(self)
        taylor_form.title("Taylor solver parameters")
        taylor_form.geometry("400x200")
        taylor_form_submit = Button(taylor_form, text="Submit",
                                    command=lambda: self.submit_taylor_para(taylor_form, ents))
        taylor_form_submit.pack(side=tk.BOTTOM)
        taylor_form_cancel = Button(taylor_form, text="Cancel",
                                    command=lambda: self.cancel_window(taylor_form))
        taylor_form_cancel.pack(side=tk.TOP)
        fields = ('order', 'time step', 'x0', 'y0', 'z0')
        self.params = dict(self.integrators['Taylor'].get_solver_params())
        begin = {'x0': self.begin[0], 'y0': self.begin[1], 'z0': self.begin[2]}
        self.params.update(begin)
        ents = self.makeform(taylor_form, fields)

    def runge(self):
        """
        window for 4th order Runge-Kutta solver parameters
        :return:
        """
        self.lorenz['state'] = DISABLED
        self.rungeint['state'] = DISABLED
        self.taylint['state'] = DISABLED
        runge_form = tk.Toplevel(self)
        runge_form.title("Taylor solver parameters")
        runge_form.geometry("400x200")
        runge_form_submit = Button(runge_form, text="Submit",
                                   command=lambda: self.submit_runge_para(runge_form, ents))
        runge_form_submit.pack(side=tk.BOTTOM)
        runge_form_cancel = Button(runge_form, text="Cancel",
                                   command=lambda: self.cancel_window(runge_form))
        runge_form_cancel.pack(side=tk.TOP)
        fields = ('time step', 'x0', 'y0', 'z0')
        self.params = dict(self.integrators['Runge-Kutta'].get_solver_params())
        begin = {'x0': self.begin[0], 'y0': self.begin[1], 'z0': self.begin[2]}
        self.params.update(begin)
        ents = self.makeform(runge_form, fields)

    def step(self):
        """
        changes visualisation to 1 step animation
        :return:
        """
        self.ani.__dict__['_func'] = self.animate_one

    def step_ten(self):
        """
        changes visualisation to 10 step animation
        :return:
        """
        self.ani.__dict__['_func'] = self.animate_ten

    def step_pause(self):
        """
        stops and starts animation
        :return:
        """
        if self.paused:
            self.ani.event_source.start()
            self.btn_text.set("Pause")
        else:
            self.ani.event_source.stop()
            self.btn_text.set("Play")
        self.paused = not self.paused
