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


def makeform(widget, fields, params):
    """
    generic form making for changing parameters windows
    :param params: default parameters to display
    :param widget: window that will be changed
    :param fields: fields to employ
    :return:
    """
    entries = {}
    fields = [fields] if isinstance(fields, str) else fields
    for field in fields:
        row = Frame(widget)
        lab = tk.Label(row, width=22, text=field + ": ", anchor='w')
        ent = tk.Entry(row)
        ent.insert(0, params[field])
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries[field] = ent
    return entries


class Wiz(tk.Tk):
    """
    window for visualisation
    takes dict of integrators for init
    """

    def __init__(self, integrators):
        tk.Tk.__init__(self)
        self.wm_title("Lorenz equation animation")
        self.begin = np.asarray((1, 1, 1))
        self.integrators = integrators
        self.equation = integrators['Taylor'].lorenz
        self.integ = integrators['Taylor']
        self.wyniki = []
        self.reset_plot()
        self.init_plot()
        #       setting the plot and axes limits,
        #       calling canvas first so that plot manipulation on site is possible
        fig = Figure(figsize=(5, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(fig, master=self)  # A tk.DrawingArea.
        self.canvas.draw()
        self.axes = fig.add_subplot(111, projection='3d')
        self.axes.set_xlim3d(-10, 15)
        self.axes.set_ylim3d(-10, 20)
        self.axes.set_zlim3d(0, 50)
        self.main_plot_line, = self.axes.plot(self.wyniki[:, 0], self.wyniki[:, 1],
                                              self.wyniki[:, 2], color="green")
        self.current_plot_position = self.axes.scatter3D([self.wyniki[-1, 0]],
                                                         [self.wyniki[-1, 1]],
                                                         [self.wyniki[-1, 2]], color="red")
        self.axes.set_axis_off()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        #setting animation
        self.ani = FuncAnimation(fig, self.animate_one, interval=200)
        self.paused = False
        #setting matplotlib toolbar, part of features is not working with 3d
        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()
        self.canvas.mpl_connect("key_press_event", self.on_key_press)
        #setting 1st toolbar
        toolbar2 = Frame(master=self, bd=1, relief=FLAT)
        toolbar2.pack(side=BOTTOM, fill=X)
        quit_but = tk.Button(master=toolbar2, text="Quit", command=self._quit)
        quit_but.pack(side=tk.RIGHT)
        reset_but = tk.Button(master=toolbar2, text="Reset plot", command=self.reset_plot)
        reset_but.pack(side=tk.RIGHT)
        self.lorenz = tk.Button(master=toolbar2, text="Lorenz", command=self.equation_show)
        self.lorenz.pack(side=tk.LEFT)
        self.starting_point = tk.Button(master=toolbar2, relief=RAISED,
                               text="Starting point", command=self.start_set)
        self.starting_point.pack(side=LEFT)
        #setting 2nd toolbar
        toolbar3 = Frame(master=self, bd=1, relief=RAISED)
        toolbar3.pack(side=TOP, fill=X)
        self.taylint = tk.Button(master=toolbar3, relief=RAISED, text="Taylor", command=self.taylor)
        self.taylint.pack(side=LEFT)
        self.rungeint = tk.Button(master=toolbar3, relief=RAISED,
                                  text="Runge-Kutta 4", command=self.runge)
        self.rungeint.pack(side=LEFT)
        step_ten = tk.Button(master=toolbar3, relief=FLAT, text="Step x10", command=self.step_ten)
        step_ten.pack(side=RIGHT)
        step = tk.Button(master=toolbar3, relief=FLAT, text="Step", command=self.step)
        step.pack(side=RIGHT)
        self.btn_text = tk.StringVar()
        self.btn_text.set("Pause")
        step_pause = tk.Button(master=toolbar3, relief=FLAT,
                               textvariable=self.btn_text, command=self.step_pause)
        step_pause.pack(side=RIGHT)
        self.btn_text_axes = tk.StringVar()
        self.btn_text_axes.set("Axes Off")
        axes_show = tk.Button(master=toolbar2, relief=FLAT,
                               textvariable=self.btn_text_axes, command=self.axes_switch)
        axes_show.pack(side=RIGHT)
        self.ax_state = False

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
        self.wyniki = np.asarray([self.begin, temp])

    def animate_one(self, _):
        """
        animates 1 frame
        :param _: not used
        """
        self.wyniki = np.vstack([self.wyniki, self.integ.step(self.wyniki[-1])])
        self.main_plot_line.set_data(self.wyniki[:, 0], self.wyniki[:, 1])
        self.main_plot_line.set_3d_properties(self.wyniki[:, 2])
        self.current_plot_position._offsets3d = ([self.wyniki[-1, 0]],
                                                 [self.wyniki[-1, 1]], [self.wyniki[-1, 2]])

    def animate_ten(self, _):
        """
        animates 10 frames
        :param _: not used
        """
        for _ in range(10):
            self.wyniki = np.vstack([self.wyniki, self.integ.step(self.wyniki[-1])])
        self.main_plot_line.set_data(self.wyniki[:, 0], self.wyniki[:, 1])
        self.main_plot_line.set_3d_properties(self.wyniki[:, 2])
        self.current_plot_position._offsets3d = ([self.wyniki[-1, 0]],
                                                 [self.wyniki[-1, 1]], [self.wyniki[-1, 2]])

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
        self.button_state(NORMAL)
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
        self.button_state(NORMAL)
        self.integrators['Taylor'].set_solver_params(params)
        self.integ = self.integrators['Taylor']
        widget.destroy()

    def submit_runge_para(self, widget, ents):
        """
        changes Taylor solver parameters on button click
        :param widget:
        :param ents:
        :return:
        """
        params = {e: float(ents[e].get()) for e in ents}
        self.button_state(NORMAL)
        self.integrators['Runge-Kutta'].set_solver_params(params)
        self.integ = self.integrators['Runge-Kutta']
        widget.destroy()

    def submit_start_point(self, widget, ents):
        """
        changes starting point on button click
        :param widget:
        :param ents:
        :return:
        """
        params = {e: float(ents[e].get()) for e in ents}
        self.button_state(NORMAL)
        self.begin[0] = params['x0']
        self.begin[1] = params['y0']
        self.begin[2] = params['z0']
        widget.destroy()

    def start_set(self):
        """
        window with starting point coords
        :return:
        """
        self.button_state(DISABLED)
        start_form = tk.Toplevel(self)
        start_form.title("Taylor solver parameters")
        start_form.geometry("300x150")
        start_form_submit = Button(start_form, text="Submit",
                                    command=lambda: self.submit_start_point(start_form, ents))
        start_form_submit.pack(side=tk.BOTTOM)
        start_form_cancel = Button(start_form, text="Cancel",
                                    command=lambda: self.cancel_window(start_form))
        start_form_cancel.pack(side=tk.TOP)
        fields = ('x0', 'y0', 'z0')
        params = {'x0': self.begin[0], 'y0': self.begin[1], 'z0': self.begin[2]}
        ents = makeform(start_form, fields, params)

    def cancel_window(self, widget):
        """
        closes form window without updating parameters
        :param widget: window
        :return:
        """
        self.button_state(NORMAL)
        widget.destroy()

    def equation_show(self):
        """
        window with Lorenz equation parameters
        :return:
        """
        self.button_state(DISABLED)
        lorenz_form = tk.Toplevel(self)
        lorenz_form.title("Lorenz equation parameters")
        lorenz_form.geometry("300x150")
        lorenz_form_submit = Button(lorenz_form, text="Submit",
                                    command=lambda: self.submit_lorenz_para(lorenz_form, ents))
        lorenz_form_submit.pack(side=tk.BOTTOM)
        lorenz_form_cancel = Button(lorenz_form, text="Cancel",
                                    command=lambda: self.cancel_window(lorenz_form))
        lorenz_form_cancel.pack(side=tk.TOP)
        fields = ('sigma', 'rho', 'beta')
        params = self.equation.get_params()
        ents = makeform(lorenz_form, fields, params)

    def taylor(self):
        """
        window with Taylor solver parameters
        :return:
        """
        self.button_state(DISABLED)
        taylor_form = tk.Toplevel(self)
        taylor_form.title("Taylor solver parameters")
        taylor_form.geometry("300x150")
        taylor_form_submit = Button(taylor_form, text="Submit",
                                    command=lambda: self.submit_taylor_para(taylor_form, ents))
        taylor_form_submit.pack(side=tk.BOTTOM)
        taylor_form_cancel = Button(taylor_form, text="Cancel",
                                    command=lambda: self.cancel_window(taylor_form))
        taylor_form_cancel.pack(side=tk.TOP)
        fields = ('order', 'time step')
        params = dict(self.integrators['Taylor'].get_solver_params())
        ents = makeform(taylor_form, fields, params)

    def runge(self):
        """
        window for 4th order Runge-Kutta solver parameters
        :return:
        """
        self.button_state(DISABLED)
        runge_form = tk.Toplevel(self)
        runge_form.title("Runge-Kutta solver parameters")
        runge_form.geometry("300x150")
        runge_form_submit = Button(runge_form, text="Submit",
                                   command=lambda: self.submit_runge_para(runge_form, ents))
        runge_form_submit.pack(side=tk.BOTTOM)
        runge_form_cancel = Button(runge_form, text="Cancel",
                                   command=lambda: self.cancel_window(runge_form))
        runge_form_cancel.pack(side=tk.TOP)
        fields = ('time step')
        params = dict(self.integrators['Runge-Kutta'].get_solver_params())
        ents = makeform(runge_form, fields, params)

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

    def axes_switch(self):
        """
        switches axes labels, visibility
        :return:
        """
        if self.ax_state:
            self.axes.set_axis_off()
            self.btn_text_axes.set("Axes off")
        else:
            self.axes.set_axis_on()
            self.btn_text_axes.set("Axes on")
        self.ax_state = not self.ax_state

    def button_state(self, state):
        """
        sets all buttons state to state
        :param state: STATE from tk
        :return:
        """
        self.lorenz['state'] = state
        self.rungeint['state'] = state
        self.taylint['state'] = state
        self.starting_point['state'] = state
