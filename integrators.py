"""
module with several solvers for Lorenz equation implemented
"""
import numpy as np


class LorenzEquation:
    """
    object that conveys generic Lorenz equation with necessary paramteres for evaluation
    dx/dt = sigma * (y-x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    """

    def __init__(self, sigma=10.0, rho=28.0, beta=8 / 3):
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.beta = float(beta)

    def eval(self, point):
        """
        :param point: float precision, possibly ndarray 3 values
        :return: evaluation of lorentz equation for specified point as ndarray
        """
        assert len(point) == 3
        value_x = self.sigma * (point[1] - point[0])
        value_y = point[0] * (self.rho - point[2]) - point[1]
        value_z = point[0] * point[1] - self.beta * point[2]
        return np.asarray([value_x, value_y, value_z])

    def get_params(self):
        """
        :return: dictonary of equation parameters
        """
        return {"sigma": self.sigma, "rho": self.rho, "beta": self.beta}

    def set_params(self, params):
        """
        sets the parameters of equation
        params: dictionary of parameters
        """
        self.sigma = params['sigma']
        self.rho = params['rho']
        self.beta = params['beta']


class TaylorIntegrator:
    """
    solver for Lorenz equation implementing Taylor method of possibly large order
    higher order needs smaller time step as auxiliary computations with derivatives
    tends to get larger and can get to overflow
    """

    def __init__(self, lorenz: LorenzEquation):
        self.lorenz = lorenz
        self.params = {'order': 4, 'time step': 0.01}

    def step(self, initial_values):
        """
        computes one step with Taylor method
        initial_values: possible ndarray, point in 3d
        return: point in 3d after one step in ndarray
        """
        assert len(initial_values) == 3
        fx0, fy0, fz0 = self.lorenz.eval(initial_values)
        vector_of_values = np.zeros((3, self.params['order']))
        vector_of_derivatives = np.zeros((3, self.params['order']))
        vector_of_values[:, 0] = initial_values
        vector_of_derivatives[:, 0] = [fx0, fy0, fz0]
        vector_of_values[:, 1] = vector_of_derivatives[:, 0]

        for i in range(1, self.params['order'] - 1):
            vector_of_derivatives[:, i] = self.lorenz.eval(vector_of_values[:, i])
            mody = np.fromfunction(lambda j, k=i: 2 * vector_of_values[0, k - j]
                                                  * vector_of_values[2, j], dtype=int, shape=[1, i])
            modz = np.fromfunction(lambda j, k=i: 2 * vector_of_values[0, k - j]
                                                  * vector_of_values[1, j], dtype=int, shape=[1, i])
            vector_of_derivatives[1, i] -= np.sum(mody)
            vector_of_derivatives[2, i] += np.sum(modz)
            vector_of_values[:, i + 1] = vector_of_derivatives[:, i] / (i + 1)

        # print(vector_of_derivatives)
        def helper(tempv, iteracja=(self.params['order'] - 1)):
            if iteracja != 0:
                return helper(tempv * self.params['time step'] +
                              vector_of_values[:, iteracja - 1], iteracja - 1)
            return tempv

        wynik = helper(vector_of_values[:, self.params['order'] - 1])
        return wynik

    def get_lorenz_params(self):
        """
        :return: dictionary of parameters of Lorenz equation
        """
        return self.lorenz.get_params()

    def set_lorenz_params(self, params):
        """
        :param params: dictionary of parameters for Lorentz equation
        """
        self.lorenz.set_params(params)

    def get_solver_params(self):
        """
        time step and order
        :return: dict of params
        """
        return self.params

    def set_solver_params(self, params):
        """
        :param params: dict of parameters, ie. time step and order
        """
        self.params['time step'] = params['time step']
        self.params['order'] = int(params['order'])


class RungeKutta4th:
    """
    Runge Kutta 4th order solver
    """

    def __init__(self, lorenz: LorenzEquation):
        self.lorenz = lorenz
        self.params = {'time step': 0.01}

    def step(self, initial_values):
        """
        :param initial_values: 3d coord
        :return: 3d coord after 1 step as ndarray
        """
        assert len(initial_values) == 3
        init = np.asarray(initial_values)
        first = self.lorenz.eval(init)
        second = self.lorenz.eval(init + 0.5 * self.params['time step'] * first)
        third = self.lorenz.eval(init + 0.5 * self.params['time step'] * second)
        fourth = self.lorenz.eval(init + self.params['time step'] * third)
        result = init + self.params['time step'] * (first + 2 * second + 2 * third + fourth) / 6
        return result

    def get_lorenz_params(self):
        """
        :return: dictionary of parameters
        """
        return self.lorenz.get_params()

    def set_lorenz_params(self, params):
        """
        :param params: dictionary of parameters for Lorentz equation
        """
        self.lorenz.set_params(params)

    def get_solver_params(self):
        """
        time step and order
        :return: dict of params
        """
        return self.params

    def set_solver_params(self, params):
        """
        :param params: dict of time step and order
        """
        self.params['time step'] = params['time step']
