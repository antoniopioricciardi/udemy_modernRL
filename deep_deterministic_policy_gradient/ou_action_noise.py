import numpy as np


class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        """

        :param mu: mean for the noise
        :param sigma: std dev
        :param theta:
        :param dt: time parameter
        :param x0: starting value
        """
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    # allows to use name of the object as a function
    def __call__(self):
        """
        get the temporal correlation of the noise
        :return:
        """
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)