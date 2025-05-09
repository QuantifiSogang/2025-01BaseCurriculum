import numpy as np
import pandas as pd
from typing import Union
class MonteCarloSimulation:
    def __init__(self, interest_rate: float, initial_price: float, maturity: float, sigma: float,
                 dividend_yield: float, nObs: int, slices: int, random_state: Union[bool, int] = False):
        np.random.seed(random_state if isinstance(random_state, int) else None)
        self.S0 = initial_price
        self.T = maturity
        self.dividend_yield = dividend_yield
        self.nObs = nObs
        self.slices = slices
        self.dt = self.T / self.slices
        self.mu = interest_rate
        self.sigma = sigma
        self.paths = np.zeros((nObs, slices + 1))
        self.paths[:, 0] = self.S0

    def _generate_standard_normal_random_variables(self, correlation_matrix=None):
        if correlation_matrix is None:
            z = np.random.standard_normal((self.nObs, self.slices))
        else:
            z = np.random.multivariate_normal(np.zeros(2), correlation_matrix, (self.nObs, self.slices))
        return z

    def geometric_brownian_motion(self):
        z = self._generate_standard_normal_random_variables()
        drift = (self.mu - self.dividend_yield - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * z
        self.paths[:, 1:] = self.S0 * np.exp(np.cumsum(drift + diffusion, axis=1))
        return self.paths

    def vasicek_model(self, kappa: float, theta: float, sigma_r: float):
        r = np.zeros((self.nObs, self.slices + 1))
        r[:, 0] = self.mu
        z = self._generate_standard_normal_random_variables()
        for t in range(1, self.slices + 1):
            dr = kappa * (theta - r[:, t-1]) * self.dt + sigma_r * z[:, t-1] * np.sqrt(self.dt)
            r[:, t] = r[:, t-1] + dr
        return r

    def cox_ingersoll_ross_model(self, kappa: float, theta: float, sigma_r: float):
        if not 2 * kappa * theta > sigma_r ** 2:
            raise ValueError("2 * kappa * theta must be greater than sigma_r ** 2 for CIR model.")
        r = np.zeros((self.nObs, self.slices + 1))
        r[:, 0] = self.mu
        z = self._generate_standard_normal_random_variables()
        for t in range(1, self.slices + 1):
            dr = kappa * (theta - r[:, t-1]) * self.dt + sigma_r * np.sqrt(r[:, t-1]) * z[:, t-1] * np.sqrt(self.dt)
            r[:, t] = np.maximum(r[:, t-1] + dr, 0)  # Ensure non-negative rates
        return r

    def heston_model(self, kappa: float, theta: float, sigma_v: float, rho: float = 0.0):
        correlation_matrix = np.array([[1, rho], [rho, 1]])
        z = self._generate_standard_normal_random_variables(correlation_matrix)
        s = np.zeros((self.nObs, self.slices + 1))
        v = np.zeros((self.nObs, self.slices + 1))
        s[:, 0] = self.S0
        v[:, 0] = self.sigma ** 2
        for t in range(1, self.slices + 1):
            dv = kappa * (theta - v[:, t-1]) * self.dt + sigma_v * np.sqrt(v[:, t-1]) * z[:, t-1, 1] * np.sqrt(self.dt)
            ds = s[:, t-1] * (self.mu * self.dt + np.sqrt(np.maximum(v[:, t-1], 0)) * z[:, t-1, 0] * np.sqrt(self.dt))
            v[:, t] = np.maximum(v[:, t-1] + dv, 0)  # Ensure non-negative variance
            s[:, t] = s[:, t-1] + ds
        return s

    def ornstein_uhlenbeck(self, kappa: float, theta: float, sigma: float):
        ou_paths = np.zeros((self.nObs, self.slices + 1))
        ou_paths[:, 0] = self.S0
        z = self._generate_standard_normal_random_variables()
        for t in range(1, self.slices + 1):
            ou_paths[:, t] = ou_paths[:, t - 1] + kappa * (theta - ou_paths[:, t - 1]) * self.dt + sigma * z[:,
                                                                                                           t - 1] * np.sqrt(
                self.dt)
        return ou_paths

    def jump_diffusion_model(self, jump_intensity: float, jump_mean: float, jump_std: float):
        jd_paths = np.zeros((self.nObs, self.slices + 1))
        jd_paths[:, 0] = self.S0
        z = self._generate_standard_normal_random_variables()
        jumps = np.random.poisson(lam=jump_intensity * self.dt, size=(self.nObs, self.slices))
        for t in range(1, self.slices + 1):
            jump_sizes = np.random.normal(loc=jump_mean, scale=jump_std, size=(self.nObs, jumps[:, t - 1].max()))
            total_jump = np.array([jump_sizes[i, :jumps[i, t - 1]].sum() for i in range(self.nObs)])
            jd_paths[:, t] = jd_paths[:, t - 1] * np.exp(
                (self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * z[:, t - 1] * np.sqrt(self.dt)) + total_jump
        return jd_paths