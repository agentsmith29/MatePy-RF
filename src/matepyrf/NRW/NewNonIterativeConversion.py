import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import logging
from rich.logging import RichHandler

from scipy.constants import c as c_const

from .NRWBase import NRWBase

class NewNonIterative(NRWBase):

    def __init__(self, measurement_data: rf.Network, f_c, L, l_p1=0, l_p2=0, name="NewNonIterative",):
        super().__init__(measurement_data, f_c= f_c, L=L, l_p1=l_p1, l_p2=l_p2)

     
        self.mu_r = self.permeability(self.freq, self.S11, self.S21, self.n)
        self.eps_r = self.permittivity(self.freq, self.S11, self.S21, 1, self.n)
        # filter all values that are < 10 and >= 1 in self.eps_r
        #self.eps_r = np.where(np.abs(self.eps_r) < 10, 10, self.eps_r)
        #self.eps_r = np.where(np.abs(self.eps_r) >= 1, 1, self.eps_r)

      
        self.abs_epsr, self.tanDelta = self.convert1(self.eps_r)
        self.logger.info(f"Relative permittivity: {self.eps_r}")
        
    def mu_eff(self, lam_og, delta, beta):
        """
        Calculate the effective permeability mu_eff from S11 and S21.
        mu_eff = (1 + S11) / (1 - S11)
        """

        mu_eff = lam_og*delta*beta
        self.logger.debug(f"Effective permeability mu_eff = {self.rect(mu_eff)}")
        return np.zeros_like(self.freq) + 1  # ensure mu_r is an array of the same shape as f
        
    def eps_eff(self, lam_og, beta, delta):
        """
        Calculate the effective permittivity eps_eff from S11 and S21.
        eps_eff = (1 + S11) / (1 - S11)
        """        
        #eps_eff = np.power(_lam_og*_beta, n+1)*np.power(_delta, n-1)
        # eps_eff = lam_og*beta/delta
        eps_eff = np.power(lam_og*beta, 2)
        self.logger.debug(f"Effective permittivity eps_eff = {self.rect(eps_eff)} ({self.polar(eps_eff)})")
        return eps_eff

    def permeability(self, f, S11, S21, n: int):
        """
        Calculate the relative permeability mu_r from S11 and S21.
        mu_r = (1 + S11) / (1 - S11)
        """
        _lam_0 = c_const / f  # free space wavelength
        _X = self.calc_X(S11, S21)
        _Gamma = self.calc_Gamma1(_X)
        _T = self.calc_T(S11, S21, _Gamma)
        # self.logger.debug("[4.1]  From equation 1.5 ([1], P20, Eq. 1.5) , calculate alpha = ln(1/T)")
        _alpha = self.calc_alpha(_T, n)
        # self.logger.debug("[4.2]  Calculate beta = 1/Lambda")
        _beta = self.calc_beta(_alpha, self.sample_length, n)
        # self.logger.debug("[4.3]  Calculate delta = (1 + Gamma) / (1 - Gamma)")
        _delta = self.calc_z(_Gamma)
        # self.logger.debug("[4.4]  Caculate lam_og = 1 / (np.sqrt((1/lam_0**2) - 1/np.power(lam_c**2)))")
        _lam_og = self.calc_lam_og(_lam_0, self.lam_c)

        mu_r = _beta * _delta * _lam_og
        self.logger.info(f"Calculated relative permeability mu_r(n = {n}) = {mu_r}")
        return np.zeros_like(self.freq) + 1  # ensure mu_r is an array of the same shape as f

    def permittivity(self, f, S11, S21, mu_r, n: int):
        """
        Calculate the relative permittivity eps_r from S11 and S21.
        eps_r = (1 + S11) / (1 - S11)
        """
        _lam_0 = c_const / f  # free space wavelength
        _X = self.calc_X(S11, S21)
        _Gamma = self.calc_Gamma1(_X)
        _T = self.calc_T(S11, S21, _Gamma)
        # self.logger.debug("[4.1]  From equation 1.5 ([1], P20, Eq. 1.5) , calculate alpha = ln(1/T)")
        _alpha = self.calc_alpha(_T, n)
        # self.logger.debug("[4.2]  Calculate beta = 1/Lambda")
        _beta = self.calc_beta(_alpha, self.sample_length, n)
        # self.logger.debug("[4.3]  Calculate delta = (1 + Gamma) / (1 - Gamma)")
        _delta = self.calc_z(_Gamma)
        # self.logger.debug("[4.4]  Caculate lam_og = 1 / (np.sqrt((1/lam_0**2) - 1/np.power(lam_c**2)))")
        _lam_og = self.calc_lam_og(_lam_0, self.lam_c)

        _eps_eff = self.eps_eff(_lam_og, _beta, _delta)
        # if mu_r is None:
        #     _mu_eff = self.mu_eff(_lam_og, _delta, _beta)
        # else:
        #     _mu_eff = mu_r
        #     self.mu_r = np.zeros_like(self.f) + _mu_eff  # ensure mu_r is an array of the same shape as f
        _mu_eff = np.zeros_like(self.freq) + mu_r  # ensure mu_r is an array of the same shape as f

        eps_r = ( (1 - (np.power(_lam_0, 2) / np.power(self.lam_c, 2)) )*_eps_eff + 
                 ((np.power(_lam_0, 2) /np.power(self.lam_c, 2)))*(1/_mu_eff) )
       
        self.logger.debug(f"Calculated relative permittivity eps_r(n = {n}) = {self.rect(eps_r)} ({self.polar(eps_r)})")
        return eps_r
