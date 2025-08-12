
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import logging
from rich.logging import RichHandler
from scipy.constants import c as c_const

from .NRWBase import NRWBase
from ..Waveguide import Waveguide

class NicholsonRossWeirConverstion(NRWBase):

    def __init__(self, measurement_data: rf.Network, 
                 waveguide_system: Waveguide,
                 sample_length, l_p1=0, l_p2=0, name="nicholson_ross_weir_conversion"):
        super().__init__(measurement_data, waveguide_system= waveguide_system, L=sample_length, l_p1=l_p1, l_p2=l_p2)
        self.name = name

      

        # (6) Calculate delta from equation (1.4)
        self.mu_r = self.permeability(self.Z1, 0)    
        self.eps_r = self.permittivity(self.mu_r, 0)    

        # check if mu_r is over 1, and under 10
        # self.mu_r = np.where(np.abs(self.mu_r) < 1, 1, self.mu_r)
        # self.mu_r = np.where(np.abs(self.mu_r) > 5, 5, self.mu_r)
        # # cut the first value and interpolate it from the remaining ones
        #if self.freq is not None and len(self.freq) > 2:
        #    self.mu_r = np.interp(self.freq, self.freq[1:], self.mu_r[1:])
        #    self.eps_r = np.interp(self.freq, self.freq[1:], self.eps_r[1:])
        # print(self.f, self.S11, self.S12, self.n)
        # print(self.mu_r)
        # check if eps_r is over 1, and under 10
        # self.eps_r = np.where(np.abs(self.eps_r) < 1, 1, self.eps_r)
        # self.eps_r = np.where(np.abs(self.eps_r) > 10, 10, self.eps_r)
        self.abs_epsr, self.tanDelta = self.convert1(self.eps_r)
        # Save eps_r and mu_r  as file
        # with open(f"{name}_eps_mu.txt", "w") as f:
        #     f.write(f"Relative permittivity: {self.eps_r} ({self.rect2pol(self.eps_r)})\n")
        #     f.write(f"Relative permeability: {self.mu_r} ({self.rect2pol(self.mu_r)})\n")
        #     f.write(f"Loss tangent: {self.tanDelta}\n")
        # with f as
        self.logger.debug(self.interm_calc_df)
        self.logger.info(f"Relative permittivity: {self.str_array_repr(self.abs_epsr)} ({self.str_array_repr(self.abs_epsr)}), "
                         f"relative permeability: {self.str_array_repr(self.mu_r)} ({self.str_array_repr(self.mu_r)}) at frequency {self.str_array_repr(self.freq)}Hz")

    def permeability(self, Z1, n: int):
        """	
            Calculate the relative permeability mu_r from the given parameters.
            mu_r = (1 + Gamma_1)/1
            [1], Page 20, Equation 1.4
        """
        #_lam_0 = c_const / f  # free space wavelength
        #_X = self.calc_X(S11, S21)
        #_Gamma = self.calc_Gamma1(_X)
        #_T = self.calc_Z1(S11, S21, _Gamma)
        # self.logger.debug("[4.1]  From equation 1.5 ([1], P20, Eq. 1.5) , calculate alpha = ln(1/T)")
        _alpha = self.calc_alpha(Z1, n)
        # self.logger.debug("[4.2]  Calculate beta = 1/Lambda")
        _beta = self.calc_beta(_alpha, self.sample_length)
        # self.logger.debug("[4.3]  Calculate delta = (1 + Gamma) / (1 - Gamma)")
       
        self.logger.debug("Calculating relative permeability mu_r: mu_r = (_beta * _delta) * _lam_og")
        mu_r = (_beta * self.z) * self.lam_og
        self.logger.info(f"[Calculated relative permeability mu_r(n = {n}) = {self.str_array_repr(mu_r, polar=False)}")
        return mu_r

    def permittivity(self, mu_r, n: int):
        # (1) Calculate X from S11 and S21
        #         self.logger.debug("[4.1]  From equation 1.5 ([1], P20, Eq. 1.5) , calculate alpha = ln(1/T)")
    
        
        term1 = np.power(self.lam0, 2) / np.power(self.lam_c, 2)
        self.logger.debug(f"Calculating eps_r =  mu_r * ((1/z)^2) * (1 - (lam0^2/lamc^2)) + (lam0^2/lamc^2)*1/mu_r")
        eps_r =  mu_r * np.power((1/self.z) ,2) * (1 - (term1)) + term1/mu_r
        self.logger.info(f"Calculated relative permittivity: eps_r(n = {n}) = {self.str_array_repr(eps_r, polar=False)}")
        return eps_r
