from scipy.constants import c, mu_0, epsilon_0
import numpy as np

class Waveguide():
    """
        Represents a rectangular waveguide.
    """

    def __init__(self, L, a, b =None, name='WR12'):
        self.L = L
        self.a = a
        if b is None:
            self.b = a / 2  # Default height is half the width
        else:
            self.b = b
            
        if name is None:
            if self.a == "3.0988" and self.b == self.a/2:
                self._name = "WR12"
        else:
            self._name = name


    def cutoff_frequency(self, eps_r=1, mu_r=1):
        """
        Calculate the cutoff frequency for a rectangular waveguide.
        :param m: mode number in the x direction
        :param n: mode number in the y direction
        :return: cutoff frequency in Hz
        """
        f_c = 1 / (2*self.a*np.sqrt(epsilon_0 * mu_0 * eps_r * mu_r))
        return f_c
    
    def cutoff_wavelength(self, eps_r=1, mu_r=1):
        """
        Calculate the cutoff wavelength for a rectangular waveguide.
        :param eps_r: relative permittivity
        :param mu_r: relative permeability
        :return: cutoff wavelength in meters
        """
        return c / self.cutoff_frequency(eps_r, mu_r)

    @property
    def width(self):
        """
        Calculate the width of the waveguide.
        :return: width in meters
        """
        return self.a
    
    @property
    def height(self):
        """
        Calculate the height of the waveguide.
        :return: height in meters
        """
        return self.b
    
    @property
    def length(self):
        """
        Calculate the length of the waveguide.
        :return: length in meters
        """
        return self.L
    
    @property
    def name(self):
        """
        Get the name of the waveguide.
        :return: name of the waveguide
        """
        return self._name
    
    def __repr__(self):
        return f"Waveguide(name={self.name}, L={self.L}, a={self.a}, b={self.b})"   
    
    def __str__(self):
        return f"Waveguide: {self.name}, Length: {self.L} m, Width: {self.a} m, Height: {self.b} m"