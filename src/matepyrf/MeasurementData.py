import skrf as rf
import numpy as np
import pandas as pd
import logging
from rich.logging import RichHandler
from scipy.constants import c as c_const
import pathlib

from .Waveguide import Waveguide


class MeasurementData():
    
    
    def __init__(self, measurement_data: rf.Network, waveguide_system: Waveguide, sample_length, l_p1, l_p2):
        """
            Base Class for the Nicholson-Ross-Weir conversion.
        Args:
            measurement_data: skrf.Network object containing the S-parameters
            f_c: cutoff frequency in Hz
            L: length of the sample in m
            l_p1: Position of the sample, from the port1
            l_p2: Position of the sample, from the port2
        """
        self.waveguide_system = waveguide_system
        self.logger.info(f"Waveguide system: {self.waveguide_system}")

        self.measurement_data = measurement_data
        self.logger.info(f"Taking measurement data: \n{measurement_data}")

        self.f_c = waveguide_system.cutoff_frequency()  # cutoff frequency in Hz
        self.lam_c = waveguide_system.cutoff_wavelength()
        self.logger.info(f"Cutoff frequency: {self.f_c} Hz, Cutoff wavelength: {self.lam_c} m")

        self.sample_length = sample_length      # length of the sample in m
        self.l_p1 = l_p1  # position of the sample from port1 in m
        self.l_p2 = l_p2  # position of the sample from port2 in m
        self.wg_length = self.l_p1 + self.l_p2 + self.sample_length  # total length of the waveguide including the sample in m


        self._s_params = self._create_s_params_dataframe(self.measurement_data)#, clip=(1026, 1028)
        self._deembedd(l_p1=l_p1, l_p2=l_p2)  # de-embed S-parameters based on the positions of the sample from port1 and port2

        self.mu_r = np.zeros_like(self.f)  # relative permeability, initialized to 1
        self.eps_r = np.zeros_like(self.f)  # relative permeability, initialized to 
        

        # Storing temporary calulöation resuls
        self._tmp_folder = pathlib.Path("tmp")  # Temporary folder for storing calculation results
        # use pathlib to check if folder exists, if not create it
        if not self._tmp_folder.exists():
            self._tmp_folder.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created temporary folder: {self._tmp_folder}")
        else:
            self.logger.info(f"Temporary folder already exists: {self._tmp_folder}")
 
    def _create_s_params_dataframe(self, measurement_data: rf.Network, clip: tuple = None) -> pd.DataFrame:

        s_data = measurement_data.s.reshape(-1, 4).astype(complex)
        s_dB_data = measurement_data.s_db.reshape(-1, 4).real
        f = measurement_data.frequency.f.reshape(-1, 1).real  # frequency in Hz, reshaped to a column vector
        
        # now  clip so only the row between clip[0] and clip[1] are used
        if clip is not None and len(clip) == 2 and clip[0] < clip[1]:
            self.logger.warning(f"Clipping data to rows {clip[0]}:{clip[1]}")
            # print(f"Clipping data to rows {clip[0]}:{clip[1]}")
            s_data = s_data[clip[0]:clip[1], :]
            s_dB_data = s_dB_data[clip[0]:clip[1], :]
            tau_mea = tau_mea[clip[0]:clip[1], :]
            f = f[clip[0]:clip[1], :]

        _s_params = pd.DataFrame(
            columns=['freq', 'S11', 'S12', 'S21', 'S22', 'S11dB', 'S12dB', 'S21dB', 'S22dB'],
        )
        _s_params['freq'] = f.flatten()  # frequency in Hz
        _s_params['S11'] = s_data[:, 0]  # S11 parameter
        _s_params['S12'] = s_data[:, 1]  # S12 parameter
        _s_params['S21'] = s_data[:, 2]  # S21 parameter
        _s_params['S22'] = s_data[:, 3] # S22 parameter
        _s_params['S11dB'] = s_dB_data[:, 0]  # S11 in dB
        _s_params['S12dB'] = s_dB_data[:, 1]  # S12 in dB
        _s_params['S21dB'] = s_dB_data[:, 2]  # S21 in dB
        _s_params['S22dB'] = s_dB_data[:, 3]  # S22 in dB
        
        _s_params["freq"] = _s_params["freq"].astype(np.longdouble)  # ensure frequency is float type
        _s_params["S11"] = _s_params["S11"].astype(complex)  # ensure S11 is complex type 
        _s_params["S21"] = _s_params["S21"].astype(complex)  # ensure S21 is complex type
        _s_params["S12"] = _s_params["S12"].astype(complex)  # ensure S12 is complex type
        _s_params["S22"] = _s_params["S22"].astype(complex)  # ensure S22 is complex type
        _s_params["S11dB"] = _s_params["S11dB"].astype(np.longdouble)  # ensure S11dB is float type
        _s_params["S12dB"] = _s_params["S12dB"].astype(np.longdouble)  # ensure S12dB is float type
        _s_params["S21dB"] = _s_params["S21dB"].astype(np.longdouble)  # ensure S21dB is float type
        _s_params["S22dB"] = _s_params["S22dB"].astype(np.longdouble)  # ensure S22dB is float type

        self.logger.info(f"Created S-parameters DataFrame with shape: {_s_params.shape} and columns: {_s_params.columns.tolist()}")
        self.logger.debug(f"S-parameters DataFrame:\n{_s_params}")
        return _s_params
    
    @property
    def s_params(self):
        """
        Returns the S-parameters as a pandas DataFrame.
        """
        return self._s_params
    
    @property 
    def S11(self):
        """
        Returns the S11 parameter as a numpy array.
        """
        return self.s_params['S11'].values
    
    @S11.setter
    def S11(self, value):
        """
        Sets the S11 parameter in the S-parameters DataFrame.
        """
        self.s_params['S11'] = value

    @property
    def S11dB(self):
        """
        Returns the S11 parameter in dB as a numpy array.
        """
        return self.s_params['S11dB'].values
    
    @S11dB.setter
    def S11dB(self, value):
        """
        Sets the S11 parameter in dB in the S-parameters DataFrame.
        """
        self.s_params['S11dB'] = value

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def S21(self):
        """
        Returns the S21 parameter as a numpy array.
        """
        return self.s_params['S21'].values
    
    @S21.setter
    def S21(self, value):
        """
        Sets the S21 parameter in the S-parameters DataFrame.
        """
        self.s_params['S21'] = value

    @property
    def S21dB(self):
        """
        Returns the S21 parameter in dB as a numpy array.
        """
        return self.s_params['S21dB'].values
    
    @S21dB.setter
    def S21dB(self, value):
        """
        Sets the S21 parameter in dB in the S-parameters DataFrame.
        """
        self.s_params['S21dB'] = value

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def S12(self):
        """
        Returns the S12 parameter as a numpy array.
        """
        return self.s_params['S12'].values
    
    @S12.setter
    def S12(self, value):
        """
        Sets the S12 parameter in the S-parameters DataFrame.
        """
        self.s_params['S12'] = value

    @property
    def S12dB(self):
        """
        Returns the S12 parameter in dB as a numpy array.
        """
        return self.s_params['S12dB'].values
    
    @S12dB.setter
    def S12dB(self, value):
        """
        Sets the S12 parameter in dB in the S-parameters DataFrame.
        """
        self.s_params['S12dB'] = value

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def S22(self):
        """
        Returns the S22 parameter as a numpy array.
        """
        return self.s_params['S22'].values
    
    @S22.setter
    def S22(self, value):
        """
        Sets the S22 parameter in the S-parameters DataFrame.
        """
        self.s_params['S22'] = value

    @property
    def S22dB(self):
        """
        Returns the S22 parameter in dB as a numpy array.
        """
        return self.s_params['S22dB'].values
    
    @S22dB.setter
    def S22dB(self, value):
        """
        Sets the S22 parameter in dB in the S-parameters DataFrame.
        """
        self.s_params['S22dB'] = value

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def f(self):
        """
        Returns the frequency as a numpy array.
        """
        return self.s_params['freq'].values
    
    @f.setter
    def f(self, value):
        """
        Sets the frequency in the S-parameters DataFrame.
        """
        self.s_params['freq'] = value

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def calc_R(self, freq, lam_c, l_p):
        """
            reference plane transformation expressions
        """
        _R = np.exp(self.gamma(freq, lam_c, eps_r=1, mu_r=1) * l_p)
        return _R
    

    def _deembedd(self, l_p1=0, l_p2=0):
        """
        De-embed S-parameters based on the positions of the sample from port1 and port2.
        This method is a placeholder and should be implemented in subclasses.
        
        Args:
            s_params: skrf.Network object containing the S-parameters
            l_p1: Position of the sample from port1 in m
            l_p2: Position of the sample from port2 in m
        """
        if l_p1 < 0 or l_p1 > 0:
            self.logger.info(f"De-embedding S-parameters with sample length L = {self.sample_length}, "
                             f"Port 1 position = {l_p1} Port 2 position = {l_p2} m. "
                             f"Total measurement length L={self.wg_length} m")
            self.logger.debug(self.wg_drawing(self.sample_length, l_p1, l_p2))

            self.S11 = self.s_params.apply(
                lambda row: row['S11'] * np.power(self.calc_R(row['freq'], self.lam_c, l_p1), 2), axis=1)

            self.logger.info(f"De-embedded S11: {self.str_array_repr(self.s_params['S11'].values)}")
        else:
            self.logger.warning(f"No de-embedding for S11, l_p1 = {l_p1} m")

        if (l_p1 < 0 or l_p1 > 0) and (l_p2 < 0 or l_p2 > 0):
            self.S21 = self.s_params.apply(
                lambda row: -row['S21'] * 
                   ( self.calc_R(row['freq'], self.lam_c, l_p1)*self.calc_R(row['freq'], self.lam_c, l_p2)),  axis=1)
            self.logger.info(f"De-embedded S21: {self.str_array_repr(self.s_params['S21'].values)}")
            self.logger.debug(f"De-embedded S-parameters DataFrame:\n{self.s_params}")
        else:
            # conjungate S21 if l_p1 or l_p2 is not set
            self.S21 = self.s_params['S21'].values
            self.logger.warning(f"No de-embedding for S21, l_p1 = {l_p1} m, l_p2 = {l_p2} m")

    def wg_drawing(self, L, l_p1, l_p2):
        return (f"Sample Drawing:\n"
                  f"|----------------#####----------|\n"
                  f"P1              |     |        P2\n"
                  f"|<-----lp1----->|<-L->|<--lp2-->|\n"
                  f"|<----------WG length---------->|\n"
                  f"lp1={l_p1:.2e}, L={L:.2e}, lp2={l_p2:.2e}, wg_length={self.wg_length:.2e}")
    
    # ==================================================================================================================
    # print helper functions
    # ==================================================================================================================
    def str_array_repr(self, arr, n=2, polar=True):
        """
            Retuns any compolex list, tuple, ndarray as [] ([]) with the maximum of n values an ... inbetween
        """
        def _repr(_n):
            if polar:
                _arr = [f"({self.polar(x)})" for x in arr]
                return f"[{', '.join(map(str, _arr[:_n]))}, ..., {', '.join(map(str, _arr[-1-_n:-1]))}]"
            else:
                _arr = [f"({self.rect(x)})" for x in arr]
                return f"[{', '.join(map(str, _arr[:_n]))}, ..., {', '.join(map(str, _arr[-_n:]))}]"
            
        if len(arr) <= n:
            return str(arr)
        elif n is None or n <= 0:
            n = len(arr) // 2

        
        return _repr(n)

        
    def polar(self, num, decimals=2):
        return f"{np.round(np.abs(num),decimals)} ∠ {np.round(np.angle(num, deg=True), decimals)}"
    
    def rect(self, num, decimals=2):
       return f"{np.round(np.real(num),decimals)} + j{np.round(np.imag(num), decimals)}"