import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import logging
from rich.logging import RichHandler
from scipy.constants import c as c_const
from matplotlib.widgets import Slider
import pandas as pd
from scipy import stats
from matepyrf.MeasurementData import MeasurementData
from tqdm import tqdm
import io
import time
import unicodedata as ud
from scipy.stats import zscore
from scipy.interpolate import UnivariateSpline
import numpy as np
import scipy.integrate as integrate
from scipy.signal import find_peaks

from ..Waveguide import Waveguide

"""
This code defines a base class for the Nicholson-Ross-Weir conversion of S-parameters to dielectric properties.

See the following publications for further details:
[1] Nicholson, Ross, and Weir, "A Generalized Method for Extracting the Dielectric Properties of Materials from 
    Measured Scattering Parameters", IEEE Transactions on Microwave Theory and Techniques, 1976.
[2] https://nvlpubs.nist.gov/nistpubs/Legacy/TN/nbstechnicalnote1355r.pdf
"""

# UNicode characters for Greek letters, for better readability in the log output
STR_Alpha = ud.lookup("GREEK CAPITAL LETTER ALPHA")
STR_alpha = ud.lookup("GREEK SMALL LETTER ALPHA")
STR_Beta = ud.lookup("GREEK CAPITAL LETTER BETA")
STR_beta = ud.lookup("GREEK SMALL LETTER BETA")
STR_Gamma = ud.lookup("GREEK CAPITAL LETTER GAMMA")
STR_gamma = ud.lookup("GREEK SMALL LETTER GAMMA")
STR_Delta = ud.lookup("GREEK CAPITAL LETTER DELTA")
STR_delta = ud.lookup("GREEK SMALL LETTER DELTA")
STR_Lambda = ud.lookup("GREEK CAPITAL LETTER LAMDA")
STR_lambda = ud.lookup("GREEK SMALL LETTER LAMDA")
STR_pi = ud.lookup("GREEK SMALL LETTER PI")
STR_tau = ud.lookup("GREEK SMALL LETTER TAU")

class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        
        self.logger = logger
        self.level = level or logging.INFO

        self.logger.info(f"Initializing TqdmToLogger with logger {logger} and level {level}.")

    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.debug(self.buf)

class NRWBase(MeasurementData):

    # ==================================================================================================================
    # 
    # ==================================================================================================================
    def __init__(self, measurement_data,  
                 waveguide_system: Waveguide,
                 L, l_p1, l_p2):
        """
            Base Class for the Nicholson-Ross-Weir conversion.
        Args:
            measurement_data: skrf.Network object containing the S-parameters
            f_c: cutoff frequency in Hz
            L: length of the sample in m
            l_p1: Position of the sample, from the port1
            l_p2: Position of the sample, from the port2
        """
        self.logger = logging.getLogger(__name__)
        self.tqdm_out = TqdmToLogger(self.logger,level=logging.INFO)
        super().__init__(measurement_data, waveguide_system, sample_length=L, l_p1=l_p1, l_p2=l_p2)
        self.interm_calc_df = self._s_params.copy()  # intermediate calculations DataFrame, copy of the S-parameters DataFrame

        self.mu_r = np.zeros_like(self.freq)  # relative permeability, initialized to 1
        self.eps_r = np.zeros_like(self.freq)  # relative permeability, initialized to 
        
        
        self.X = self.calc_X(self.S11, self.S21)
        self.Gamma1 = self.calc_Gamma1(self.X)
        self.Z1 = self.calc_Z1(self.S11, self.S21, self.Gamma1)
        self.z = self.calc_z(self.Gamma1)
        self.lam_og = self.calc_lam_og(self.lam0, self.lam_c)
        

        try:
            # self.tau_mea = measurement_data.s21.group_delay.reshape(-1, 1).real
            self._s_params['tau_mea_S21']  = -np.gradient(np.unwrap(np.angle(measurement_data.s21.s.squeeze())), measurement_data.frequency.w, axis=0).reshape(-1, 1) # group delay in seconds
            # make a polyfit of _T
            # _polyfit_T = np.polyfit(self._scaled_f, np.unwrap(np.angle(_T)), 1)
            # self._s_params['_T'] = (np.angle(_T))  # store the polynomial fit of the phase of T
            # # _polyfit_T = _polyfit_T(self._scaled_f)  # evaluate the polynomial at the scaled frequency
            # self._s_params['polyfit_T'] = _polyfit_T[0]*self._scaled_f + _polyfit_T[1]  # store the polynomial fit
            # group delay in seconds
            #self.tau_mea = self.calc_tau_mea(self.freq, self.Z1)  # group delay in seconds
            #for __n in range(0, 5):
            self.tau_mea = self.calc_tau_mea(self.freq, self.Z1)  # group delay in seconds
            
            # self._s_params['tau_mea'] = -_polyfit_T[0]/(2*np.pi)  # group delay in seconds
            # self._s_params['v_g_mea']  =  self.sample_length / self._s_params['tau_mea'].values
            
            #v = c_const/np.sqrt(1*2.8)
            #self._s_params['length_calc'] = self._s_params['tau_mea'].values * (v * np.sqrt(1-(v/ (2*3.0988e-3*self.f))**2) )
            
            # peaks = find_peaks(-self.S11, height=-20, distance=10)
            
            # _delta_f = np.diff(measurement_data.frequency.f[peaks[0]])
            # peaks = [measurement_data.frequency.f[peaks[0]][i] + _delta_f[i]/2 for i in range(len(_delta_f))]  # filter out peaks with delta_f < 0.1
            # print(f"Found {peaks} peaks in S11 data: {_delta_f}")
            # # use the data and interpolate the delta_f value
            # # delta_f_interp = UnivariateSpline(peaks, _delta_f, s=0)(self._scaled_f)
            # # print(delta_f_interp)
            # # self._s_params['v_g_mea_peaks'] = 2*delta_f_interp*self.L
            # self._s_params['lam_g_mea']  = self._s_params['v_g_mea']/self.f
            
            
            # self._s_params['n']  = self.sample_length / self._s_params['lam_g_mea'].values
            # self.logger.debug(f"Calculated group delay from measurement data: {  self._s_params['tau_mea'].values}")
        except Exception as e:

            self.logger.warning(f"Group delay data not available: {e}")
            self._s_params['tau_mea'] = np.zeros((self.freq.shape[0], 1)).real  # initialize to zeros if not available
            raise e
            # Add group delay tau_g if available    

        self.logger.debug(f"S-Parameters DataFrame:\n{self._s_params}")
        self._s_params.to_excel(f'{self._tmp_folder}/s_parameters.xlsx', index=False)
        # Save S-Parameters to Excel file
        
        try:
            self.n = self.estimate_n_from_group_delay()  # estimate the number of wavelengths n
            self.n = np.zeros_like(self.freq, dtype=int) +4 # initialize n to zeros if estimation fails	
            # make a ramp from 5 to 7 with n=self.f datapoints
            #self.n = np.round(np.linspace(5, 7, len(self.f)), 0).astype(int)  # initialize n to a ramp from 5 to 7 if estimation fails
        except Exception as e:
            # self.logger.disabled = False
            self.n = np.zeros_like(self.freq, dtype=int)  # initialize n to zeros if estimation fails	
            self.logger.error(f"Error estimating n: {e}")
            raise e 

        # self.logger.disabled = False
    
    # ==================================================================================================================
    # Getter and setter
    # ==================================================================================================================
    @property
    def X(self):
        """
            Returns the calculated X parameter.
        """
        return self.interm_calc_df['X']
    
    @X.setter
    def X(self, value):
        """
            Sets the calculated X parameter.
        """
        self.interm_calc_df['X'] = value
    
    @property
    def Gamma1(self):
        """
            Returns the reflection coefficient Gamma.
        """
        return self.interm_calc_df['Gamma1']
    
    @Gamma1.setter
    def Gamma1(self, value):
        """
            Sets the reflection coefficient Gamma.
        """
        self.interm_calc_df['Gamma1'] = value
   
    @property
    def Z1(self):
        """
            Returns the transmission coefficient T.
        """
        return self.interm_calc_df['Z1']
    
    @Z1.setter
    def Z1(self, value):
        """
            Sets the transmission coefficient T.
        """
        self.interm_calc_df['Z1'] = value
    
    @property
    def tau_mea(self):
        """
            Returns the measured group delay.
        """
        return self._s_params['tau_mea']
    
    @tau_mea.setter
    def tau_mea(self, value):
        """
            Sets the measured group delay.
        """
        self._s_params['tau_mea'] = value
    
    @property
    def n(self):
        """
            Returns the number of wavelengths n to resolve the integer phase ambiguity.
        """
        return self._s_params['n']
    
    @n.setter
    def n(self, value):
        """
            Sets the number of wavelengths n to resolve the integer phase ambiguity.
        """
        self._s_params['n'] = value
    
    @property
    def mu_r(self):
        """
            Returns the relative permeability mu_r.
        """
        return self.interm_calc_df['mu_r']
        
    
    @mu_r.setter
    def mu_r(self, value):
        """
            Sets the relative permeability mu_r.
        """
        self.s_params['mu_r'] = value
        self.interm_calc_df['mu_r'] = value

    @property
    def eps_r(self):
        """
            Returns the relative permittivity eps_r.
        """
        return self.interm_calc_df['eps_r']
    
    @eps_r.setter
    def eps_r(self, value):
        """
            Sets the relative permittivity eps_r.
        """
        self.s_params['eps_r'] = value
        self.interm_calc_df['eps_r'] = value
    
    
    # ==================================================================================================================
    # 
    # ==================================================================================================================
    def gamma(self, f: float, lambda_c: float, eps_r: float= 1, mu_r: float= 1):
        '''
            Propagation constant gamma for a rectangular waveguide filled with material eps_r and mu_r.

            gamma = 1i*sqrt(((omega^   2)*epsr*mur)/(c_const^2) - ((2*pi)/(lambdac))^2)
            
            Taken form DOI: 10.1109/IMOC.2011.6169318
            Comsol expression: 1i*sqrt(((omega^2)*epsr*mur)/(c_const^2) - ((2*pi)/(lambdac))^2)
            Calulates the propagation constant gamma based on the angular frequency, relative permittivity, 
            relative permeability, and cutoff wavelength.
            For an air-filled waveguide, mur is typically 1 and epsr is close to 1.
        Args:
            f: frequency in Hz
            epsr: relative permittivity
            mur: relative permeability
            lambdac: cutoff wavelength in m
        '''
        _omega = 2 * np.pi * f  # angular frequency
        _gamma = 1j* np.sqrt( (( np.power(_omega, 2) * eps_r * mu_r) / (c_const**2)) - ((2 * np.pi) / lambda_c)**2 )
        return _gamma

    # ==================================================================================================================
    # Estimate the integer phase ambiguity value `n` that minimizes the RMS error between calculated and measured 
    # roup delay.
    # ==================================================================================================================
    def _calculate_errors(self, f, n_range: tuple):
        """
            Estimate the integer phase ambiguity value `n` that minimizes the
            RMS error between calculated and measured group delay.

            Parameters:
            - n_range: tuple (min_n, max_n+1), inclusive range of candidate n

            Updates:
            - self.n: best n value
            - self.mu_r, self.eps_r: arrays at optimal n
        """
        self.logger.debug(f"Estimate the integer phase ambiguity value `n`. Range is {n_range[0]} to {n_range[1]-1}. "
                          "I'll disable the logger for this operation.")
        # diable the logger
        
        # print(S11, S21, tau_meas, f)
     
        errors  = pd.DataFrame(index=f)  # DataFrame to store errors for each n
        tau = pd.DataFrame(index=f)  # DataFrame to store calculated tau for each n
        _tmp_calc  = []#pd.DataFrame(index=f)  # DataFrame to store errors for each n
        
        #- set nrange t0 4
        # n_range =(3,5)
        pbar = tqdm(range(n_range[0], n_range[1]), desc="Estimating tau_g", file=self.tqdm_out)
        for n in pbar:
            self.logger.disabled = True
            # pbar.set_description(f"Estimating for n = {n}...")  # Write to the logger
            # Compute tau_calc using Equation (1.7)
            mu_r = self.permeability(self.Z1, n)
            eps_r = self.permittivity(mu_r)
            # print(f, np.imag(eps_r))
            # create a 3x3 tensor with the imag part on the diagonal
            # eps_r_tensor = np.array([
            #     [np.imag(eps_r[0]), 0, 0],
            #     [0, np.imag(eps_r[0]), 0], 
            #     [0, 0, np.imag(eps_r[0])]])
            # # Create the correct tensor shape (n,3,3):
            # eps_r_tensor = np.array([np.eye(3)*np.real(eps) for eps in eps_r])


            # print(f"    -> eps_r_tensor: {eps_r_tensor} for n={n} and eps_r={eps_r[0]}")
            # eps_re = self.kk_real_from_imag(f, eps_r.real)
            # print(f"    -> eps_re: {eps_re} for n={n} and eps_r={eps_r[0]}")
            # plot the eps_r.real and the eps_re
            # plt.figure(figsize=(10, 6))
            # plt.plot(f, eps_r.real, label='eps_r.real', color='blue')
            # plt.plot(f, eps_re, label='eps_re', color='orange', linestyle='--')
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Relative Permittivity')
            # plt.title(f'Relative Permittivity Comparison for n={n}')
            # plt.legend()
            # plt.grid()
            # plt.show()
            # calculate the error
            # err = eps_r.imag - eps_re
            # print(f"    -> Error: {err} for n={n} and eps_r={eps_r[0]}")
            # _f_scaled = f# / 1e9  # convert frequency to GHz
            # _lam_c = c_const / (self.f_c)  #/ 1e9 cutoff wavelength in m
            # lam0 = c_const / _f_scaled  # free space wavelength in m
            # _L = self.sample_length#*1e9 # convert length to mm

            _freq, _eps_r, _mu_r, tau_calc, interm_result = self.tau_calculated1(f, eps_r, mu_r, self.sample_length, self.lam_c)
            # _lam_g = self.lam_g(_f_scaled, n)
            # lam_g_estimated = self.lam_g_estimated(eps_r, mu_r, lam0, _lam_c)
            
            # v = c_const / np.sqrt(_eps_r * _mu_r)  # speed of light in the material
            # length_calc = self._s_params['tau_mea'].values * (v * np.sqrt(1-(v/ (2*3.0988e-3*self.f))**2) )

            # a = 3.0988e-3
            # # calculate c in material using the formula c = 1/sqrt(mu_r*eps_r)
            # c_mat = c_const / np.sqrt(eps_r * mu_r)  # speed of light in the material
            # print(f"eps_r: {self.eps_r}, mu_r: {self.mu_r}")
            # _length  = tau_calc * (c_mat * np.sqrt(1 - (c_mat/(2*a*_f_scaled)**2)))
            # print(f"tau_calc: {tau_calc}, tau_meas: {tau_meas}, f: {f}, n: {n}, lam_g_estimated: {lam_g_estimated}, _length: {_length} m, c_mat: {c_mat} m/s")

            # __X = self.calc_X(S11, S21)
            # __Gamma = self.reflection_coefficient(__X)
            # __T = self.transmission_coefficient(S11, S21, __Gamma)
            # # self.logger.debug("[4.1]  From equation 1.5 ([1], P20, Eq. 1.5) , calculate alpha = ln(1/T)")
            # __alpha = self.calc_alpha(__T, n)

            
            # _d_beta_df = self.sample_length * np.gradient(self.calc_beta(__alpha, self.sample_length, n), f)

            # eps_re = [self.kkr(f, _epsimg)  for _epsimg, _f in  zip(np.imag(eps_r), f)] # calculate the scalar Kramers-Kronig relation for eps_r
            _sample_length, _lam_c, _product, _d_product_df, _term1, _term2, _sqrt_term, _numerator, _denominator = interm_result
            err = np.abs((tau_calc.real - self.tau_mea.values.real).real)  # calculate the error between calculated and measured group delay
            # err = np.abs(np.longdouble(tau_calc) - np.longdouble(self._s_params['tau_mea'].values))


            # concat the error as a new columen with the namen n
            errors = pd.concat([errors, pd.DataFrame({f'{n}': err}, index=f)], axis=1)
            tau = pd.concat([tau, pd.DataFrame({f'{n}': tau_calc}, index=f)], axis=1)
            _tmp_calc.append((n, pd.DataFrame({
                'freq': _freq, 
                # 'n': n, 
                'eps_r': _eps_r, 'mu_r': _mu_r,
                'tau_calc': tau_calc, 
                'tau_mea': self.tau_mea.values, 
                'err': err, 
                '_sample_length': _sample_length,
                '_lam_c': _lam_c,
                '_product': _product,
                '_d_product_df': _d_product_df, 
                '_term1': _term1, 
                '_term2': _term2, 
                'sqrt_term': _sqrt_term, 
                '_numerator': _numerator, 
                '_denominator': _denominator,
                #
                # 'length_calc': length_calc,
                # '_calc_beta': _d_beta_df
                # 'lam_g_mea': self._s_params['lam_g_mea'].values,
                # 'lam_g': _lam_g,
                # 'lam_g_estimated': lam_g_estimated,
                # 'n': _L/self._s_params['lam_g_mea'].values,
                # '_length': _length,
                }, index=f
            )))
            self.logger.disabled = False


        # #export the errors and tau to a csv file
        writer = pd.ExcelWriter(f'{self._tmp_folder}/tau_intermediate_results.xlsx', engine="auto")
        for _n, _df in _tmp_calc:
            _df.to_excel(writer, sheet_name=f'n_{_n}', index=False)
            self.logger.debug(f"Exported data for n={_n} to Excel.")
        # writer.save()
        writer.close()
        # exit()  

        # find the columen in each row with the minimum value
        self.logger.debug("Calculating the minimum error and corresponding n value for each frequency point.")
        cols_to_check = errors.columns.difference(['f'])
        errors['error'] = errors[cols_to_check].min(axis=1)
        errors['n'] = errors[cols_to_check].idxmin(axis=1)
        errors['n'] = errors['n'].str.replace(' ', '').astype(int)  # remove spaces and convert to int

        # Select a single n, based on a majority vote
        self.logger.debug("Selecting the most common n value across all frequency points.")
        n_counts = errors['n'].value_counts()
        if not n_counts.empty:
            self.logger.debug(f"n counts: {n_counts}")
            errors['n'] = n_counts.idxmax()

        errors.to_excel(f'{self._tmp_folder}/tau_error_calculation.xlsx')
        tau.to_excel(f'{self._tmp_folder}/tau_calculation.xlsx')

        errors_filtered = errors[['error', 'n']]
        
        self.logger.debug(f"{errors_filtered}")
        return errors_filtered
   
    def estimate_n_from_group_delay(self, n_range=(0, 10)):
        # check if more than 1 frequency point is available
        if len(self.s_params['freq']) < 2:
            self.logger.error("Not enough frequency points to estimate n.")
            return 0
        self.logger.debug(f"Estimating n from group delay with frequency range {self.s_params['freq'].min()}Hz to {self.s_params['freq'].max()}Hz")
        err_and_n = self._calculate_errors(self.s_params['freq'].values, n_range)
        
        # append the errors to the s_params DataFrame
       
        self._s_params = self._s_params.merge(
            err_and_n, left_on='freq', right_index=True, how='left'
        )
        self._s_params['n_orig'] = err_and_n['n'].values
        # filter outlieres using the z-score method
        self.logger.debug("Filtering outliers using z-score method.")

# Step 1: Calculate the z-score for column 'n'
        z_scores = zscore(err_and_n['n'], nan_policy='omit')

        # Step 2: Detect outliers (|z| >= 3)
        outlier_mask = np.abs(z_scores) >= 2

        # Step 3: Set outliers to NaN, preserve index
        err_and_n['n_original'] = err_and_n['n']  # Optional: keep original for reference
        err_and_n.loc[outlier_mask, 'n'] = np.nan

        # Step 4: Mark which values are NaN (to track which get interpolated)
        interpolation_mask = err_and_n['n'].isna()

        # Step 5: Interpolate NaN values in 'n'
        err_and_n['n'] = np.round(err_and_n['n'].interpolate(method='linear'), 0)
        # Step 6: Add column 'intp' to track interpolated values
        err_and_n['intp'] = interpolation_mask


        # Step 7: Store back the updated 'n' values
        self._s_params['n'] = err_and_n['n'].values
        self._s_params['intp'] = err_and_n['intp'].values
        
       
        # tanken from https://stackoverflow.com/questions/61143998/numpy-best-fit-line-with-outliers
        # Then you filter the outliers, based on the column mean and standard deviation#
        # df = pd.DataFrame(zip(self.s_params['freq'].values, self.s_params['n'].values), columns=['freq', 'n'])
        # df = df[(np.abs(stats.zscore(df)) < 2.5).all(axis=1)]
        # # print(f"Filtered df: {df}")
        # b, m = np.polyfit(self.s_params['freq'], self.s_params['n'], 1)
        # n = b*self.s_params['freq'] + m
        # # round the n values to the nearest integer
        # n = np.round(n).astype(int)
        # print(f"Estimated n values: {list(n)}")
        # # exit()
        # # save the estimated n-values to the s_params DataFrame
        # self.s_params['n_c'] = n
        # # round the estimated n-values to the nearest integer
        # self.s_params['n_c'] = np.round(self.s_params['n_c']).astype(int)
        
        # exit()
        # self.logger.info(f"Estimated errs values: {errs}")
        return self.s_params['n'].values
    
    # ==================================================================================================================
    # 
    # ==================================================================================================================
    def difference_quotient(self, product, f):
        dfdx = np.empty_like(product)

        # Forward difference for first point
        dfdx[0] = (product[1] - product[0]) / (f[1] - f[0])

        # Central difference for interior points
        for i in range(1, len(f) - 1):
            dfdx[i] = (product[i+1] - product[i-1]) / (f[i+1] - f[i-1])

        # Backward difference for last point
        dfdx[-1] = (product[-1] - product[-2]) / (f[-1] - f[-2])

        return dfdx

    def generalized_difference_quotient(self, y, x, n=1):
        """
        Compute the numerical derivative dy/dx using custom finite differences:
        - Uses central differences with `n` points before and after each x[i] when possible
        - Uses forward/backward differences near boundaries

        Parameters:
        - y: array-like, values of the function (e.g., eps_r * mu_r)
        - x: array-like, the independent variable (e.g., frequency)
        - n: int, number of points before and after to use for central differences

        Returns:
        - dydx: array-like, derivative values at each point
        """
        y = np.asarray(y)
        x = np.asarray(x)
        dydx = np.empty_like(y)

        N = len(x)

        for i in range(N):
            # Define slicing bounds
            i_min = max(0, i - n)
            i_max = min(N, i + n + 1)  # +1 because slicing is exclusive on the right

            # Use points around the current index
            x_slice = x[i_min:i_max]
            y_slice = y[i_min:i_max]

            # Fit linear polynomial to the local slice and take its derivative at x[i]
            # This is equivalent to a local least-squares derivative
            coeffs = np.polyfit(x_slice, y_slice, deg=1)
            dydx[i] = coeffs[0]  # slope = dy/dx

        return dydx

    # ==================================================================================================================
    # 
    # ==================================================================================================================
    def tau_calculated1(self, freq, eps_r, mu_r, sample_length, lam_c):
        """
        Compute calculated group delay τ_cal using full Equation [2] (2.48)
        
        Parameters:
        - f: frequency array [Hz]
        - eps_r: relative permittivity (array or scalar)
        - mu_r: relative permeability (array or scalar)
        - L: sample length [m]

        Returns:
        - tau_cal: calculated group delay [s]
        """
        # Product and derivative
        eps_r = eps_r.values
        mu_r = mu_r.values
        
        product = eps_r * mu_r
        d_product_df = np.gradient(product, freq)  # Initialize to zero, will be calculated later

        term1 = freq * product
        term2 = 0.5 * np.power(freq, 2) * d_product_df
        # print(f"    -> (eps_r*mu_r).real: {product} and d_product_df: {d_product_df}")

        # Square root term in denominator
        sqrt_term = np.sqrt(
                ((product * np.power(freq, 2)) / (np.power(c_const, 2)))
                - 
                (1 / np.power(lam_c, 2))
            )
        # print(f"    -> sqrt_term: {sqrt_term}")

        # Numerator of derivative
        numerator = (term1 + term2)
        denominator = sqrt_term

        # Final τ_cal
        tau_calc = ((sample_length/np.power(c_const, 2)) * (numerator / denominator))
        return freq, eps_r, mu_r, tau_calc, (sample_length, lam_c, product, d_product_df, term1, term2, sqrt_term, numerator, denominator)
    
    # ==================================================================================================================
    # 
    # ==================================================================================================================
    def tau_calculated2(self, f, eps_r, mu_r, L, lam_c):
        """
        Compute calculated group delay τ_cal using full Equation (1.7)
        
        Parameters:
        - f: frequency array [Hz]
        - eps_r: relative permittivity (array or scalar)
        - mu_r: relative permeability (array or scalar)
        - L: sample length [m]
        - lambda_c: cutoff wavelength (set to np.inf if none)

        Returns:
        - tau_cal: calculated group delay [s]
        """
        # Product and derivative
        _res = np.ndarray(shape=f.shape, dtype=complex)
        # for i in range(len(f)):
        #     if i > 1 and i < len(f) - 2:
        # _f_sel_rng = f[i-2:i+2]  # Select the current frequency
        # print(f"    -> Selected frequency range for gradient: {_f_sel_rng} Hz")
        # _f_sel = f[i]  # Current frequency
        _lam = c_const / f 
        print("I work with the following frequencies:")
        print(f"    -> Frequencies: {f} Hz and wavelengths: {_lam} m")

        # print(f"    -> Current frequency: {_f_sel} Hz, lambda: {_lam} m")
        product = np.sqrt((eps_r*mu_r)/np.power(_lam, 2) - 1/np.power(lam_c, 2))
        # print(f"    -> Selected product: {product} (eps_r: {eps_r[i]}, mu_r: {mu_r[i]}, lambda_c: {lam_c})")
        grad = np.gradient(product, f)  # Derivative of product w.r.t frequency
        res = L*grad	  # Use edge_order=2 for better accuracy at the edges	#
            # else:
            #     res = np.nan
            # np.append(_res, res)
            
                
        return f, eps_r, mu_r, res, grad
    
    def lam_g(self, f, n):
        """
        returns the group wavelength lam_g
        """
        _X = self.calc_X(self.S11, self.S21)
        _Gamma = self.calc_Gamma1(_X)
        _T = self.calc_T(self.S11, self.S21, _Gamma)
        # self.logger.debug("[4.1]  From equation 1.5 ([1], P20, Eq. 1.5) , calculate alpha = ln(1/T)")
        _alpha = self.calc_alpha(_T, n)

        return 1/self.calc_beta(_alpha, self.sample_length, n).real
    
    # ==================================================================================================================
    # Standard NRW calculations
    # [1] Nicholson, Ross, and Weir, "A Generalized Method for Extracting the Dielectric Properties of Materials from
    #     Measured Scattering Parameters", IEEE Transactions on Microwave Theory and Techniques, 1976.
    # [2] https://nvlpubs.nist.gov/nistpubs/Legacy/TN/nbstechnicalnote1355r.pdf
    #
    # See [1], Page 34, Equation 5 - 7
    # See [2], Page 10, Equation 2.26 - 2.30
    # Variable notation has been taken from [2] and is used in the code.
    # ==================================================================================================================
    def calc_X(self, S11, S21):
        """
        Calculate *X* for the reflection coeficient *Gamma1*, which is given explicitly in terms of the scattering 
        parameters *S11* and *S21* where
        ```
        X = (1-V1*V2)(V1-V2) =(S11^2 - S21^2 + 1) / (2*S11)
        ```
        and 
        ```
        V1 = S21 + S11, V2 * S21 - S11
        ```
        [1], Page 34, Equation 6
        [2], Page 10, Equation 2.27
        
        Args:
            S11: S-parameter S11, complex numpy array
            S21: S-parameter S21, complex numpy array
        
        Returns:
            Complex numpy array representing *X*, the intermediate variable for calculating the reflection coefficient 
            *Gamma1*.
        """ 
        self.logger.info(f"Calculating X = (S11^2 - S21^2 + 1) / (2*S11)")
        self.logger.debug(f"I'll work with \nS11: {self.str_array_repr(S11)} and \nS21:{self.str_array_repr(S21)}")
        _X = (np.power(S11,2) - np.power(S21,2) + 1) / (2*S11) 
        self.logger.info(f"X = {self.str_array_repr(_X)}")
        self.logger.debug(f"X = {self.str_array_repr(_X, polar=False)}")
        return _X

    def calc_Gamma1(self, X):
        """
        Calculate the reflection coefficient *Gamma1* for the NRW procedure from *X*
        ```
            Γ = Gamma1 = X + sqrt(X^2 - 1)
        ```
        with
        ```
            X = (S11^2 - S21^2 + 1) / (2*S11)
        ```
        The definition of *X* is defined in method
        ```
        def calc_X(self, S11, S21)
        ```
        Args:
            X: complex numpy array, the intermediate variable calculated from S11 and S21.
        
        Returns:
            Complex numpy array representing the reflection coefficient Gamma1.

        [1], Page 34, Equation 5
        [2], Page 10, Equation 2.26
        """
        self.logger.debug(f"Calculating reflection coefficient {STR_Gamma} = X + sqrt(X^2 - 1)")
        _Gamma_pos = X + np.sqrt(np.power(X,2) - 1)
        _Gamma_neg = X - np.sqrt(np.power(X,2) - 1)
       
        # Apply the condition |gamma| < 1 element-wise for numpy arrays
        _Gamma = np.where(np.abs(_Gamma_pos) < 1, _Gamma_pos, _Gamma_neg)
        self.logger.info(f"{STR_Gamma} = {self.str_array_repr(_Gamma)}")
        self.logger.debug(f"{STR_Gamma} = {self.str_array_repr(_Gamma, polar=False)}")
        return _Gamma

    def calc_Z1(self, S11, S21, Gamma):
        """
        Calculate the transmission coefficient *Z1* for the Nicolson-Ross-Weir procedure from the scattering 
        parameters *S11* and *S21* where
        ```
            Z1 = (S11 + S21 - Gamma1) / (1 - (S11  + S21) * Gamma1)
        ```
        with *Gamma1* 
        ```
            Gamma1 = X + sqrt(X^2 - 1)
        ```
        The definition of *Gamma1* is defined in method
        ```
        def calc_Gamma1(self, X)
        ```

        Args:
            S11: S-parameter S11, complex numpy array
            S21: S-parameter S21, complex numpy array
            Gamma: reflection coefficient Gamma1, complex numpy array
        Returns:
            Complex numpy array representing the transmission coefficient *Z1*.

        For more details, see the references:
        [1], Page 34, Equation 7
        [2], Page 10, Equation 2.30
        """
        self.logger.debug(f"Calculating transmission coefficient "
                          f"Z1 = (S11 + S21 - {STR_Gamma}) / (1 - (S11  + S21) * {STR_Gamma})")
        _Z1 = (S11 + S21 - Gamma) / (1 - (S11 + S21) * Gamma)
        self.logger.info(f"Z1 = {self.str_array_repr(_Z1)}")
        self.logger.debug(f"Z1 = {self.str_array_repr(_Z1, polar=False)}")
        return _Z1

    # ===================================================================================================================

    def calc_tau_mea(self, freq, Z1):
        """
            Calculate the measured group delay tau_mea from the angle of the transmission coefficient Z1.
            ```
            tau_mea = (1/2*pi) * d(phi)/d(freq)
            ```
            with
            ```
            phi = np.unwrap(np.angle(Z1))
            ```
        """
        # # Check if more than 1 frequency point is available
        # if len(freq) < 2:
        #     self.logger.error("Not enough frequency points to calculate group delay.")
        #     return np.zeros((len(freq), 1))
        
        # self.logger.debug(f"Calculating measured group delay tau_mea = -d(unwrap(angle(Z1)))/d(2*pi*freq)")
        # _omega = 2*np.pi*freq
        # _tau_mea = -np.gradient(np.unwrap(np.angle(Z1)), _omega, axis=0).reshape(-1, 1) 
        _n = np.zeros_like(self.freq, dtype=int)  # initialize n to zeros
        _alpha = self.calc_alpha(self.Z1, _n)  # calculate alpha from Z1 and n
        _tau_mea  = np.gradient(
            self.sample_length*self.calc_beta(_alpha, self.sample_length),
            self.freq, axis=0).reshape(-1, 1) # group delay in seconds
        
        return _tau_mea

    # ==================================================================================================================
    # Simplification of terms, appearing multiple times in the calculations
    # ==================================================================================================================
    def calc_lam_og(self, lam_0, lam_c):
        """
        Calculate the free space wavelength at the given frequency.
        lam_og = 1 / (np.sqrt((1/np.power(lam_0,2)) - 1/np.power((lam_c), 2)))
        """
        
        self.logger.debug(f"Calculating {STR_lambda}_og = 1 / (sqrt((1/({STR_lambda}0^2)) - 1/{STR_lambda}c^2))")
        lam_og = 1 / (np.sqrt( (1/np.power(lam_0,2)) - (1/np.power(lam_c, 2))) )
        self.logger.info(f"{STR_lambda}_og({STR_lambda}0 = {self.str_array_repr(np.round(lam_0,2))}, "
                          f"{STR_lambda}c = {np.round(lam_c,2):.2e}) "
                          f"= {self.str_array_repr(lam_og)}")
        self.logger.debug(f"{STR_lambda}_og({STR_lambda}0 = {self.str_array_repr(np.round(lam_0,2))}, "
                          f"{STR_lambda}c = {np.round(lam_c,2):.2e}) "#
                          f"= {self.str_array_repr(lam_og, polar=False)}")
        return lam_og
    
    def calc_z(self, Gamma):
        """
        Calculate the normalized impedance *z*, defined as the ratio of the reflection coefficient Gamma of 
        the waveguide
        ```
        z = (1 + Gamma) / (1 - Gamma)
        ```
        """
        # Placeholder for Gamma, this should be calculated or provided

        self.logger.debug(f"Calculating z = (1+{STR_Gamma})/(1-{STR_Gamma})")
        delta = (1 + Gamma) / (1 - Gamma)
        self.logger.info(f"z = (1+{STR_Gamma})/(1-{STR_Gamma}) = {self.str_array_repr(delta)}")
        self.logger.debug(f"z = (1+{STR_Gamma})/(1-{STR_Gamma}) = {self.str_array_repr(delta, polar=False)}")
        return delta
    
    # ==================================================================================================================
    # Simplifaction of some terms, for calulation 1/Lambda
    # ==================================================================================================================
    def calc_alpha(self, Z1: np.ndarray, n: int) -> np.ndarray:
        """
            Calculate the intermediate value *alpha*
            ```
            alpha = ln(1/Z1)
            ```
            the natural logarithm of (1/Z1). 
            Equation has an infinite number of roots since the imaginary part of the term ln(1/Z1) is equal to (j*2*pi*n), 
            where n = 0, +-1, +-2, ..., the integer of (L/lam_g) where lam_g is the group delay wavelength.
            Implementation:
            ```
            np.log(np.abs(1/Z1)) + 1j * np.unwrap(np.angle(1/Z1, False)) + 2*np.pi*n
            ```

            args:
                Z1: transmission coefficient, defined as `(S11 + S21 - Gamma1) / (1 - (S11  + S21) * Gamma1)`
                n: number of root, to resolve the phase ambiguity, integer value. See [1] and [2] for more details.
            
            Returns:
                Complex numpy array representing the intermediate value *alpha*.
                
            [1], Page 34, Equation 1.5
            [2], Page 16, Equation 2.45
        """
        self.logger.debug(f"Calculating {STR_alpha} = ln(1/Z1) for n = {self.str_array_repr(n)}")
        ln_inv_Z1_mag = np.log(np.abs(1/Z1))
        ln_inv_Z1_imag = np.unwrap(np.angle(1/Z1, False)) + 2*np.pi*n  # This is the argument of 1/T
        ln_1_Z1 = ln_inv_Z1_mag + 1j * ln_inv_Z1_imag
        self.logger.info(f"{STR_alpha}(n={self.str_array_repr(n)}) = ln(1/T) = ln({self.str_array_repr(np.abs(1/Z1))}) + j({self.str_array_repr(np.angle(1/Z1, False))}+2*pi*{n}) ={self.str_array_repr(ln_1_Z1)}")
        self.logger.debug(f"{STR_alpha}(n={self.str_array_repr(n)}) = ln(1/T) = ln({self.str_array_repr(np.abs(1/Z1))}) + j({self.str_array_repr(np.angle(1/Z1, False))}+2*pi*{n}) = {self.str_array_repr(ln_1_Z1, polar=False)}")
        return ln_1_Z1

    def calc_beta(self, alpha, sample_length) -> np.ndarray:
        """
        Calculate the intermediate value *1/Lamda = beta*, defined as
        ```
        beta = 1/(Lamda) = sqrt( -( 1/(2*np.pi*L) * ln(1/Z1))^2 ) = sqrt( -( 1/(2*np.pi*L) *alpha)^2 )
        ```
        where *alpha* is the intermediate value calculated from the transmission coefficient *Z1* and the integer phase 
        ambiguity value *n*.
        ```
        alpha = ln(1/Z1) = abs(ln(1/Z1)) + j * (arg(1/Z1) + 2 * pi * n)
        ```
        and is defined in the method
        ```
        def calc_alpha(self, Z1, n)
        ```
        The sign of the square root is chosen such that the real part of *beta* is positive, i.e., *Re(beta) > 0*.
        
        Args:
            alpha: complex value of `alpha = ln(1/Z1)`. For resolving the phase ambiguity see [1] and [2]
            sample_length: length of the sample in m

        Returns:
            beta: Complex numpy array representing the intermediate value *1/(Lamda)*.

        References:
            [1], Page 34, Equation 10 and 11
            [2], Page 16, Equation 2.48 and 2.49
        """
        _str_beta = ud.lookup("GREEK SMALL LETTER BETA")
        _str_Lambda = ud.lookup("GREEK CAPITAL LETTER LAMDA")
        _str_alpha = ud.lookup("GREEK SMALL LETTER ALPHA")
        _str_pi = ud.lookup("GREEK SMALL LETTER PI")
        # Placeholder for eps_r and mu_r, these should be calculated or provided
        beta2 = -np.pow( (1/(2*np.pi*sample_length)) * alpha, 2)
        self.logger.debug(f"Calculating {_str_beta}(L = {sample_length:.2e})^2 = 1/({_str_Lambda}^2) = -(1/(2{_str_pi}L) * {_str_alpha})^2")
        self.logger.info(f"{_str_beta}(L = {sample_length:.2e})^2 = -(1/(2{_str_pi}L) * {_str_alpha})^2) = {self.str_array_repr(beta2)}")
        self.logger.debug(f"{_str_beta}(L = {sample_length:.2e})^2 = -(1/(2{_str_pi}L) * {_str_alpha})^2) = {self.str_array_repr(beta2, polar=False)}")
        beta_neg =  -np.sqrt(beta2)
        beta_pos = np.sqrt(beta2)
        # Real(beta) > 0
        beta = np.where(np.real(beta_pos) >= 0, beta_pos, beta_neg)
        self.logger.info(f"{_str_beta}(L = {sample_length:.2e}) = sqrt({_str_beta}^2) = {self.str_array_repr(beta)}")
        self.logger.debug(f"{_str_beta}(L = {sample_length:.2e}) = sqrt({_str_beta}^2) = {self.str_array_repr(beta, polar=False)}")
        return beta

    # ==================================================================================================================


    # ==================================================================================================================
    # 
    # ==================================================================================================================
    def permeability(self, *arg, **kwargs) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this method to calculate relative permeability mu_r.")
    
    def permittivity(self, *arg, **kwargs) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this method to calculate relative permittivity eps_r.")
    
    # ==================================================================================================================
    # 
    # ==================================================================================================================
    def lam_g_estimated(self, eps_r, mu_r, lam_0, lam_c):
        """
        Estimate the value of n based on an initial guess of eps_r and mu_r
        
        Arguments:
        eps_r : estimated relative permittivity
        mu_r : estimated relative permeability
        lam0 : free space wavelength
        lam_c : cutoff wavelength

        """
        # Calculate the propagation constant gamma
        gamma = 1j*((2*np.pi)/lam_0) * np.sqrt(eps_r * mu_r - np.power((lam_0/lam_c), 2))
        self.logger.info(f">>> gamma: {self.rect(gamma)} ({self.polar(gamma)})")
        inv_Gamma = 1j*(gamma/(2*np.pi))
        self.logger.info(f">>> inv_Gamma: {self.rect(inv_Gamma)} ({self.polar(inv_Gamma)})")
        inv_lam_g = (1/inv_Gamma).real
        self.logger.info(f">>> Estimated inverse group delay: {inv_lam_g} m")
        #lam_g = 1 / inv_lam_g
        #self.logger.info(f">>> Estimated group delay: {lam_g} m")
        return inv_lam_g



    # ==================================================================================================================
    # 
    # ==================================================================================================================
    def convert1(self, permittivity) -> tuple:
        """
        Convert the complex permittivity to the real part and the loss tangent tanDelta.
        Args:
            permittivity: complex permittivity (numpy array or scalar)
        Returns:
            eps_r: real part of the permittivity (numpy array)
            tanDelta: loss tangent (numpy array)
        """
        if isinstance(permittivity, pd.Series):
            permittivity = permittivity.values

        eps_r = np.abs(permittivity).astype(float)
        self.interm_calc_df['abs(eps_r)'] = eps_r
        # Filter real part values to be non-negative and greater than 1
        #eps_r = np.clip(eps_r, 1, None)  # Ensure eps_r >= 1
        tanDelta = (permittivity.imag / permittivity.real).astype(float)
        self.interm_calc_df['tanDelta'] = tanDelta
        # Filter loss tangent values to be non-negative and less than 1
        #tanDelta = np.clip(tanDelta, 0, 1)

        self.logger.info(f"Converted complex permittivity to real part: {self.str_array_repr(eps_r)} and loss tangent: {self.str_array_repr(tanDelta)}")
        return eps_r, tanDelta

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def calculate_trendline(self, x, y, degree=10):
        """
        Calculate a polynomial trendline of specified degree for the given x and y data.
        
        Args:
            x: x data points (numpy array)
            y: y data points (numpy array)
            degree: degree of the polynomial fit (default is 1 for linear fit)
        
        Returns:
            p: coefficients of the polynomial fit
        """
        p = np.polyfit(x, y, degree)
        trendpoly = np.poly1d(p)
        self.logger.info(f"Calculated trendline coefficients: {p}")
        return x, trendpoly(x)
    
    def plot(self):
        # Create the figure and subplots
        fig, axs = plt.subplots(4, 1, figsize=(10, 12))
        fig.subplots_adjust(left=0.1, bottom=0.15, hspace=0.4)
        self.fig = fig

        fig.suptitle(
            self.__class__.__name__ + f" - L={self.sample_length:0.3e} m, f_c={self.f_c/1e9:.2f} GHz",
            fontsize=16
        )

        self.plot_s_params_db(ax=axs[0])  # Plot S11 and S21 in dB

        # Plot 2: n vs frequency
        ax2 = axs[1]
        ax2b = ax2.twinx()
        try:
            ax2.plot(self.freq, self.n, label='Number of Wavelengths (n) (Interpolated)', linewidth=2, color='green')
            # ax2.plot(self.freq, self.n, label='Number of Wavelengths (n) (Calculated)', linestyle='--', color='orange')
            # ax2.plot(self.f, self.Gamma, label='Reflection coefficient (Gamma)')
            # ax2.plot(self.f, self.T, label='Transmission_coefficient (T)')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Value')
            ax2.legend()
        except Exception as e:
            self.logger.error(f"Error plotting number of Wavelengths (n): {e}")
            ax2.text(0.5, 0.5, f'Error plotting number of Wavelengths (n): {e}',
                    fontsize=12, ha='center', transform=ax2.transAxes)
        # self._plot_interpolation_boxes(ax2)  # Add interpolation boxes to the eps_r plot

        # Plot 3: eps_r and tanDelta
        self.plot_dielectric_losses(ax=axs[2], trendline=False)  # Use the new method to plot eps_r and tanDelta
        self.plot_magnetic_losses(ax=axs[3], trendline=False)  # Use the new method to plot eps_r and tanDelta
        # self._plot_interpolation_boxes(ax3)  # Add interpolation boxes to the eps_r plot

        # tight layout to adjust spacing between subplots
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title
        

        

        # Add a slider below the plots
        #ax_slider = fig.add_axes([0.2, 0.05, 0.65, 0.03])
        #self.slider = Slider(
        #     ax=ax_slider,
        #     label="n (phase ambiguity)",
        #     valmin=-100,
        #     valmax=100,
        #     valinit=self.n[0],
        #     valstep=1
        # )


        # def update(val):
        #     self.n = int(val)
        #     self.logger.info(f"Slider updated to n = {self.n}")

        #     # Recalculate updated data
        #     epsr_new = self.permittivity(self.S11, self.S21, self.n)
        #     mur_new = self.permeability(self.S11, self.S21, self.n).real

        #     # Update plot data
        #     self.p_epsr.set_ydata(epsr_new)
        #     self.p_mu.set_ydata(mur_new)

        #     # Rescale axes for eps_r and mu_r
        #     ax3.relim()
        #     ax3.autoscale()
        #     ax3b.relim()
        #     ax3b.autoscale()
        #     ax4.relim()
        #     ax4.autoscale()

        #     self.fig.canvas.draw_idle()


        # self.slider.on_changed(update)
    
    def plot_s_params_db(self, ax=None):
         # Plot 1: S11 and S12 in dB
        ax.title.set_text(f'S-Parameters in dB: {self.measurement_data.name}')
        ax.plot(self.freq, self.S11dB, label='S11 (dB)', color='blue')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('S11 (dB)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        axS21 = ax.twinx()
        axS21.plot(self.freq, self.S21dB, label='S21 (dB)', color='red')
        axS21.set_ylabel('S21 (dB)', color='red')
        axS21.tick_params(axis='y', labelcolor='red')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = axS21.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    def plot_lossTan_dissapFactor(self, ax=None, trendline=True):
        # Plot 3: eps_r and tanDelta
        ax.title.set_text('Realitve Permittivtiy, Loss tangent: $\epsilon_r = \epsilon_r^{\prime}(1-j \mathrm{tan}\delta)$')
        self.p_epsr, = ax.plot(self.freq, self.eps_r.values.real, label='$\epsilon_r^{\prime}$', color='blue')
        
        if trendline:
            _f, epsr_trend = self.calculate_trendline(self.freq, self.abs_epsr, degree=1)
            ax.plot(_f, epsr_trend, label='Trend $|\epsilon_r|$', color='green', linestyle='--')
        # set yaxis to 1
        # ax3.set_ylim(1, self.eps_r.real.max() * 1.1)  # Ensure y-axis starts at 1   
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('(Real part) Relative Permittivity ($\epsilon_r$)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        _axb = ax.twinx()
        self.p_tan, = _axb.plot(self.freq, self.tanDelta, label='Loss Tangent $tan\delta$', color='red')
        if trendline:
            _axb.set_ylabel('Loss Tangent ($tan\delta$)', color='red')
            _axb.tick_params(axis='y', labelcolor='red')

        lines3, labels3 = ax.get_legend_handles_labels()
        lines4, labels4 = _axb.get_legend_handles_labels()
        ax.legend(lines3 + lines4, labels3 + labels4, loc='upper right')

    def plot_dielectric_losses(self, ax=None, trendline=True):
        # Plot 3: eps_r and tanDelta
        ax.title.set_text('Dielectric Losses: $\epsilon_r = \epsilon_r^{\prime} - j  \epsilon_r^{\prime\prime}$')
        self.p_epsr, = ax.plot(self.freq, self.eps_r.values.real, label='(Real part) $\epsilon_r^{\prime}$', color='blue')
        
        if trendline:
            _f, eps_real_trend = self.calculate_trendline(self.freq, self.eps_r.values.real, degree=1)
            ax.plot(_f, eps_real_trend, label='Trend $\epsilon_r^{\prime}$', color='green', linestyle='--')
        # set yaxis to 1
        # ax3.set_ylim(1, self.eps_r.real.max() * 1.1)  # Ensure y-axis starts at 1   
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('$\epsilon_r^{\prime}$ (Real part)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        _axb = ax.twinx()
        self.p_tan, = _axb.plot(self.freq, self.eps_r.values.imag, label='(Imaginary part) $\epsilon_r^{\prime\prime}$', color='red')
        
        if trendline:
            _f, eps_imag_trend = self.calculate_trendline(self.freq, self.eps_r.values.imag, degree=1)
            _axb.plot(_f, eps_imag_trend, label='Trend $\epsilon_r^{\prime\prime}$', color='orange', linestyle='--')

        _axb.set_ylabel('$\epsilon_r^{\prime\prime}$ (Imaginary part)', color='red')
        _axb.tick_params(axis='y', labelcolor='red')

        lines3, labels3 = ax.get_legend_handles_labels()
        lines4, labels4 = _axb.get_legend_handles_labels()
        ax.legend(lines3 + lines4, labels3 + labels4, loc='upper right')

        # self._plot_interpolation_boxes(ax3)  # Add interpolation boxes to the eps_r plot

    def plot_magnetic_losses(self, ax=None, trendline=True):
        # Plot 4: mu_r (real part)
        ax.title.set_text('Magnetic losses: $\mu_r = \mu_r^{\prime} - j  \mu_r^{\prime\prime}$')
        self.p_mu, = ax.plot(self.freq, self.mu_r.values.real, label='Real Permeability ($\mu_r^{\prime}$)', color='blue')
       
        if trendline:
            _f, mu_real_trend = self.calculate_trendline(self.freq, self.mu_r.values.real, degree=1)
            ax.plot(_f, mu_real_trend, label='Trend $\epsilon_r^{\prime}$', color='green', linestyle='--')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Real Permeability ($\mu_r^{\prime}$)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.legend(loc='upper right')

        _axb = ax.twinx()
        self.p_mu_imag, = _axb.plot(self.freq, self.mu_r.values.imag, label='Imaginary Permeability ($\mu_r^{\prime\prime}$)', color='red')
        
        if trendline:
            _f, mu_imag_trend = self.calculate_trendline(self.freq, self.mu_r.values.imag, degree=1)
            _axb.plot(_f, mu_imag_trend, label='Trend $\mu_r^{\prime}$', color='orange', linestyle='--')

        _axb.set_ylabel('Imaginary Permeability ($\mu_r^{\prime\prime}$)', color='red')
        _axb.tick_params(axis='y', labelcolor='red')
        lines5, labels5 = ax.get_legend_handles_labels()   	
        lines6, labels6 = _axb.get_legend_handles_labels()
        ax.legend(lines5 + lines6, labels5 + labels6, loc='upper right')



    def _plot_interpolation_boxes(self, ax):
                # add a red box at every place, where self._s_params['intp'] is true
        intp = self._s_params['intp'].values
        freqs = self.freq  # frequency array (same length as intp)

        in_segment = False
        start_freq = None
        label_added = False

        for i in range(len(intp)):
            if intp[i] and not in_segment:
                # Start of an interpolated segment
                start_freq = freqs[i]
                in_segment = True
            elif not intp[i] and in_segment:
                # End of an interpolated segment
                end_freq = freqs[i]
                if not label_added:
                    ax.axvspan(start_freq, end_freq, color='red', alpha=0.3, label='Interpolated region')
                    label_added = True
                else:
                    ax.axvspan(start_freq, end_freq, color='red', alpha=0.3)
                in_segment = False

        # Handle case if interpolation segment goes to the end
        if in_segment:
            end_freq = freqs[-1]
            if not label_added:
                ax.axvspan(start_freq, end_freq, color='red', alpha=0.3, label='Interpolated region')
            else:
                ax.axvspan(start_freq, end_freq, color='red', alpha=0.3)

    # ==================================================================================================================
    # 
    # ==================================================================================================================
    def kkr(self, de, eps_imag, cshift=1e-6):
        """Calculate the Kramers-Kronig transformation on imaginary part of dielectric

        Doesn't correct for any artefacts resulting from finite window function.

        Args:
            de (float): Energy grid size at which the imaginary dielectric constant
                is given. The grid is expected to be regularly spaced.
            eps_imag (np.array): A numpy array with dimensions (n, 3, 3), containing
                the imaginary part of the dielectric tensor.
            cshift (float, optional): The implemented method includes a small
                complex shift. A larger value causes a slight smoothing of the
                dielectric function.

        Returns:
            A numpy array with dimensions (n, 3, 3) containing the real part of the
            dielectric function.
        """
        eps_imag = np.array(eps_imag)
        nedos = eps_imag.shape[0]
        cshift = complex(0, cshift)
        w_i = np.arange(0, nedos*de, de, dtype=np.complex128)
        w_i = np.reshape(w_i, (nedos, 1, 1))

        def integration_element(w_r):
            factor = w_i / (w_i**2 - w_r**2 + cshift)
            total = np.sum(eps_imag * factor, axis=0)
            return total * (2/np.pi) * de + np.diag([1, 1, 1])

        return np.real([integration_element(w_r) for w_r in w_i[:, 0, 0]])

    def kkr_scipy(self, frequencies, eps_imag, eps_inf=1.0):
        omega = 2 * np.pi * frequencies
        eps_real = np.zeros_like(eps_imag)

        # Interpolation function for epsilon imaginary
        eps_imag_interp = lambda wp: np.interp(wp, omega, eps_imag)

        omega_min, omega_max = omega[0], omega[-1]

        for i, w in enumerate(omega):

            # Slightly shift integration boundaries if w coincides exactly
            delta = 1e-6 * (omega_max - omega_min)
            a, b = omega_min, omega_max
            if np.isclose(w, omega_min):
                a += delta
            if np.isclose(w, omega_max):
                b -= delta

            def integrand(wp):
                return wp * eps_imag_interp(wp)

            # Perform Cauchy principal value integral avoiding exact boundary match
            integral, _ = quad(
                integrand,
                a, b,
                weight='cauchy',
                wvar=w,
                limit=500
            )

            eps_real[i] = eps_inf + (2 / np.pi) * integral

        return eps_real

    def kkr_tensor_corrected(self, frequencies, eps_imag_tensor, eps_inf=1.0, cshift=1e-6):
        """
        Corrected KK function for tensorial permittivity using actual frequencies.

        Parameters:
            frequencies (np.array): actual frequency vector in Hz, shape (n,)
            eps_imag_tensor (np.array): imaginary permittivity tensor, shape (n,3,3)
            eps_inf (float): permittivity at infinite frequency (default=1)
            cshift (float): small shift for numerical stability (default=1e-6)

        Returns:
            np.array: real permittivity tensor (n,3,3)
        """
        omega = 2 * np.pi * frequencies
        n_freq = len(omega)
        eps_real_tensor = np.zeros_like(eps_imag_tensor)

        # Loop over tensor elements independently
        for a in range(3):
            for b in range(3):
                eps_imag_ab = eps_imag_tensor[:, a, b]

                # Interpolate imaginary part for smooth integration
                interp_eps_imag = lambda wp: np.interp(wp, omega, eps_imag_ab)

                # Integrate numerically for each frequency
                for i, w in enumerate(omega):
                    def integrand(wp):
                        return wp * interp_eps_imag(wp) / (wp**2 - w**2 + 1j*cshift)

                    integral_real = np.trapz(np.real(integrand(omega)), omega)
                    eps_real_tensor[i, a, b] = eps_inf + (2/np.pi) * integral_real

        return eps_real_tensor

    def subtractive_kk_imag(self, frequencies, eps_real):
        """
        Compute imaginary part of permittivity using subtractive KK relation.

        Parameters:
            frequencies: np.array of frequencies (Hz)
            eps_real: np.array of real permittivity values at those frequencies

        Returns:
            eps_imag: np.array of imaginary permittivity (same shape)
        """
        omega = 2 * np.pi * frequencies
        eps_imag = np.zeros_like(eps_real)
        
        for i, wi in enumerate(omega):
            eps_i = eps_real[i]

            integral = 0.0
            for j in range(len(omega) - 1):
                # Integration interval
                wj = omega[j]
                wj1 = omega[j+1]
                dw = wj1 - wj
                # Midpoint value
                wm = 0.5 * (wj + wj1)

                # Linear interpolate real epsilon over this interval
                epsj = eps_real[j]
                epsj1 = eps_real[j+1]

                def eps_real_interp(w):
                    # linear interpolation
                    return epsj + (epsj1 - epsj) * (w - wj) / (wj1 - wj)

                # Trapezoidal integration at midpoint
                num_points = 5  # More = better accuracy
                ws = np.linspace(wj, wj1, num_points)
                integrand = (eps_real_interp(ws) - eps_i) / (ws**2 - wi**2+ 1e-6)
                integral += np.trapz(integrand, ws)

            eps_imag[i] = -2 * wi / np.pi * integral

        return eps_imag

    def kk_real_from_imag(self, frequencies, eps_imag, eps_inf=1.0):
        """
        Reconstruct real part of permittivity using KK relation from imaginary part.

        Parameters:
            frequencies : ndarray
                Frequency values (Hz)
            eps_imag : ndarray
                Imaginary part of permittivity at the corresponding frequencies
            eps_inf : float
                Estimated permittivity at infinite frequency (default=1.0)

        Returns:
            eps_real : ndarray
                Reconstructed real part of permittivity
        """
        omega = 2 * np.pi * frequencies
        eps_real = np.zeros_like(eps_imag)

        for i, wi in enumerate(omega):
            integral = 0.0
            for j in range(len(omega) - 1):
                wj, wj1 = omega[j], omega[j+1]
                dw = wj1 - wj

                # Midpoint rule for each interval
                num_points = 5
                ws = np.linspace(wj, wj1, num_points)
                dws = ws[1] - ws[0]

                epsj = eps_imag[j]
                epsj1 = eps_imag[j+1]
                def eps_interp(w):
                    return epsj + (epsj1 - epsj) * (w - wj) / (wj1 - wj)

                for wk in ws:
                    if np.isclose(wk, wi):
                        continue  # skip singularity
                    eps_wk = eps_interp(wk)
                    integral += wk * eps_wk / (wk**2 - wi**2) * dws

            eps_real[i] = eps_inf + (2 / np.pi) * integral

        return eps_real

    # ==================================================================================================================
    # 2-Port Solution Where position is determined solely by airline L_air and L
    # ==================================================================================================================
    def calc_x(self, f, S11, S21, S12, S22, L_air, L):
        """
            Calculate x = (S21*S12 - S11*S22)*exp(2*gamma0*(L_air-L))

            Parameters:
                S11, S21, S12, S22: Scattering parameters (numpy arrays)
                L_air: Length of the air gap (m)
                L: Length of the sample (m)
            
            Returns:
                x: Calculated complex value
        """
        _gamma0 = self.gamma(f, self.lam_c, 1, 1)
        _x = (S21 * S12 - S11 * S22) * np.exp(2 * _gamma0 * (L_air - L))
        return _x

    def calc_y(self, f, S21, S12, L_air, L):
        """
            Calculate y = (S21 + S12)*exp(2*gamma0*(L_air-L))

            Parameters:
                S11, S21, S12, S22: Scattering parameters (numpy arrays)
                L_air: Length of the air gap (m)
                L: Length of the sample (m)
            
            Returns:
                x: Calculated complex value
        """
        _gamma0 = self.gamma(f, self.lam_c, 1, 1)
        _y = (S21 + S12) * np.exp(2 * _gamma0 * (L_air - L))
        return _y
    
    def calc_Z(self, f, S11, S21, S12, S22, L_air, L):
        """

            Calculates the transmission coefficient Z based on the scattering parameters S11, S21, S12, and S22.

            Calculate Z = (x + 1) / 2*y +- np.sqrt( ((x + 1)**2)/2*y - 1 )

            Parameters:
                S11, S21, S12, S22: Scattering parameters (numpy arrays)
                L_air: Length of the air gap (m)
                L: Length of the sample (m)

            Returns:
                Z: Calculated complex value
        """
        _x = self.calc_x(f, S11, S21, S12, S22, L_air, L)
        _y = self.calc_y(f, S21, S12, L_air, L)

        _Z_pos = (_x+1)/(2*_y) + np.sqrt( ((_x+1)**2)/(2*_y) - 1 )
        _Z_neg = (_x+1)/(2*_y) - np.sqrt( ((_x+1)**2)/(2*_y) - 1 )
        # Select the correct root based on the condition |Z| < 1
        _Z = np.where(np.abs(_Z_pos) < 1, _Z_pos, _Z_neg)
        return _Z

    def reflection_coefficient2(self, f: np.ndarray, S11: np.ndarray, Z: np.ndarray, x: np.ndarray, L: float):
        """
            The reflection coefficient 
            ```
            Gamma2 = +- sqrt((x-Z^2)/(x*Z^2 - 1))
            ```
            from 
            ```
            x = (S21 * S12 - S11 * S22) * exp(2 * gamma0*(L_air-L))
            ```
            and  
            ```
            Z = sqrt( (x-Z^2)
            ``` to resolve the ambiguity of the sign of the square root for finding the physical 
            root of the transmission coefficient determined by Z, defined by 
            ```
                def calc_Z(self, f, S11, S21, S12, S22, L_air, L):
            ```
            The ambiguity in the plus-or-minus sign in eq (2.54) can be resolved by considering the reflection 
            coefficient Gamma3 calculated from S11 alone:
            ```
                def reflection_coefficient3(self, f, S11, Z, L)
            ```
            Reference: [2], Page 18, Equation 2.54

            Args:
                f: frequency array [Hz]
                S11: S-parameter S11
                Z: characteristic impedance [Ohms]	
                    see
                    ```
                        def calc_Z(self, f, S11, S21, S12, S22, L_air, L)
                    ```
                    for implementation of Z calculation
                L: length of the sample [m]
                x: calculated reactance from S-parameters
        """
        
        _Gamma3 = self.reflection_coefficient3(f, S11, Z, L)

        _Gamma2_plus = np.sqrt( (x-np.power(Z, 2))/(x*np.power(Z, 2) - 1) )
        _Gamma2_neg = -np.sqrt( (x+np.power(Z, 2))/(x*np.power(Z, 2) + 1) )
        # The ambiguity in the plus-or-minus sign  can be resolved by considering the reflection coefficient Gamma3
        # calculated from S11 alone 
        _sign_Gamma3 = np.sign(_Gamma3)
        # Check that the sign of the reflection coefficient Gamma3 is the same as the sign of the square root
        _Gamma2 = np.where(_sign_Gamma3 == np.sign(_Gamma2_plus), _Gamma2_plus, _Gamma2_neg)

        return _Gamma2
    
    def reflection_coefficient3(self, f, S11, Z, L):
        """
        Calculate the reflection coefficient Γ3 from S11, Z and L.
        """
        _gamma0 = self.gamma(f, self.lam_c, 1, 1)  # Propagation constant
        _alpha = np.exp(-2*_gamma0*L)
        term1 = _alpha*(np.power(Z, 2) - 1)

        sqrt_term_1 = np.power(_alpha, 2) * np.power(Z, 4)
        sqrt_term_2 = 2 *  np.power(Z, 2) * (2*S11 - np.power(_alpha, 2)) 
        sqrt_term = np.sqrt(sqrt_term_1 + sqrt_term_2 + np.power(_alpha, 2))
        
        nominator_plus = term1 + np.sqrt(sqrt_term)
        nominator_neg = term1 - np.sqrt(sqrt_term)
        denominator = 2*S11*(np.power(Z, 2))

        _Gamma3_plus = nominator_plus / denominator
        _Gamma3_neg = nominator_neg / denominator

        # The correct root is the one that satisfies the condition |Gamma| <= 1
        _Gamma3 = np.where(np.abs(_Gamma3_plus) <= 1, _Gamma3_plus, _Gamma3_neg)
        
        return _Gamma3

    # ==================================================================================================================
    # 2-Port Solution Where Position is Determined Solely by airline L_air and L
    # Interative Solution
    # [2], Page 18, Equation 2.51 and 2.52
    # ==================================================================================================================