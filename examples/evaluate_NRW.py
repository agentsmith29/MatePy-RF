import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import logging
from rich.logging import RichHandler
import sys

#include the matpyrf packgae
sys.path.append(r'MatePy-RF\src')

from matepyrf.NRW.NicholsonRossWeirConverstion import NicholsonRossWeirConverstion
from matepyrf.NRW.NewNonIterativeConversion import NewNonIterative
from matepyrf.Waveguide import Waveguide

# for constants
from scipy.constants import c as c_const
# setup logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
FORMAT = "%(message)s"
logging.basicConfig(
    level="DEBUG", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

# disable matplotlib.font_manager
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('PIL.PngImagePlugin').disabled = True



# Create a waveguide object for WR12 sized waveguide
wr12 = Waveguide(101.6e-3, 3.0988e-3, name='WR12')

# Open the simulation data file
nw_sim2 = rf.Network(r'smatrix__wr12__60freq90__eps_2_8__mur_1_0__tanDelta_0_01__l1_81_6mm__l2_5_0mm_L_15_0mm.s2p')

# Set the correct reference plance positions 
lp2 = 5e-3
l_total = 101.6e-3
l_sample = 15e-3
lp1 = l_total - l_sample - lp2

# Create a NicholsonRossWeirConverstion object
nrw2 = NicholsonRossWeirConverstion(nw_sim2, waveguide_system=wr12, sample_length=l_sample , l_p1=lp1, l_p2=lp2)
# plot the results
nrw2.plot()
# show the results
plt.show()




