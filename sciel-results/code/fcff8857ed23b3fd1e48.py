<<<<<<< SEARCH
"""Combined: model-and-optimize with efficient analytic gradient + alpha-based init + multi-start polish."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import roots_laguerre, roots_genlaguerre
from numpy.polynomial.polynomial import polyfromroots, polymul
=======
"""Combined: model-and-optimize with efficient analytic gradient + alpha-based init + multi-start polish."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import roots_laguerre, roots_genlaguerre
from numpy.polynomial.polynomial import polyfromroots, polymul
from scipy.linalg import eig
>>>>>>> REPLACE
