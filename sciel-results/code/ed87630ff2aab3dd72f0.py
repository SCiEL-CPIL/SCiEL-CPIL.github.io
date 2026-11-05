<<<<<<< SEARCH
"""Optimized product-formula coefficients for 6-site XXZ Heisenberg chain.
Uses a model-and-optimize approach: build the actual Hamiltonian, evaluate
the true spectral-norm error, and optimize coefficients via scipy.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.linalg import expm

_START = time.time()
DEADLINE = _START + 130.0  # safety margin before 150s kill
=======
"""Optimized product-formula coefficients for 6-site XXZ Heisenberg chain.
Uses a model-and-optimize approach: build the actual Hamiltonian, evaluate
the true spectral-norm error, and optimize coefficients via custom methods.
"""
import time
import numpy as np
from scipy.linalg import expm

_START = time.time()
DEADLINE = _START + 130.0  # safety margin before 150s kill
>>>>>>> REPLACE

