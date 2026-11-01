<<<<<<< SEARCH
"""
Refined program for packing 50 equal circles in a unit circle.
Strategy:
1. Robust Soft-Min Surrogate with vectorized gradient computation.
2. Multi-stage optimization: coarse annealing -> fine annealing -> high-precision polish.
3. Diverse initializations: ring-based, hexagonal lattice, and random.
4. Efficient score computation and gradient evaluation.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 118.0  # Safety margin for 120s limit
=======
"""
Refined program for packing 50 equal circles in a unit circle.
Strategy:
1. Rigid-body dynamics with wall particles for self-organizing dense packing.
2. Robust Soft-Min Surrogate with vectorized gradient computation.
3. Multi-stage optimization: dynamics -> coarse annealing -> fine annealing -> high-precision polish.
4. Diverse initializations: ring-based, hexagonal lattice, and random.
5. Efficient score computation and gradient evaluation.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 118.0  # Safety margin for 120s limit
N_WALLS = 200  # Number of wall particles to model the container boundary
>>>>>>> REPLACE

