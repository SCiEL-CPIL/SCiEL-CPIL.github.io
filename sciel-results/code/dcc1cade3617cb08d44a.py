import math
import time
import numpy as np

_START = time.time()
DEADLINE = 245.0  # wall-clock safety margin (evaluator kills at 270s)

def _time_left():
    return DEADLINE - (time.time() - _START)

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3(theta, phi, lam):
    c, s = math.cos(theta/2), math.sin(theta/2)
    return np.array([[c, -math.exp(1j*lam)*s],
                     [math.exp(1j*phi)*s, math.exp(1j*(phi+lam))*c]], dtype=complex)
