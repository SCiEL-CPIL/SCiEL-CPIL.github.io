import numpy as np
import time
import math
import random

_START = time.time()
_DEADLINE = _START + 130.0  # Safety margin before 150s kill

N = 24
SQRTN = math.sqrt(N)

def _time_left():
    return _DEADLINE - time.time()

def _supnorm_fast(coeffs, n_grid=32768):
    """
    Fast sup-norm estimate via FFT on a dense grid.
    """
    padded = np.zeros(n_grid, dtype=float)
    padded[:N] = coeffs
    vals = np.fft.fft(padded)
    return float(np.max(np.abs(vals)))

def _golay_rudin_shapiro(n):
    """
    Generate Golay-Rudin-Shapiro sequence of length n.
    """
    p = [1]
    while len(p) < n:
        p = p + [-c for c in p] # Wait, standard GRS is p, p-q or similar?
        # Actually, the standard recursion is:
        # a_{2n} = a_n, a_{2n+1} = -a_n
        # Let's just generate the full sequence up to power of 2 >= n and truncate.
        # Or use the property: GR sequence is defined by binary weight parity.
        # c_k = (-1)^{popcount(k)}
        
    # Let's use the parity definition, it's simpler and correct for GRS.
    coeffs = []
    for k in range(n):
        # Count bits in binary representation of k
        # Python 3: bin(k).count('1')
        if bin(k).count('1') % 2 == 0:
            coeffs.append(1)
        else:
            coeffs.append(-1)
    return coeffs

def _hill_climb(coeffs):
    """
    Perform 1-bit flip hill climbing until local minimum.
    """
    coeffs = list(coeffs)
    current_val = _supnorm_fast(coeffs)
    
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        # We iterate through bits multiple times to ensure we don't miss a path
        # due to the order of updates (though steepest descent usually picks best).
        # For small N, checking all bits is cheap.
        for k in range(N):
            if _time_left() < 5.0:
                break
            
            # Flip bit k
            coeffs[k] = -coeffs[k]
            new_val = _supnorm_fast(coeffs)
            
            if new_val < current_val - 1e-12:
                current_val = new_val
                improved = True
            else:
                # Revert
                coeffs[k] = -coeffs[k]
                
    return coeffs, current_val

def _two_bit_hill_climb(coeffs):
    """
    Perform 2-bit flip hill climbing.
    """
    coeffs = list(coeffs)
    current_val = _supnorm_fast(coeffs)
    
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        for i in range(N):
            for j in range(i + 1, N):
                if _time_left() < 5.0:
                    break
                
                coeffs[i] = -coeffs[i]
                coeffs[j] = -coeffs[j]
                
                new_val = _supnorm_fast(coeffs)
                
                if new_val < current_val - 1e-12:
                    current_val = new_val
                    improved = True
                else:
                    # Revert
                    coeffs[i] = -coeffs[i]
                    coeffs[j] = -coeffs[j]
                    
        if _time_left() < 5.0:
            break

    return coeffs, current_val

def construct():
    """
    Return 24 coefficients in {+1, -1} minimizing max_{|z|=1} |P(z)| / sqrt(24).
    """
    random.seed(42)
    np.random.seed(42)
    
    # Start with Golay-Rudin-Shapiro
    best_coeffs = _golay_rudin_shapiro(N)
    best_val = _supnorm_fast(best_coeffs)
    
    # Phase 1: 1-bit Hill Climb
    best_coeffs, best_val = _hill_climb(best_coeffs)
    
    # Phase 2: 2-bit Hill Climb (to escape shallow local minima if any)
    best_coeffs, best_val = _two_bit_hill_climb(best_coeffs)
    
    # Phase 3: Random Restarts with 2-bit perturbations if time allows
    # We try a few random flips and re-climb
    restarts = 0
    while _time_left() > 10.0:
        # Perturb best solution
        candidate = list(best_coeffs)
        # Flip 2 random bits
        i, j = random.sample(range(N), 2)
        candidate[i] = -candidate[i]
        candidate[j] = -candidate[j]
        
        # Climb from here
        new_coeffs, new_val = _hill_climb(candidate)
        # Optionally 2-bit climb too if time is plentiful
        if _time_left() > 20.0:
            new_coeffs, new_val = _two_bit_hill_climb(new_coeffs)
            
        if new_val < best_val - 1e-12:
            best_val = new_val
            best_coeffs = new_coeffs
            restarts += 1
            # If we improved, we might have found a better basin, keep going
        else:
            # If no improvement, try a different random perturbation
            # To avoid getting stuck in the same basin, we can increase perturbation size
            # But for now, just random 2-bit flips are good.
            pass
            
        restarts += 1
        if restarts > 50: # Cap restarts to avoid infinite loop if no improvement
             break

    return best_coeffs

if __name__ == "__main__":
    result = construct()
    print(result)
