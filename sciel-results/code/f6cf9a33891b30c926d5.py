"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

def _autoconvolve_max_abs(s):
    """Compute max |autoconvolution| for a sequence s."""
    ac = fftconvolve(s, s, mode='full')
    return np.max(np.abs(ac))

def _squared_integral(s):
    """Compute (sum s)^2."""
    return (np.sum(s)) ** 2

def _objective(x):
    """
    Objective function for optimization.
    x is a vector of continuous values in [-1, 1] (via sigmoid transform).
    We minimize max|autoconv| / (sum s)^2.
    """
    s = np.tanh(x)  # map to [-1, 1]
    # Avoid division by zero
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return 1e10
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    return max_ac / sq_int

def _soft_max_ac(s, temperature=1.0):
    """
    Differentiable surrogate for max |autoconvolution|.
    Uses log-sum-exp of (ac/temperature)^2 to approximate max ac^2.
    """
    ac = fftconvolve(s, s, mode='full')
    ac_sq = ac ** 2
    # log-sum-exp for numerical stability
    max_val = np.max(ac_sq)
    if max_val < 1e-10:
        return 0.0
    lse = max_val + np.log(np.sum(np.exp(ac_sq - max_val)))
    return np.sqrt(lse)

def _soft_objective(x, temperature=1.0):
    """
    Soft objective: use log-sum-exp surrogate for max |ac|.
    """
    s = np.tanh(x)
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return 1e10
    soft_max_ac = _soft_max_ac(s, temperature)
    return soft_max_ac / sq_int

def _score_sequence(s):
    """Compute the true score for a sequence."""
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return np.inf
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    return max_ac / sq_int

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    best_score = np.inf
    best_s = None
    
    # Initial random sign patterns
    np.random.seed(42)
    
    # Start with a few random initializations
    for init_iter in range(20):
        if time.time() > deadline:
            break
        
        # Random initial x (before tanh)
        x0 = np.random.randn(N) * 0.1
        
        # Use L-BFGS-B on the soft objective
        try:
            result = minimize(
                _soft_objective,
                x0,
                method='L-BFGS-B',
                args=(1.0,),
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            # Extract the solution
            s_continuous = np.tanh(result.x)
            
            # Binary refinement: threshold to signs
            s_binary = np.sign(s_continuous)
            s_binary[s_binary == 0] = 1.0
            
            # Evaluate the true objective on the binary solution
            score_bin = _score_sequence(s_binary)
            
            if score_bin < best_score:
                best_score = score_bin
                best_s = s_binary.copy()
                
        except Exception:
            continue
        
        # Also try the continuous solution directly (in case it's better)
        s_cont = np.tanh(result.x)
        score_cont = _score_sequence(s_cont)
        if score_cont < best_score:
            best_score = score_cont
            best_s = s_cont.copy()
    
    # Local search: sign flips on the best solution found
    if best_s is not None:
        current_s = best_s.copy()
        current_score = best_score
        
        # Determine if we are working with binary or continuous values
        is_binary = np.allclose(current_s, np.sign(current_s))
        
        for flip_iter in range(500):
            if time.time() > deadline:
                break
            
            # Try flipping a random index (or adjusting value if continuous)
            idx = np.random.randint(N)
            old_val = current_s[idx]
            
            if is_binary:
                current_s[idx] = -old_val
            else:
                # Small random perturbation for continuous case
                current_s[idx] = old_val + np.random.randn() * 0.1
                current_s[idx] = np.clip(current_s[idx], -1, 1)
            
            # Compute new score
            new_score = _score_sequence(current_s)
            
            if new_score < current_score:
                current_score = new_score
                best_score = new_score
                best_s = current_s.copy()
            else:
                # Revert
                current_s[idx] = old_val
        
        # Simulated annealing-like exploration
        for sa_iter in range(200):
            if time.time() > deadline:
                break
            
            idx = np.random.randint(N)
            old_val = current_s[idx]
            
            if is_binary:
                current_s[idx] = -old_val
            else:
                current_s[idx] = old_val + np.random.randn() * 0.2
                current_s[idx] = np.clip(current_s[idx], -1, 1)
            
            new_score = _score_sequence(current_s)
            
            delta = new_score - current_score
            # Accept with probability exp(-delta / T) where T decreases
            T = max(0.01, 0.1 * (1 - sa_iter / 200.0))
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_s = current_s.copy()
            else:
                current_s[idx] = old_val
    
    # If we didn't find anything, return all ones
    if best_s is None:
        return np.ones(N)
    
    # Return as a sequence (list or array)
    return best_s
# EVOLVE-BLOCK-END
