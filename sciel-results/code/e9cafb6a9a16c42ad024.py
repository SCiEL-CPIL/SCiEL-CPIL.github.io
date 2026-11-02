"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as optimize

N = 400

# EVOLVE-BLOCK-START
def _autoconv(x):
    """Compute linear autoconvolution using FFT."""
    n_fft = 2 * N
    fx = np.fft.fft(x, n=n_fft)
    power = fx * np.conj(fx)
    c = np.fft.ifft(power).real
    # First N elements are lags 0 to N-1
    return c[:N]

def _score(x):
    """Calculate the score: max(abs(autoconv)) / (sum(x))^2"""
    c = _autoconv(x)
    max_abs_c = np.max(np.abs(c))
    sum_x = np.sum(x)
    sq_sum = sum_x * sum_x
    if sq_sum < 1e-12:
        return float('inf')
    return max_abs_c / sq_sum

def _soft_score_and_grad(x, beta=50.0):
    """
    Compute smooth surrogate score and its gradient.
    Uses soft-max of |autoconv| to make the objective differentiable.
    """
    c = _autoconv(x)
    abs_c = np.abs(c)
    
    # Soft-max: log(sum(exp(beta * abs_c))) / beta
    # This approximates max(abs_c) when beta is large
    max_val = np.max(abs_c)
    exp_vals = np.exp(beta * (abs_c - max_val))  # subtract max for numerical stability
    log_sum_exp = np.log(np.sum(exp_vals)) + max_val * beta
    soft_max = log_sum_exp / beta
    
    sum_x = np.sum(x)
    sq_sum = sum_x * sum_x
    if sq_sum < 1e-12:
        return float('inf'), np.zeros_like(x)
    
    # Score = soft_max / sq_sum
    score = soft_max / sq_sum
    
    # Gradient computation
    # d(score)/dx = (1/sq_sum) * d(soft_max)/dx - (soft_max / sq_sum^2) * 2 * sum_x * dx
    # 
    # d(soft_max)/dx[k] = (1/beta) * sum_j d(soft_max)/d(abs_c[j]) * d(abs_c[j])/dx[k]
    # d(soft_max)/d(abs_c[j]) = beta * exp(beta * abs_c[j]) / sum(exp(beta * abs_c))
    #                          = beta * exp_vals[j] / sum(exp_vals)
    # d(abs_c[j])/dx[k] = sign(c[j]) * dc[j]/dx[k]
    # 
    # dc[j]/dx[k]: autoconv[j] = sum_i x[i] * x[i+j]
    # d(autoconv[j])/dx[k] = sum_i (delta_{i,k} * x[i+j] + x[i] * delta_{i+j,k})
    #                      = x[k+j] (if k+j < N) + x[k-j] (if k-j >= 0)
    
    weights = exp_vals / np.sum(exp_vals)  # normalized weights for soft-max
    
    # Compute gradient of soft_max with respect to x
    grad_soft = np.zeros_like(x)
    for k in range(N):
        # d(soft_max)/dx[k] = sum_j weights[j] * sign(c[j]) * (x[k+j] + x[k-j])
        # where x[k+j] is 0 if k+j >= N, x[k-j] is 0 if k-j < 0
        for j in range(N):
            term = 0.0
            if k + j < N:
                term += x[k + j]
            if k - j >= 0:
                term += x[k - j]
            grad_soft[k] += weights[j] * np.sign(c[j]) * term
    
    # d(score)/dx = (1/sq_sum) * grad_soft - (2 * soft_max * sum_x / sq_sum^2) * ones
    grad_score = grad_soft / sq_sum - (2.0 * soft_max * sum_x / (sq_sum * sq_sum)) * np.ones_like(x)
    
    return score, grad_score

def _local_search(x, max_steps=500):
    """
    Local search by flipping individual signs to minimize the true score.
    """
    current_score = _score(x)
    x = x.copy()
    
    for step in range(max_steps):
        improved = False
        # Try flipping each position
        for i in range(N):
            old_val = x[i]
            x[i] *= -1
            new_score = _score(x)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                x[i] = old_val
        
        if not improved:
            break
    
    return x, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with all ones (known good solution)
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Perform gradient-based continuous optimization on a smooth surrogate
    # Then snap to signs and do local search
    
    num_restarts = 3
    beta = 30.0  # Softness parameter
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create initial point: all ones with some random perturbations
        if r == 0:
            x0 = np.ones(N)
        else:
            # Random restart: mix of signs
            x0 = np.ones(N)
            num_flips = N // 4
            indices = np.random.choice(N, num_flips, replace=False)
            x0[indices] *= -1
        
        # Gradient descent on smooth surrogate
        x_curr = x0.copy()
        lr = 0.01
        patience = 20
        no_improve = 0
        best_soft_score = _soft_score_and_grad(x_curr, beta)[0]
        
        for step in range(500):
            if time.time() > deadline:
                break
            
            score, grad = _soft_score_and_grad(x_curr, beta)
            
            if np.isnan(score) or np.isinf(score):
                break
            
            if score < best_soft_score:
                best_soft_score = score
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                break
            
            # Gradient step with clipping to keep values in [-1, 1]
            x_curr = x_curr - lr * grad
            x_curr = np.clip(x_curr, -1.0, 1.0)
        
        # Snap to signs
        x_snapped = np.sign(x_curr)
        x_snapped[x_snapped == 0] = 1.0  # Handle zeros
        
        # Local search from snapped solution
        x_ls, score_ls = _local_search(x_snapped, max_steps=200)
        
        if score_ls < best_score:
            best_score = score_ls
            best_x = x_ls.copy()
    
    # Final local search from best solution
    x_final, score_final = _local_search(best_x, max_steps=300)
    if score_final < best_score:
        best_score = score_final
        best_x = x_final
    
    return best_x
# EVOLVE-BLOCK-END
