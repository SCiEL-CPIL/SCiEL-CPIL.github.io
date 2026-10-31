"""LJ38 cluster global minimum via Multi-start Simulated Annealing."""
import numpy as np
from scipy.optimize import minimize
import time

N = 38

def _lj_energy_grad(pos):
    """
    Calculate total Lennard-Jones energy and gradient.
    Vectorized for performance.
    """
    # Calculate all pairwise distances
    # pos shape: (N, 3)
    # diff shape: (N, N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(r2, 1.0)
    
    # We only need upper triangle for energy, but let's compute full then sum half
    # Or just use upper triangle indices for efficiency in memory if N is large, but N=38 is small.
    # Let's stick to full matrix for simplicity in vectorization, then mask.
    
    # inv_r2 = 1/r2
    # inv_r6 = 1/r6
    # E = 4 * sum (inv_r12 - inv_r6)
    # dE/dr2 = 4 * (-12 * r^-14 + 6 * r^-8) = 4 * r^-14 * (-12 + 6*r2)
    # dE/dpos_i = sum_j dE/dr2 * d(r2)/dpos_i
    # d(r2)/dpos_i = -2 * (pos_i - pos_j) = 2 * (pos_j - pos_i)
    
    # Let's compute terms
    r2_safe = r2.copy()
    r2_safe[0,0] = 1.0 # Just to be safe, though we mask later
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    inv_r12 = inv_r6 * inv_r6
    
    # Energy term
    e_pair = 4.0 * (inv_r12 - inv_r6)
    
    # Mask diagonal to zero
    np.fill_diagonal(e_pair, 0.0)
    energy = 0.5 * np.sum(e_pair) # Sum all pairs, divide by 2
    
    # Gradient
    # grad_i = sum_{j != i} dE/dpos_i
    # dE/dpos_i = 4 * (-12 * r^-14 + 6 * r^-8) * (pos_i - pos_j) ? 
    # Let's re-derive carefully.
    # U = 4(r^-12 - r^-6)
    # dU/dr = 4(-12r^-13 + 6r^-7)
    # dU/dpos_i = dU/dr * dr/dpos_i
    # r = |pos_i - pos_j|
    # dr/dpos_i = (pos_i - pos_j) / r
    # dU/dpos_i = 4(-12r^-13 + 6r^-7) * (pos_i - pos_j) / r
    # = 4(-12r^-14 + 6r^-8) * (pos_i - pos_j)
    
    # Let's compute coeff = 4 * (-12 * inv_r14 + 6 * inv_r8)
    # inv_r8 = inv_r2 * inv_r6
    # inv_r14 = inv_r2 * inv_r12
    inv_r8 = inv_r2 * inv_r6
    inv_r14 = inv_r2 * inv_r12
    
    coeff = 4.0 * (-12.0 * inv_r14 + 6.0 * inv_r8)
    
    # coeff shape: (N, N)
    # diff shape: (N, N, 3) where diff[i,j] = pos[i] - pos[j]
    # grad[i] = sum_j coeff[i,j] * diff[i,j]
    
    # Zero out diagonal in coeff
    np.fill_diagonal(coeff, 0.0)
    
    # grad = coeff[:,:,None] * diff
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _relax(pos):
    """Local L-BFGS relaxation."""
    try:
        # Use a wrapper for minimize
        def fun(x):
            e, g = _lj_energy_grad(x)
            return e, g
            
        res = minimize(fun, pos, jac=True, method='L-BFGS-B',
                       options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10})
        if res.success or res.fun < _lj_energy_grad(pos)[0]:
            return res.x
        return pos
    except Exception:
        return pos

def _simulated_annealing(pos, T_start=10.0, T_end=0.01, steps=2000, step_size=0.2):
    """
    Run Simulated Annealing starting from pos.
    """
    current_pos = pos.copy()
    current_energy, _ = _lj_energy_grad(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    # Cooling schedule
    cooling = (T_end / T_start) ** (1.0 / steps)
    
    for i in range(steps):
        T = T_start * (cooling ** i)
        
        # Propose new position
        # Gaussian perturbation
        delta = np.random.randn(N, 3) * step_size
        new_pos = current_pos + delta
        
        new_energy, _ = _lj_energy_grad(new_pos)
        
        # Acceptance criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            current_pos = new_pos
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
                
    return best_pos, best_energy

def construct():
    """
    Construct the LJ38 cluster using Multi-start Simulated Annealing.
    """
    start_time = time.time()
    timeout = 110.0 # Leave 10s buffer
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    best_pos_global = None
    best_energy_global = np.inf
    
    # Number of restarts
    # We will run as many restarts as possible within the time limit.
    # Each SA run takes some time. Let's estimate.
    # N=38, steps=2000. Vectorized energy/grad is fast.
    # Let's try 20 restarts with 1000 steps each, or adaptively.
    
    restarts = 0
    while time.time() - start_time < timeout:
        # Generate random initial position
        # Scale to be roughly within the cluster size
        # Typical cluster diameter ~ 10-15 sigma
        pos0 = np.random.randn(N, 3) * 3.0
        
        # Run SA
        pos_sa, e_sa = _simulated_annealing(pos0, T_start=5.0, T_end=0.05, steps=1500, step_size=0.3)
        
        # Relax the best SA result
        pos_relaxed = _relax(pos_sa)
        e_relaxed, _ = _lj_energy_grad(pos_relaxed)
        
        if e_relaxed < best_energy_global:
            best_energy_global = e_relaxed
            best_pos_global = pos_relaxed.copy()
            
        restarts += 1
        
        # Check time
        if time.time() - start_time > timeout:
            break
            
    # If we didn't find anything (shouldn't happen), return random
    if best_pos_global is None:
        best_pos_global = np.random.randn(N, 3) * 3.0
        best_energy_global, _ = _lj_energy_grad(best_pos_global)
        
    # Final polish: Run a few more L-BFGS iterations on the best global
    # Sometimes L-BFGS stops a bit early.
    final_pos = _relax(best_pos_global)
    final_energy, _ = _lj_energy_grad(final_pos)
    
    if final_energy < best_energy_global:
        best_pos_global = final_pos
        best_energy_global = final_energy
        
    return best_pos_global
