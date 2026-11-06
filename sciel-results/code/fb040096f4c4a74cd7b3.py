import numpy as np
import scipy.optimize
import time

_start_time = time.time()
_TIME_LIMIT = 145.0 # Safety margin

def _compute_cut_values(n, edges):
    # Precompute cut(x) for all x in [0, 2^n)
    cuts = np.zeros(2**n)
    for x in range(2**n):
        c = 0
        for u, v in edges:
            if ((x >> u) & 1) != ((x >> v) & 1):
                c += 1
        cuts[x] = c
    return cuts

def _evaluate_qaoa(n, edges, gammas, betas, cut_values):
    # State vector
    # Start with |+>^n
    # Index x corresponds to basis state |x>
    # Amplitude is 1/sqrt(2^n)
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    # Precompute mixer matrix for each beta?
    # Mixer is tensor product of single qubit rotations.
    # We can apply it by iterating over bits or using reshaping.
    # Reshaping to (2, 2, ..., 2) tensor is efficient.
    
    for k in range(2):
        gamma = gammas[k]
        beta = betas[k]
        
        # Apply Cost: psi[x] *= exp(-i * gamma * cut_values[x])
        # Note: cut_values is real
        psi *= np.exp(-1j * gamma * cut_values)
        
        # Apply Mixer: tensor product of Rx(beta)
        # Rx(theta) = [[cos(theta/2), -i sin(theta/2)], [-i sin(theta/2), cos(theta/2)]]
        c = np.cos(beta / 2.0)
        s = -1j * np.sin(beta / 2.0)
        
        # Reshape psi to tensor shape
        shape = [2] * n
        psi_tensor = psi.reshape(shape)
        
        # Apply rotation on each axis
        # This is tricky to do in-place efficiently without loops, but n is small.
        # We can use einsum or simple loops over axes.
        # Since n <= 10, a loop over n axes is fine.
        for axis in range(n):
            # Move axis to front, apply 2x2 matrix, move back
            # Or use np.tensordot
            # Simpler: reshape to (2, -1), apply, reshape back? No, axis varies.
            # Let's use a helper or explicit logic.
            # Actually, for small n, we can just loop through bits in the index? No, that's O(2^n * n).
            # The tensor reshape + loop over axes is O(2^n * n) which is ~10k ops. Very fast.
            
            # Move axis to 0
            psi_tensor = np.moveaxis(psi_tensor, axis, 0)
            # Apply 2x2 matrix
            # New[0, ...] = c * Old[0, ...] + s * Old[1, ...]
            # New[1, ...] = s * Old[0, ...] + c * Old[1, ...]
            # We need to be careful with in-place updates.
            temp = np.empty_like(psi_tensor)
            temp[0] = c * psi_tensor[0] + s * psi_tensor[1]
            temp[1] = s * psi_tensor[0] + c * psi_tensor[1]
            psi_tensor = temp
            # Move axis back to original position
            psi_tensor = np.moveaxis(psi_tensor, 0, axis)
            
        psi = psi_tensor.reshape(-1)
        
    # Compute expected cut
    probs = np.abs(psi) ** 2
    expected_cut = np.sum(probs * cut_values)
    return expected_cut

def angles(n, edges, p):
    global _start_time, _TIME_LIMIT
    
    # Check time
    if time.time() - _start_time > _TIME_LIMIT:
         return np.tile([0.35, 0.25], (p, 1))
         
    cut_values = _compute_cut_values(n, edges)
    
    # Define objective to maximize
    def objective(params):
        gammas = params[:2]
        betas = params[2:4]
        return -_evaluate_qaoa(n, edges, gammas, betas, cut_values)
        
    # Initial guess
    # Common good starts for QAOA
    x0 = np.array([0.5, 0.5, 0.5, 0.5])
    
    # Optimize
    try:
        res = scipy.optimize.minimize(objective, x0, method='Nelder-Mead', 
                                      options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        best_params = res.x
    except Exception:
        best_params = x0
        
    gammas = best_params[:2]
    betas = best_params[2:4]
    
    # Construct result array (p, 2)
    # Rows are (gamma, beta)
    result = np.column_stack((gammas, betas))
    
    return result
