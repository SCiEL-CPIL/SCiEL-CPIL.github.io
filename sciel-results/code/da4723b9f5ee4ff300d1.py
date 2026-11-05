"""Optimized quantum circuit parameters using global stochastic search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

_DEADLINE = 148.0
_START_TIME = time.time()

def _check_deadline(margin=3.0):
    return (time.time() - _START_TIME) < (_DEADLINE - margin)

def _build_hamiltonian(n, coupling, field):
    """Build H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    total_dim = 2 ** n
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    H = np.zeros((total_dim, total_dim), dtype=complex)
    
    # ZZ terms
    for i in range(n - 1):
        term = np.array([[1]], dtype=complex)
        for q in range(n):
            if q == i or q == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += -coupling * term

    # X terms
    for i in range(n):
        term = np.array([[1]], dtype=complex)
        for q in range(n):
            if q == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += -field * term

    return H

def _circuit_state_vectorized(params, n, layers):
    """
    Vectorized circuit state evolution.
    Uses tensor reshaping for Ry and bitwise operations for CZ.
    """
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to all qubits
        # Ry(theta) = [[c, -s], [s, c]]
        # We apply this to each qubit.
        # Since the operations on different qubits in a layer commute? No, they don't necessarily, 
        # but Ry on different qubits commute. So we can apply them in any order or even "simultaneously" 
        # if we handle the tensor structure correctly.
        # However, the standard definition is sequential. Since they commute, the result is the same.
        # Let's apply them one by one using tensor slicing.
        
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            # Reshape state to tensor [2]*n
            # Qubit q is at axis n-1-q (little-endian bit order in index, MSB in tensor axis 0)
            # Wait, let's verify the axis mapping again.
            # Index idx. Bit q is (idx >> q) & 1.
            # In a numpy array reshaped to [2]*n, axis 0 is the most significant bit (slowest varying).
            # Axis n-1 is the least significant bit (fastest varying).
            # So qubit q (bit q) corresponds to axis n-1-q.
            
            axis = n - 1 - q
            state_tensor = state.reshape([2] * n)
            
            # We need to mix the slices along 'axis'.
            # Slice 0: index along axis is 0.
            # Slice 1: index along axis is 1.
            
            # Create new tensor
            new_tensor = np.zeros_like(state_tensor)
            
            # Get slices
            # We need to be careful with np.take or slicing.
            # Let's use np.moveaxis to bring the target axis to the front for easier manipulation?
            # Or just use slicing with None.
            
            # Example n=2, q=0. axis=1.
            # state_tensor[i, j]. i is qubit 1, j is qubit 0.
            # Ry on qubit 0 acts on j.
            # new[i, 0] = c*old[i, 0] - s*old[i, 1]  <-- WAIT.
            # Ry = [[c, -s], [s, c]].
            # |0> -> c|0> + s|1>.
            # |1> -> -s|0> + c|1>.
            # So:
            # new_0 = c * old_0 + s * old_1 ? No.
            # U |0> = U [1, 0]^T = [c, s]^T. So amplitude of |0> is c, |1> is s.
            # U |1> = U [0, 1]^T = [-s, c]^T. So amplitude of |0> is -s, |1> is c.
            
            # Let's denote old amplitudes as a_0 (bit=0) and a_1 (bit=1).
            # New amplitude for bit=0: c * a_0 - s * a_1.
            # New amplitude for bit=1: s * a_0 + c * a_1.
            
            # Let's extract slices.
            # slice_0: where bit q is 0.
            # slice_1: where bit q is 1.
            
            # Using np.take_along_axis or manual slicing is tricky with dynamic axes.
            # Let's use a helper to get slices.
            
            # Actually, for n<=6, a simple loop over indices is fast enough and less error-prone.
            # 2^6 = 64 states. 3 layers * 6 qubits * 64 indices = 1152 operations.
            # This is extremely fast in Python if vectorized per qubit?
            # Let's try a vectorized approach using bit masks.
            
            # Create a mask for bit q.
            # For each index idx, if bit q is 0, it contributes to new[idx] and new[idx | (1<<q)].
            # If bit q is 1, it contributes to new[idx] and new[idx ^ (1<<q)].
            
            # This logic is hard to vectorize simply without advanced indexing.
            # Let's stick to the loop for Ry but optimize it.
            # Or use the tensor approach correctly.
            
            # Tensor approach:
            # Move axis 'axis' to 0.
            state_moved = np.moveaxis(state_tensor, axis, 0)
            # Now state_moved[0, ...] is slice 0, state_moved[1, ...] is slice 1.
            
            new_moved = np.zeros_like(state_moved)
            new_moved[0] = c * state_moved[0] - s * state_moved[1]
            new_moved[1] = s * state_moved[0] + c * state_moved[1]
            
            # Move axis back
            state = np.moveaxis(new_moved, 0, axis).reshape(-1)
            
        if layer < layers - 1:
            # Apply CZ on adjacent pairs
            # CZ flips sign if both bits are 1.
            # We can do this by iterating over pairs and using masks.
            for q in range(n - 1):
                mask1 = (np.arange(2 ** n) >> q) & 1
                mask2 = (np.arange(2 ** n) >> (q + 1)) & 1
                both_1 = (mask1 == 1) & (mask2 == 1)
                state[both_1] *= -1.0

    return state

def _objective(x, n, coupling, field, layers, H, ground_energy):
    params = x.reshape(layers, n)
    state = _circuit_state_vectorized(params, n, layers)
    energy = np.vdot(state, H @ state).real
    return (energy - ground_energy) / n

def angles(n, coupling, field, layers):
    global _START_TIME
    _START_TIME = time.time()
    
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = np.min(np.linalg.eigvalsh(H).real)
    
    # Bounds for angles: [-pi, pi]
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    # Differential Evolution is a global optimizer.
    # It requires a population size.
    # For small n (4, 6) and layers=3, dimension is 12 or 18.
    # Pop size 10-20 is usually good.
    
    # We need to be careful with the time limit.
    # DE can be slow. We will use a tight budget.
    
    # Strategy:
    # 1. Run DE with a moderate number of generations.
    # 2. If time remains, refine with Nelder-Mead or L-BFGS-B from the best DE result.
    
    best_result = None
    best_score = np.inf
    
    try:
        # Use a callback to check time?
        # SciPy's differential_evolution doesn't have a direct time callback, 
        # but we can limit maxiter and tol.
        
        # Let's estimate time per function call.
        # Vectorized state is fast.
        
        de_result = differential_evolution(
            _objective,
            bounds,
            args=(n, coupling, field, layers, H, ground_energy),
            maxiter=50,
            tol=1e-8,
            popsize=15,
            seed=42,
            polish=False # We will polish manually to save overhead and control time
        )
        
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x
            
    except Exception:
        pass
    
    # If we have time, polish the result with L-BFGS-B
    if _check_deadline(margin=5.0) and best_result is not None:
        try:
            from scipy.optimize import minimize
            polish_result = minimize(
                _objective,
                best_result,
                args=(n, coupling, field, layers, H, ground_energy),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if polish_result.fun < best_score:
                best_score = polish_result.fun
                best_result = polish_result.x
        except Exception:
            pass

    if best_result is None:
        # Fallback to zero if DE failed or timed out completely
        best_result = np.zeros(layers * n)
        
    best_params = best_result.reshape(layers, n)
    best_params = np.nan_to_num(best_params, nan=0.0, posinf=0.0, neginf=0.0)
    return best_params
# EVOLVE-BLOCK-END
