"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def _qaoa_expected_cut(n, edges, gammas, betas):
    """
    Compute the expected cut size for a given QAOA circuit with p layers.
    
    The circuit starts in |+>^n and applies for each layer l:
      1. exp(-i gamma_l * C) where C is the MaxCut Hamiltonian
      2. exp(-i beta_l * sum_i X_i)
    
    The expected cut is sum over edges (i,j) of P[i != j] where P is the
    probability distribution of the final state.
    
    For efficiency with small n, we can use the fact that:
    - The X mixer is diagonal in the Z basis? No, X flips Z basis states.
    - Actually, let's think about this more carefully.
    
    The initial state is |+>^n = (1/sqrt(2^n)) sum_z |z>
    
    After applying the cost unitary U_C = exp(-i gamma C), the state becomes:
    |psi> = U_C |+>^n
    
    Then U_X = exp(-i beta sum X_i) = prod_i exp(-i beta X_i) (since they commute)
    
    For a single qubit, exp(-i beta X) = cos(beta) I - i sin(beta) X
    
    The expected cut for edge (i,j) is:
    <psi| (1 - Z_i Z_j)/2 |psi> where |psi> is the final state.
    
    Actually, let me reconsider. The MaxCut Hamiltonian C = sum_{(i,j) in E} (1 - Z_i Z_j)/2
    So the cost unitary is U_C = exp(-i gamma * C)
    
    For the expected cut, we need:
    E[cut] = sum_{(i,j) in E} <(1 - Z_i Z_j)/2>
    
    This is a quantum expectation value. For small n (6-10), we can compute this
    using matrix operations or by tracking the state.
    
    However, for n=10, the state space is 2^10 = 1024, which is manageable.
    
    Let's implement this using the state vector approach for small n.
    """
    if n > 12:
        # Fallback for larger n - use a simpler approximation
        # For now, return a heuristic value
        return 0.5 * len(edges)
    
    dim = 1 << n
    
    # Build the cost Hamiltonian C
    # C = sum_{(i,j) in E} (1 - Z_i Z_j)/2
    # For each edge, (1 - Z_i Z_j)/2 is a diagonal matrix in the Z basis
    # with eigenvalue 1 if z_i != z_j, 0 otherwise
    
    # The cost unitary U_C = exp(-i gamma * C) is diagonal in the Z basis
    # since C is diagonal in the Z basis
    # U_C[z, z] = exp(-i gamma * c_z) where c_z is the cut value for configuration z
    
    # The mixer U_X = exp(-i beta * sum X_i)
    # This is NOT diagonal in the Z basis, but it is a product of single-qubit gates
    
    # Strategy: 
    # 1. Start with |+>^n state
    # 2. For each layer:
    #    a. Apply U_C (diagonal, easy)
    #    b. Apply U_X (product of single-qubit X rotations)
    # 3. Compute expected cut
    
    # Initial state: |+>^n = (1/sqrt(2^n)) * ones(2^n)
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute cut values for all configurations
    cut_values = np.zeros(dim, dtype=np.float64)
    for z in range(dim):
        cut_val = 0
        for (i, j) in edges:
            bit_i = (z >> i) & 1
            bit_j = (z >> j) & 1
            if bit_i != bit_j:
                cut_val += 1
        cut_values[z] = cut_val
    
    for gamma, beta in zip(gammas, betas):
        # Apply cost unitary: state[z] *= exp(-i gamma * cut_values[z])
        state = state * np.exp(-1j * gamma * cut_values)
        
        # Apply mixer: exp(-i beta * sum X_i) = prod_i exp(-i beta X_i)
        # For each qubit i, exp(-i beta X_i) acts on the i-th qubit
        # In the computational basis, this means:
        # For each pair of states that differ only in bit i, apply:
        # [cos(beta), -i sin(beta); -i sin(beta), cos(beta)]
        
        for i in range(n):
            # For each pair (z, z') where z' = z XOR (1 << i)
            # Apply the 2x2 rotation
            mask = 1 << i
            for z in range(dim):
                if (z & mask) == 0:
                    z2 = z | mask
                    a = state[z]
                    b = state[z2]
                    state[z] = np.cos(beta) * a - 1j * np.sin(beta) * b
                    state[z2] = -1j * np.sin(beta) * a + np.cos(beta) * b
    
    # Compute expected cut
    # E[cut] = sum_{(i,j) in E} <(1 - Z_i Z_j)/2>
    # <Z_i Z_j> = sum_z |psi_z|^2 * z_i * z_j where z_i = (-1)^bit_i
    # Actually, Z_i Z_j has eigenvalue (+1) if bits are same, (-1) if different
    # So (1 - Z_i Z_j)/2 has eigenvalue 1 if bits differ, 0 if same
    
    probs = np.abs(state) ** 2
    expected_cut = 0.0
    for (i, j) in edges:
        # Sum over all states where bit_i != bit_j
        mask_i = 1 << i
        mask_j = 1 << j
        for z in range(dim):
            bit_i = (z >> i) & 1
            bit_j = (z >> j) & 1
            if bit_i != bit_j:
                expected_cut += probs[z]
    
    return expected_cut

def _exhaustive_maxcut(n, edges):
    """Compute the exact maximum cut by brute force for small n."""
    if n > 15:
        # Fallback: use a heuristic
        return 0.5 * len(edges)
    
    best = 0
    for z in range(1 << n):
        cut = 0
        for (i, j) in edges:
            bit_i = (z >> i) & 1
            bit_j = (z >> j) & 1
            if bit_i != bit_j:
                cut += 1
        if cut > best:
            best = cut
    
    return best

def _optimize_angles(n, edges, p, time_limit):
    """
    Optimize QAOA angles for a given graph using gradient-free optimization.
    Uses Nelder-Mead or a custom local search.
    """
    from scipy.optimize import minimize
    
    # Initial guess
    x0 = np.array([0.35, 0.25] * p, dtype=np.float64)
    
    def objective(x):
        gammas = x[0::2]
        betas = x[1::2]
        return -_qaoa_expected_cut(n, edges, gammas, betas)
    
    # Use Nelder-Mead with time budget
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Try a few different initial points
    np.random.seed(42)
    
    for start_idx in range(5):
        if _time_left() < 10:
            break
        
        if start_idx == 0:
            x_init = x0.copy()
        else:
            x_init = x0 + np.random.randn(2 * p) * 0.1
        
        try:
            result = minimize(
                objective,
                x_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-4,
                    'fatol': 1e-4,
                    'adaptive': True
                }
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x

def angles(n, edges, p):
    """
    Return QAOA angles for a given graph.
    
    Args:
        n: number of vertices
        edges: list of (i, j) edges
        p: number of QAOA layers
    
    Returns:
        np.array of shape (p, 2), rows are (gamma, beta) in radians
    """
    # Check time
    if _time_left() < 5:
        # Return default angles
        return np.tile([0.35, 0.25], (p, 1))
    
    # Optimize angles for this specific graph
    time_limit = min(30, _time_left() - 5)
    
    try:
        x_opt = _optimize_angles(n, edges, p, time_limit)
        gammas = x_opt[0::2]
        betas = x_opt[1::2]
        result = np.column_stack([gammas, betas])
        return result
    except Exception:
        # Fallback to default
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
