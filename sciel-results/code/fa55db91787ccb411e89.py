"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def _maxcut_optimum(n, edges):
    """Compute the exact maximum cut value for a graph."""
    if n == 0:
        return 0
    best = 0
    # For small n (<=10), we can enumerate all 2^n assignments
    # But 2^10 = 1024, which is fine
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            # Check if u and v are in different partitions
            bit_u = (mask >> u) & 1
            bit_v = (mask >> v) & 1
            if bit_u != bit_v:
                cut += 1
        if cut > best:
            best = cut
    return best


def _expected_cut(n, edges, gammas, betas):
    """
    Compute the expected cut value for a given QAOA parameter set.
    Uses the exact quantum simulation for small n.
    """
    # For n <= 10, we can simulate the full quantum state
    # The state is a vector of size 2^n
    dim = 1 << n
    
    # Start in |+>^n = (1/sqrt(2^n)) * [1, 1, ..., 1]
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Build the cost Hamiltonian C = sum_{(u,v) in edges} (I - Z_u Z_v)/2
    # Actually, for MaxCut, C = sum_{(u,v)} (1 - Z_u Z_v)/2
    # exp(-i gamma C) = exp(-i gamma/2 * sum (1 - Z_u Z_v))
    # = exp(-i gamma/2 * |E|) * exp(-i gamma/2 * sum (1 - Z_u Z_v) - |E|)
    # Let's just compute the phase for each basis state
    
    # For each basis state |x>, the eigenvalue of C is the number of edges cut by x
    # So exp(-i gamma C) |x> = exp(-i gamma * cut(x)) |x>
    # Wait, C = sum_{(u,v)} (1 - Z_u Z_v)/2
    # The eigenvalue of Z_u Z_v on |x> is (+1) if x_u == x_v, (-1) if x_u != x_v
    # So (1 - Z_u Z_v)/2 has eigenvalue 0 if same, 1 if different
    # So C|x> = cut(x) |x>
    # Thus exp(-i gamma C)|x> = exp(-i gamma * cut(x)) |x>
    
    # Precompute cut(x) for all x
    cuts = np.zeros(dim, dtype=np.float64)
    for x in range(dim):
        c = 0
        for u, v in edges:
            bit_u = (x >> u) & 1
            bit_v = (x >> v) & 1
            if bit_u != bit_v:
                c += 1
        cuts[x] = c
    
    # Apply cost layer: multiply each amplitude by exp(-i gamma * cut(x))
    for gamma in gammas:
        phases = np.exp(-1j * gamma * cuts)
        state *= phases
    
    # Apply mixer layer: exp(-i beta sum X)
    # sum X = X_1 + X_2 + ... + X_n
    # exp(-i beta sum X) = prod_k exp(-i beta X_k)
    # Each exp(-i beta X_k) = cos(beta) I - i sin(beta) X_k
    # This is a product of single-qubit rotations
    
    for beta in betas:
        # Apply exp(-i beta sum X) to the state
        # This is equivalent to applying R_x(-2*beta) on each qubit? No.
        # exp(-i beta X) = cos(beta) I - i sin(beta) X
        # In the computational basis, this flips the sign/phase of amplitudes
        # Actually, let's think carefully.
        # For a single qubit: exp(-i beta X) |0> = cos(beta)|0> - i sin(beta)|1>
        # exp(-i beta X) |1> = cos(beta)|1> - i sin(beta)|0>
        # So it's a rotation around the X axis by angle 2*beta? 
        # R_x(theta) = exp(-i theta X/2) = cos(theta/2) I - i sin(theta/2) X
        # So exp(-i beta X) = R_x(2*beta)
        
        # For the full n-qubit system, exp(-i beta sum X) = prod_k exp(-i beta X_k)
        # We can apply this by iterating over each qubit
        
        # For efficiency with small n, we can apply the transformation directly
        # For each qubit k, the transformation on the state is:
        # new_state[x] = cos(beta) * state[x] - i sin(beta) * state[x_flip_k]
        # where x_flip_k is x with bit k flipped
        
        new_state = np.zeros_like(state)
        for k in range(n):
            # For each basis state x, the contribution from state[x] and state[x_flip_k]
            pass
        
        # More efficient: for each pair (x, x_flip_k), apply the 2x2 rotation
        # Actually, since all qubits get the same rotation, we can do it in one pass
        # for each qubit, but we need to be careful about double counting.
        
        # Let's do it qubit by qubit
        for k in range(n):
            new_state = np.zeros_like(state)
            for x in range(dim):
                x_flip = x ^ (1 << k)
                # The amplitude at x in the new state gets contributions from:
                # - state[x] with coefficient cos(beta)
                # - state[x_flip] with coefficient -i sin(beta)
                # But we need to sum over all qubits, not just k.
                # This approach is wrong because the rotations on different qubits don't commute in a simple way for the full state.
            
            # Actually, exp(-i beta sum X) = exp(-i beta X_1) exp(-i beta X_2) ... exp(-i beta X_n)
            # since [X_i, X_j] = 0 for i != j
            # So we can apply them sequentially
            
            # Apply exp(-i beta X_k) to the state
            # For each basis state x, the new amplitude is:
            # cos(beta) * state[x] - i sin(beta) * state[x ^ (1<<k)]
            # But this is for a single qubit k. We apply this for all k.
            
            # Let's do it properly: start with the current state, and for each qubit,
            # apply the single-qubit rotation
            pass
        
        # Let me redo this more carefully
        # The operator exp(-i beta sum_{k} X_k) acts on the full Hilbert space.
        # Since X_k and X_l commute for k != l, we have:
        # exp(-i beta sum_k X_k) = prod_k exp(-i beta X_k)
        
        # For a single qubit k, exp(-i beta X_k) acts as:
        # |0> -> cos(beta)|0> - i sin(beta)|1>
        # |1> -> cos(beta)|1> - i sin(beta)|0>
        
        # So for the full state, applying exp(-i beta X_k) transforms:
        # state[x] -> cos(beta) * state[x] - i sin(beta) * state[x ^ (1<<k)]
        
        # We apply this for k = 0, 1, ..., n-1 sequentially.
        
        temp_state = state.copy()
        for k in range(n):
            new_state = np.zeros_like(temp_state)
            for x in range(dim):
                x_flip = x ^ (1 << k)
                new_state[x] = (cos_beta * temp_state[x] - 1j * sin_beta * temp_state[x_flip])
            temp_state = new_state
        state = temp_state
    
    # Now compute the expected cut
    # <cut> = sum_x |state[x]|^2 * cut(x)
    probs = np.abs(state) ** 2
    expected_cut_val = np.sum(probs * cuts)
    
    return expected_cut_val


def _optimize_angles(n, edges, p):
    """
    Optimize QAOA parameters to maximize the expected cut.
    Uses scipy optimization.
    """
    from scipy.optimize import minimize
    
    # Initial guess
    # Use a heuristic initial guess based on the graph structure
    # For p=2, we have 4 parameters: gamma_1, beta_1, gamma_2, beta_2
    
    # Let's try a few initial points and pick the best
    best_params = None
    best_value = -np.inf
    
    # Try several initial points
    initial_points = [
        [0.35, 0.25, 0.35, 0.25],  # Original
        [0.5, 0.5, 0.3, 0.2],
        [0.2, 0.3, 0.4, 0.3],
        [0.1, 0.1, 0.5, 0.5],
        [0.4, 0.4, 0.2, 0.2],
    ]
    
    for init in initial_points:
        x0 = np.array(init)
        
        def objective(params):
            gammas = params[:p]
            betas = params[p:2*p]
            val = _expected_cut(n, edges, gammas, betas)
            return -val  # Negate for minimization
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if -result.fun > best_value:
                best_value = -result.fun
                best_params = result.x
        except:
            pass
    
    if best_params is None:
        # Fallback to the original
        return np.tile([0.35, 0.25], (p, 1))
    
    gammas = best_params[:p]
    betas = best_params[p:2*p]
    
    # Format as (p, 2) array with rows (gamma, beta)
    angles_out = np.column_stack((gammas, betas))
    return angles_out


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio of the 
    independently simulated expected cut to the exact classical optimum.
    
    Args:
        n: number of nodes
        edges: list of edges (u, v)
        p: number of QAOA layers
    
    Returns:
        numpy array of shape (p, 2) with rows (gamma, beta) in radians
    """
    if p != 2:
        # Fallback for other p values
        return np.tile([0.35, 0.25], (p, 1))
    
    # Optimize angles for this specific graph
    angles_out = _optimize_angles(n, edges, p)
    
    # Ensure the output is a valid numpy array of shape (p, 2)
    angles_out = np.asarray(angles_out, dtype=np.float64)
    if angles_out.shape != (p, 2):
        angles_out = np.tile([0.35, 0.25], (p, 1))
    
    return angles_out
# EVOLVE-BLOCK-END
