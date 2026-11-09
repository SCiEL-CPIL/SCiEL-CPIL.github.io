"""
Strategy: Runtime per-graph optimization using hybrid Nelder-Mead and Gradient Ascent.
Instead of a static lookup table, we optimize the 4 QAOA parameters (gamma1, beta1, gamma2, beta2)
directly for each specific graph instance. We use a multi-start approach with Nelder-Mead
initially to escape shallow local minima, followed by gradient ascent for precise convergence.
This ensures the parameters are tailored to the specific topology of the evaluation graph,
maximizing the cut ratio beyond what generic density-based heuristics can achieve.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 110.0 # Safety margin for 120s limit

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _expected_cut(n, edges, params):
    """
    Calculate the expected cut for given QAOA parameters.
    params: array of shape (2, 2) -> [[gamma1, beta1], [gamma2, beta2]]
    """
    g1, b1 = params[0]
    g2, b2 = params[1]
    
    # Build adjacency matrix or edge list representation for efficiency
    # For n <= 10, a dense matrix is fast enough, but we can also use edge list.
    # Let's use a helper to compute probabilities.
    
    # State evolution:
    # |psi> = |+>^n
    # Apply exp(-i g1 C)
    # Apply exp(-i b1 sum X)
    # Apply exp(-i g2 C)
    # Apply exp(-i b2 sum X)
    
    # The expected cut is sum_{(i,j) in E} 0.5 * (1 - <Z_i Z_j>)
    # <Z_i Z_j> can be computed from the final state.
    
    # For small n (<=10), we can represent the state vector explicitly.
    # Dimension is 2^n. For n=10, 1024 complex numbers.
    # Operations are rotations.
    
    # However, simulating the full state for every step in an optimization loop
    # might be slow if not vectorized. Let's use the property that
    # <Z_i Z_j> depends on the product of cosines/sines of the angles 
    # applied to the relevant qubits, but interactions make it complex.
    
    # Actually, for p=2, there is a known analytic form for the expected cut
    # in terms of the angles and the graph structure, but deriving it is complex.
    # Explicit state simulation is safer and fast enough for n<=10.
    
    dim = 1 << n
    # Initialize state to |+>^n
    # |+> = (|0> + |1>)/sqrt(2)
    # |psi> = tensor product of |+>
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Helper to apply Z_i Z_j phase
    # exp(-i g Z_i Z_j) acts on basis states |x> by multiplying by exp(-i g (-1)^{x_i x_j})
    # This is equivalent to: if x_i == x_j, phase 1; else phase exp(-i g).
    # Wait, Z_i Z_j |x> = (-1)^{x_i + x_j} |x>? No.
    # Z |0> = |0>, Z |1> = -|1>.
    # Z_i Z_j |x> = (-1)^{x_i} (-1)^{x_j} |x> = (-1)^{x_i + x_j} |x>.
    # So the phase is exp(-i g (-1)^{x_i + x_j}).
    # If x_i == x_j, sum is even -> phase 1.
    # If x_i != x_j, sum is odd -> phase exp(-i g).
    
    # Helper to apply X rotation on all qubits: exp(-i b sum X)
    # This is a tensor product of single-qubit rotations:
    # exp(-i b X) = [[cos(b), -i sin(b)], [-i sin(b), cos(b)]]
    # We can apply this to each qubit.
    
    # Efficient simulation for n <= 10:
    # We can iterate over the 2^n basis states? No, that's 1024 states.
    # Applying unitaries to a vector of size 1024 is very fast in NumPy.
    
    # Let's precompute the masks for Z_i Z_j and X rotations?
    # Actually, applying X rotation to all qubits is a Kronecker product.
    # U_x = exp(-i b X)
    # U_total = U_x ⊗ U_x ⊗ ... ⊗ U_x
    # This can be applied efficiently.
    
    # Let's implement a fast simulator.
    
    # 1. Apply C phase
    # For each edge (i, j), multiply psi[x] by exp(-i g) if x_i != x_j else 1.
    # We can do this by iterating over edges.
    
    # Precompute bit masks for each qubit
    # mask[i] = 1 << i
    
    # Step 1: C phase g1
    for i, j in edges:
        # Create a mask for states where bit i != bit j
        # This is tricky to vectorize simply without precomputed tables.
        # For n=10, 1024 states. Looping 100 edges * 1024 states = 10k ops.
        # Doing this 2 times (g1, g2) is 20k ops. Very fast.
        
        # We can use numpy bitwise operations on the range of indices.
        # But we are modifying psi in place.
        pass
    
    # A better way: Precompute the "cut" phase for each edge?
    # Let's just loop over the basis states for the phase part.
    # It's n=10, so 1024 states.
    # Edges ~ 50.
    # 50 * 1024 = 51,200 operations per C layer.
    # 2 C layers = 100k ops.
    # X layers: 
    # Applying U_x to each qubit.
    # U_x is 2x2.
    # We can reshape psi into (2, 2, ..., 2) and apply rotation along each axis.
    
    # Let's implement this explicitly.
    
    # Reshape psi to (2,)*n
    # psi_reshaped = psi.reshape([2]*n)
    
    # Apply X rotation on qubit k:
    # new_psi[:, ..., 0, ..., :] = cos(b)*old_psi[:, ..., 0, ..., :] - i*sin(b)*old_psi[:, ..., 1, ..., :]
    # new_psi[:, ..., 1, ..., :] = -i*sin(b)*old_psi[:, ..., 0, ..., :] + cos(b)*old_psi[:, ..., 1, ..., :]
    
    # This is efficient.
    
    # Let's write the function.
    
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Apply C(g1)
    # We need to apply phase to each edge.
    # To avoid looping over edges in Python (slow), we can vectorize if possible.
    # But edges are arbitrary.
    # For n=10, python loop over ~50 edges is fine.
    
    # Precompute indices for each edge?
    # For edge (i,j), the indices where bit i != bit j.
    # Let's just loop over the range of dim and check? No, that's 1024 * 50 = 50k checks.
    # Better:
    # For a given edge (i,j), the pattern of bits is periodic.
    # We can construct a mask array of length dim.
    
    # Since this is called many times during optimization, we should precompute
    # the masks for the specific graph.
    # But the function signature is angles(n, edges, p).
    # We can cache the masks based on (n, tuple(sorted(edges))).
    
    pass # Placeholder for logic inside the function below

    return 0.0 # Dummy

def _simulate_cut(n, edges, g1, b1, g2, b2, masks=None):
    """
    Fast simulation of expected cut for p=2 QAOA.
    masks: dict mapping (i,j) to a boolean array of length 2^n indicating where phase applies.
    """
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Apply C(g1)
    # psi[x] *= exp(-i g1) if x_i != x_j else 1
    # This is equivalent to psi[x] *= cos(g1) - i sin(g1) * (-1)^{x_i x_j + 1}?
    # If x_i != x_j, phase is exp(-i g).
    # If x_i == x_j, phase is 1.
    # Let mask be 1 if x_i != x_j.
    # psi *= 1 + mask * (exp(-i g) - 1)
    
    phase_factor = np.exp(-1j * g1) - 1.0
    
    if masks is not None:
        for key in masks.keys():
            mask = masks[key]
            # mask is bool array
            psi *= (1.0 + mask * phase_factor)
    else:
        # Fallback (slower)
        for i, j in edges:
            # Generate mask
            mask = np.zeros(dim, dtype=bool)
            for x in range(dim):
                bi = (x >> i) & 1
                bj = (x >> j) & 1
                if bi != bj:
                    mask[x] = True
            psi *= (1.0 + mask * phase_factor)
            
    # Apply X(b1)
    # Reshape to (2,)*n
    shape = [2] * n
    psi = psi.reshape(shape)
    
    # Apply rotation to each qubit
    c = np.cos(b1)
    s = -1j * np.sin(b1)
    
    for q in range(n):
        # Move axis q to the front
        psi = np.moveaxis(psi, q, 0)
        # psi is now (2, 2, ..., 2)
        # Apply 2x2 rotation on the first axis
        old_psi = psi.copy()
        psi[0, ...] = c * old_psi[0, ...] + s * old_psi[1, ...]
        psi[1, ...] = s * old_psi[0, ...] + c * old_psi[1, ...]
        # Move axis back
        psi = np.moveaxis(psi, 0, q)
        
    psi = psi.reshape(dim)
    
    # Apply C(g2)
    phase_factor2 = np.exp(-1j * g2) - 1.0
    if masks is not None:
        for key in masks.keys():
            mask = masks[key]
            psi *= (1.0 + mask * phase_factor2)
    else:
        for i, j in edges:
            mask = np.zeros(dim, dtype=bool)
            for x in range(dim):
                bi = (x >> i) & 1
                bj = (x >> j) & 1
                if bi != bj:
                    mask[x] = True
            psi *= (1.0 + mask * phase_factor2)

    # Apply X(b2)
    psi = psi.reshape(shape)
    c = np.cos(b2)
    s = -1j * np.sin(b2)
    for q in range(n):
        psi = np.moveaxis(psi, q, 0)
        old_psi = psi.copy()
        psi[0, ...] = c * old_psi[0, ...] + s * old_psi[1, ...]
        psi[1, ...] = s * old_psi[0, ...] + c * old_psi[1, ...]
        psi = np.moveaxis(psi, 0, q)
        
    psi = psi.reshape(dim)
    
    # Calculate Expected Cut
    # E[Cut] = sum_{(i,j) in E} 0.5 * (1 - <Z_i Z_j>)
    # <Z_i Z_j> = sum_x |psi[x]|^2 * (-1)^{x_i + x_j}
    
    cut_val = 0.0
    probs = np.abs(psi)**2
    
    if masks is not None:
        # We need <Z_i Z_j> for each edge.
        # We can compute the expectation value for each edge.
        # This requires summing probs[x] * sign[x]
        # We can precompute the sign arrays for each edge.
        # But we only have masks for "cut" (x_i != x_j).
        # sign[x] = 1 if x_i == x_j, -1 if x_i != x_j.
        # sign[x] = 1 - 2 * mask[x]
        
        for key in masks.keys():
            mask = masks[key]
            # <Z_i Z_j> = sum probs * (1 - 2*mask)
            # = sum probs - 2 * sum(probs * mask)
            # sum probs = 1
            # so <Z_i Z_j> = 1 - 2 * sum(probs * mask)
            exp_val = 1.0 - 2.0 * np.sum(probs * mask)
            cut_val += 0.5 * (1.0 - exp_val)
            # Simplifies to: 0.5 * (1 - (1 - 2*sum)) = 0.5 * 2 * sum = sum(probs * mask)
            # Wait.
            # E[Cut] = sum 0.5 (1 - <ZZ>)
            # <ZZ> = 1 - 2 P(cut)
            # 1 - <ZZ> = 1 - (1 - 2 P(cut)) = 2 P(cut)
            # 0.5 * 2 P(cut) = P(cut)
            # So E[Cut] = sum P(cut for edge)
            # P(cut for edge) = sum(probs * mask)
            
    else:
        for i, j in edges:
            # Compute <Z_i Z_j>
            # sign = 1 if bits equal, -1 if different
            # We can compute this on the fly
            # But this is slow. Let's assume masks are provided or generated once.
            pass
            
    return cut_val

# Cache for masks
_MASK_CACHE = {}

def _get_masks(n, edges):
    key = (n, tuple(sorted(map(tuple, edges))))
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
        
    dim = 1 << n
    masks = {}
    for i, j in edges:
        mask = np.zeros(dim, dtype=bool)
        # Vectorized mask creation
        # We want mask[x] = True if bit i != bit j
        # We can use numpy bitwise ops on the array of indices
        indices = np.arange(dim)
        bits_i = (indices >> i) & 1
        bits_j = (indices >> j) & 1
        mask = (bits_i != bits_j)
        masks[(i,j)] = mask
    _MASK_CACHE[key] = masks
    return masks

def _expected_cut_fast(n, edges, params, masks):
    g1, b1 = params[0]
    g2, b2 = params[1]
    return _simulate_cut(n, edges, g1, b1, g2, b2, masks)

def _exact_maxcut(n, edges):
    """
    Compute exact MaxCut for small n (<=10).
    Brute force over all 2^n assignments.
    """
    dim = 1 << n
    best_cut = 0
    # Vectorized evaluation of all cuts
    # For each assignment x, cut = sum_{(i,j)} [x_i != x_j]
    # We can compute this for all x at once.
    
    indices = np.arange(dim)
    total_cut = np.zeros(dim)
    for i, j in edges:
        bits_i = (indices >> i) & 1
        bits_j = (indices >> j) & 1
        total_cut += (bits_i != bits_j)
        
    return np.max(total_cut)

def _optimize_graph(n, edges):
    """
    Optimize QAOA parameters for a specific graph.
    """
    if _time_left() < 5.0:
        return np.array([[0.35, 0.25], [0.25, 0.35]])
        
    masks = _get_masks(n, edges)
    opt_val = _exact_maxcut(n, edges)
    if opt_val == 0:
        return np.zeros((2, 2))
        
    # Objective: Maximize E[Cut] / Opt_Cut
    # Since Opt_Cut is constant, we maximize E[Cut]
    
    def objective(params):
        # params is flat array of 4
        g1, b1, g2, b2 = params
        # Ensure angles are within reasonable bounds to avoid numerical issues?
        # QAOA angles are periodic. We can constrain to [0, pi] or similar.
        # For now, let the optimizer handle it, but we might want to wrap.
        # Let's just compute.
        return _simulate_cut(n, edges, g1, b1, g2, b2, masks)
        
    # Initial guesses
    # Common good starts:
    # 1. Mean field approx
    # 2. Random
    # 3. Previous best
    
    best_params = np.array([0.35, 0.25, 0.25, 0.35])
    best_val = objective(best_params)
    
    # Multi-start Nelder-Mead
    # Number of starts: depends on time
    # Let's try 5-10 starts.
    
    starts = [
        [0.35, 0.25, 0.25, 0.35],
        [0.5, 0.5, 0.5, 0.5],
        [0.2, 0.3, 0.3, 0.2],
        [0.4, 0.1, 0.1, 0.4],
        [0.1, 0.4, 0.4, 0.1]
    ]
    
    # Add random starts if time permits
    if _time_left() > 20.0:
        rng = np.random.default_rng(42)
        for _ in range(5):
            starts.append(rng.uniform(0, np.pi, 4).tolist())
            
    from scipy.optimize import minimize
    
    for start in starts:
        if _time_left() < 10.0:
            break
            
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 200, 'adaptive': True})
            
            if res.fun > best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            continue
            
    # Polish with Gradient Ascent?
    # Nelder-Mead is usually good enough for 4D.
    # Let's try a few steps of gradient ascent if time permits.
    
    if _time_left() > 15.0:
        # Compute gradient numerically
        def gradient(params):
            grad = np.zeros(4)
            eps = 1e-5
            for i in range(4):
                p_plus = params.copy()
                p_minus = params.copy()
                p_plus[i] += eps
                p_minus[i] -= eps
                grad[i] = (objective(p_plus) - objective(p_minus)) / (2 * eps)
            return grad
            
        # Simple gradient ascent
        lr = 0.01
        for _ in range(50):
            if _time_left() < 5.0: break
            g = gradient(best_params)
            best_params += lr * g
            new_val = objective(best_params)
            if new_val > best_val:
                best_val = new_val
            else:
                # Backtracking
                lr *= 0.5
                best_params -= lr * g # revert roughly? 
                # Better to just break or reduce step
                if lr < 1e-6: break
                
    return best_params.reshape((2, 2))

def angles(n, edges, p):
    """
    Returns p x 2 array of (gamma, beta) angles.
    """
    if p != 2:
        return np.tile([0.35, 0.25], (p, 1))
        
    if n <= 1 or len(edges) == 0:
        return np.zeros((p, 2))
        
    # Check time
    if _time_left() < 2.0:
        # Fallback to static
        density = len(edges) / (n * (n - 1) / 2.0)
        if density < 0.3:
            return np.array([[0.35, 0.25], [0.25, 0.35]])
        elif density < 0.6:
            return np.array([[0.30, 0.20], [0.20, 0.30]])
        else:
            return np.array([[0.25, 0.15], [0.15, 0.25]])
            
    # Optimize
    try:
        params = _optimize_graph(n, edges)
        return params
    except Exception:
        # Fallback
        return np.array([[0.35, 0.25], [0.25, 0.35]])

# EVOLVE-BLOCK-END
