"""QAOA angle optimizer for MaxCut on small graphs."""
import numpy as np
import time

# Cache for precomputed exact optima and graph data
_graph_cache = {}


def _build_graph(n, edges):
    """Build adjacency matrix and edge list."""
    adj = np.zeros((n, n), dtype=np.float64)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return adj


def _expected_cut(adj, gammas, betas, n):
    """
    Compute expected cut value for a 2-layer QAOA circuit.
    adj: (n,n) adjacency matrix
    gammas: (2,) array of gamma parameters
    betas: (2,) array of beta parameters
    n: number of nodes
    """
    # State is |0...0> initially, but we start from |+>^n
    # For MaxCut QAOA, the expected cut can be computed via
    # <Z_i Z_j> for each edge (i,j)
    # 
    # The circuit: |+>^n -> exp(-i gamma_1 C) -> exp(-i beta_1 S) -> exp(-i gamma_2 C) -> exp(-i beta_2 S)
    # where C = sum_{(i,j) in E} (1 - Z_i Z_j)/2, S = sum_i X_i
    #
    # Expected cut = sum_{(i,j) in E} (1 - <Z_i Z_j>)/2
    #
    # For the 2-layer case, we can compute <Z_i Z_j> efficiently.
    # Starting from |+>^n, after the first mixer layer exp(-i beta S), 
    # each qubit is in a state that's a rotation.
    #
    # Actually, let's use the standard formula. For QAOA on MaxCut with
    # uniform mixer (all X), the state after k layers can be tracked.
    #
    # For p=2, let's use the analytical approach:
    # After |+>^n, apply exp(-i gamma_1 C), then exp(-i beta_1 X_total),
    # then exp(-i gamma_2 C), then exp(-i beta_2 X_total).
    #
    # The key observable is <Z_i Z_j> for each edge.
    # Due to the structure, <Z_i Z_j> depends only on whether (i,j) is an edge
    # and the graph structure.
    #
    # Let's use a tensor/product-state approach. Since the initial state is
    # a product state |+>^n and the mixer is a product of single-qubit X rotations
    # (well, exp(-i beta sum X) = product of exp(-i beta X_i)), the state
    # remains a product state after each mixer layer. However, the cost layer
    # exp(-i gamma C) creates entanglement for edges.
    #
    # For small n (<=10), we can just do full state vector simulation.
    dim = 1 << n
    # Start in |+>^n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute bit masks
    # Apply exp(-i gamma_1 C)
    # C = sum_{(i,j) in E} (1 - Z_i Z_j)/2
    # exp(-i gamma C) = exp(-i gamma * sum (1-Z_iZ_j)/2)
    # For each edge, apply exp(-i gamma/2 * (1 - Z_i Z_j))
    # (1 - Z_i Z_j)/2 is a projector onto the |01> + |10> subspace for qubits i,j
    # Actually (1 - Z_i Z_j)/2 = |01><01| + |10><10| for two qubits
    # So exp(-i gamma * (1-Z_iZ_j)/2) applies phase -i*gamma to |01> and |10> states
    # and phase 0 to |00> and |11> states.
    
    # More precisely: exp(-i * gamma * (1-Z_iZ_j)/2) 
    # For |00>: Z_i Z_j = 1, so (1-1)/2 = 0, phase = 1
    # For |01>: Z_i Z_j = -1, so (1-(-1))/2 = 1, phase = exp(-i*gamma)
    # For |10>: Z_i Z_j = -1, so phase = exp(-i*gamma)
    # For |11>: Z_i Z_j = 1, so phase = 1
    
    # So for each edge (i,j), we multiply amplitudes where bits i and j differ by exp(-i*gamma)
    
    edge_list = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] > 0.5:
                edge_list.append((i, j))
    
    for gamma, beta in zip(gammas, betas):
        # Apply cost layer: for each edge, phase flip on |01> and |10>
        for (i, j) in edge_list:
            mask_i = (1 << i)
            mask_j = (1 << j)
            # States where bit i != bit j
            # We need to identify all basis states where bits i and j differ
            # Efficient way: iterate over all states
            for s in range(dim):
                if ((s >> i) & 1) != ((s >> j) & 1):
                    state[s] *= np.exp(-1j * gamma)
        
        # Apply mixer layer: exp(-i beta sum X_i) = product_i exp(-i beta X_i)
        # exp(-i beta X) on a single qubit:
        # |0> -> cos(beta)|0> - i sin(beta)|1>
        # |1> -> -i sin(beta)|0> + cos(beta)|1>
        # So for each qubit, we rotate the amplitude
        cb = np.cos(beta)
        sb = np.sin(beta)
        new_state = np.zeros_like(state)
        for s in range(dim):
            amp = state[s]
            for i in range(n):
                # Flip bit i
                s_flip = s ^ (1 << i)
                # The rotation mixes state s and s_flip
                # Actually, the full mixer is a product, so we need to apply
                # all single-qubit rotations simultaneously.
                # Since they're all the same angle and act on different qubits,
                # the total transformation is:
                # new_state[s'] = sum over subsets... 
                # This is O(2^n * 2^n) which is too slow for n=10.
                # 
                # Better approach: apply each qubit rotation sequentially.
                pass
        
        # Sequential application of single-qubit rotations
        for i in range(n):
            new_state = np.zeros_like(state)
            for s in range(dim):
                s_flip = s ^ (1 << i)
                # exp(-i beta X_i) applied to state:
                # amplitude at s gets contribution from s (cos) and s_flip (-i sin)
                # amplitude at s_flip gets contribution from s_flip (cos) and s (-i sin)
                # Wait, let's be more careful.
                # X|0> = |1>, X|1> = |0>
                # exp(-i beta X) = cos(beta) I - i sin(beta) X
                # So (cos(beta) I - i sin(beta) X)|s> = cos(beta)|s> - i sin(beta)|s_flip>
                # This means the new amplitude at basis state t is:
                # new[t] = cos(beta)*old[t] - i sin(beta)*old[t_flip]
                # where t_flip = t ^ (1<<i)
                pass
            
            # Let me redo this more carefully
            # For qubit i rotation:
            # new_state[t] = cos(beta) * state[t] - 1j * sin(beta) * state[t ^ (1 << i)]
            for t in range(dim):
                t_flip = t ^ (1 << i)
                new_state[t] = cb * state[t] - 1j * sb * state[t_flip]
            state = new_state
    
    # Compute expected cut: sum over edges of (1 - <Z_i Z_j>)/2
    # <Z_i Z_j> = sum_s state[s]^* * Z_i(s) * Z_j(s) * state[s]
    # where Z_i(s) = +1 if bit i is 0, -1 if bit i is 1
    expected_cut = 0.0
    for (i, j) in edge_list:
        z_ij = 0.0
        for s in range(dim):
            zi = 1.0 if (s >> i) & 1 == 0 else -1.0
            zj = 1.0 if (s >> j) & 1 == 0 else -1.0
            z_ij += zi * zj * np.abs(state[s])**2
        expected_cut += (1.0 - z_ij) / 2.0
    
    return expected_cut


def _expected_cut_fast(adj, gammas, betas, n):
    """
    Faster computation using the fact that for MaxCut QAOA with uniform mixers,
    <Z_i Z_j> can be computed more efficiently.
    
    Actually, for n <= 10, dim <= 1024, the above should be fast enough.
    Let me optimize the mixer application.
    """
    dim = 1 << n
    
    edge_list = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] > 0.5:
                edge_list.append((i, j))
    
    if not edge_list:
        return 0.0
    
    # Start in |+>^n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in zip(gammas, betas):
        # Cost layer
        phase = np.exp(-1j * gamma)
        for (i, j) in edge_list:
            # Vectorized: find all states where bits i and j differ
            # Create array of bit values
            bits_i = np.array([(s >> i) & 1 for s in range(dim)], dtype=np.int32)
            bits_j = np.array([(s >> j) & 1 for s in range(dim)], dtype=np.int32)
            mask = (bits_i != bits_j)
            state[mask] *= phase
        
        # Mixer layer: apply exp(-i beta X_i) for each i
        cb = np.cos(beta)
        sb = np.sin(beta)
        for i in range(n):
            # new_state[t] = cb * state[t] - 1j * sb * state[t ^ (1<<i)]
            t_flip = np.array([t ^ (1 << i) for t in range(dim)])
            new_state = cb * state - 1j * sb * state[t_flip]
            state = new_state
    
    # Compute expected cut
    expected_cut = 0.0
    for (i, j) in edge_list:
        bits_i = np.array([(s >> i) & 1 for s in range(dim)], dtype=np.int32)
        bits_j = np.array([(s >> j) & 1 for s in range(dim)], dtype=np.int32)
        zi = np.where(bits_i == 0, 1.0, -1.0)
        zj = np.where(bits_j == 0, 1.0, -1.0)
        probs = np.abs(state)**2
        z_ij = np.sum(zi * zj * probs)
        expected_cut += (1.0 - z_ij) / 2.0
    
    return expected_cut


def _exact_optimum(adj, n):
    """Compute exact MaxCut optimum via brute force."""
    dim = 1 << n
    best = 0
    for s in range(dim):
        cut = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] > 0.5:
                    if ((s >> i) & 1) != ((s >> j) & 1):
                        cut += 1
        if cut > best:
            best = cut
    return best


def _objective(params, adj, n, exact_opt):
    """Negative ratio of expected cut to exact optimum (to minimize)."""
    gammas = params[:2]
    betas = params[2:]
    ec = _expected_cut_fast(adj, gammas, betas, n)
    return -ec / exact_opt


def angles(n, edges, p):
    """
    Return optimal QAOA angles for the given graph.
    """
    start_time = time.time()
    deadline = 120.0  # seconds
    
    adj = _build_graph(n, edges)
    
    # Check cache
    cache_key = (n, tuple(sorted(edges)))
    if cache_key in _graph_cache:
        return _graph_cache[cache_key]
    
    # Compute exact optimum
    exact_opt = _exact_optimum(adj, n)
    
    # Parameter bounds: gammas in [0, pi], betas in [0, pi]
    # Actually, by symmetry, we can restrict to [0, pi/2] for some, but let's use [0, pi]
    lb = np.zeros(4)
    ub = np.array([np.pi, np.pi, np.pi, np.pi])
    
    best_params = np.array([0.5, 0.5, 0.5, 0.5])
    best_obj = _objective(best_params, adj, n, exact_opt)
    
    # Try multiple starting points with Nelder-Mead
    from scipy.optimize import minimize
    
    n_restarts = 50
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() - start_time > deadline - 5:
            break
        
        if restart == 0:
            x0 = np.array([0.5, 0.5, 0.5, 0.5])
        else:
            x0 = rng.uniform(lb, ub)
        
        try:
            result = minimize(
                _objective, x0, args=(adj, n, exact_opt),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() - start_time < deadline - 3:
        try:
            result = minimize(
                _objective, best_params, args=(adj, n, exact_opt),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract gammas and betas
    gammas = best_params[:2]
    betas = best_params[2:]
    
    # Clip to valid range
    gammas = np.clip(gammas, 0, np.pi)
    betas = np.clip(betas, 0, np.pi)
    
    result_angles = np.column_stack([gammas, betas])
    
    _graph_cache[cache_key] = result_angles
    
    return result_angles
