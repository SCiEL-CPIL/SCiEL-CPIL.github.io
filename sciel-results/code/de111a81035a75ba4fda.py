"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 115.0  # safety margin before 120s limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _exact_maxcut(n, edges):
    """Compute exact maximum cut by brute force."""
    if not edges:
        return 0
    best = 0
    # For n up to 10, 2^n is at most 1024
    for mask in range(1 << n):
        cut = 0
        for (u, v) in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best


def _expected_cut(n, edges, gammas, betas):
    """
    Compute expected cut for QAOA with given angles.
    Uses state vector simulation for n <= 10.
    """
    num_states = 1 << n
    # Initialize state vector: |+>^n = (1/sqrt(2^n)) * ones
    psi = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    for layer in range(len(gammas)):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply cost gate: exp(-i * gamma * C)
        # C = sum_{(u,v) in edges} (1 - Z_u Z_v)/2 = |E|/2 - 1/2 * sum Z_u Z_v
        # exp(-i*gamma*C) = exp(-i*gamma*|E|/2) * exp(i*gamma/2 * sum Z_u Z_v)
        # The global phase doesn't matter, so we apply exp(i*gamma/2 * sum Z_u Z_v)
        # Z_u Z_v is diagonal: +1 if same bit, -1 if different
        phase_cost = np.ones(num_states, dtype=np.complex128)
        for (u, v) in edges:
            # Z_u Z_v = +1 if bits equal, -1 if different
            # For each state, compute product
            for s in range(num_states):
                bu = (s >> u) & 1
                bv = (s >> v) & 1
                zz = 1.0 if bu == bv else -1.0
                phase_cost[s] *= np.exp(1j * gamma * 0.5 * zz)
        psi = psi * phase_cost
        
        # Apply mixer gate: exp(-i * beta * sum X_j)
        # X_j flips bit j. The mixer is a product of single-qubit rotations.
        # exp(-i*beta*X_j) = cos(beta)*I - i*sin(beta)*X_j
        # We can apply this by permuting and combining
        # For efficiency, apply each X rotation sequentially
        for j in range(n):
            # Apply cos(beta)*I - i*sin(beta)*X_j
            # X_j maps state s to s XOR (1<<j)
            new_psi = np.zeros_like(psi)
            for s in range(num_states):
                s_flipped = s ^ (1 << j)
                new_psi[s] = np.cos(beta) * psi[s] - 1j * np.sin(beta) * psi[s_flipped]
            psi = new_psi
    
    # Compute expected cut
    # <C> = sum_{(u,v) in edges} (1 - <Z_u Z_v>)/2
    # <Z_u Z_v> = sum_s |psi_s|^2 * Z_u(s) * Z_v(s)
    probs = np.abs(psi) ** 2
    expected_cut = 0.0
    for (u, v) in edges:
        zz_expect = 0.0
        for s in range(num_states):
            bu = (s >> u) & 1
            bv = (s >> v) & 1
            zz = 1.0 if bu == bv else -1.0
            zz_expect += probs[s] * zz
        expected_cut += (1.0 - zz_expect) / 2.0
    
    return expected_cut


def _expected_cut_fast(n, edges, gammas, betas):
    """
    Faster expected cut computation using vectorized operations.
    """
    num_states = 1 << n
    psi = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    # Precompute bit values for all states
    states = np.arange(num_states, dtype=np.int32)
    
    for layer in range(len(gammas)):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply cost gate: multiply by exp(i*gamma/2 * sum Z_u Z_v)
        phase_cost = np.ones(num_states, dtype=np.float64)
        for (u, v) in edges:
            # Compute Z_u Z_v for all states vectorized
            bits_u = (states >> u) & 1
            bits_v = (states >> v) & 1
            zz = np.where(bits_u == bits_v, 1.0, -1.0)
            phase_cost *= np.exp(gamma * 0.5 * zz)
        psi = psi * phase_cost
        
        # Apply mixer: exp(-i*beta*sum X_j)
        # Apply each X rotation
        for j in range(n):
            mask = 1 << j
            # Create index arrays
            idx_same = np.where((states & mask) == 0)[0]
            idx_flip = idx_same | mask
            # psi[s] -> cos(beta)*psi[s] - i*sin(beta)*psi[s^mask]
            new_psi = np.zeros_like(psi)
            for s in idx_same:
                sf = s | mask
                new_psi[s] = np.cos(beta) * psi[s] - 1j * np.sin(beta) * psi[sf]
                new_psi[sf] = np.cos(beta) * psi[sf] - 1j * np.sin(beta) * psi[s]
            psi = new_psi
    
    # Compute expected cut
    probs = np.abs(psi) ** 2
    expected_cut = 0.0
    for (u, v) in edges:
        bits_u = (states >> u) & 1
        bits_v = (states >> v) & 1
        zz = np.where(bits_u == bits_v, 1.0, -1.0)
        zz_expect = np.sum(probs * zz)
        expected_cut += (1.0 - zz_expect) / 2.0
    
    return expected_cut


def _graph_features(n, edges):
    """Compute graph features for parameter selection."""
    num_edges = len(edges)
    density = 2.0 * num_edges / (n * (n - 1)) if n > 1 else 0.0
    
    # Degree distribution
    degrees = np.zeros(n, dtype=np.int32)
    for (u, v) in edges:
        degrees[u] += 1
        degrees[v] += 1
    
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees) if n > 0 else 0
    degree_var = np.var(degrees)
    
    # Connectivity: fraction of nodes with degree > 0
    connected_frac = np.sum(degrees > 0) / n if n > 0 else 0.0
    
    return {
        'n': n,
        'num_edges': num_edges,
        'density': density,
        'avg_degree': avg_degree,
        'max_degree': max_degree,
        'degree_var': degree_var,
        'connected_frac': connected_frac,
    }


def _select_angles(n, edges, p, features=None):
    """
    Select QAOA angles based on graph features.
    Uses a combination of heuristics and local optimization.
    """
    if features is None:
        features = _graph_features(n, edges)
    
    density = features['density']
    avg_degree = features['avg_degree']
    
    # Initial angle guesses based on graph properties
    # For sparse graphs, smaller angles tend to work better
    # For dense graphs, larger angles
    gamma0 = 0.2 + 0.3 * density
    beta0 = 0.2 + 0.2 * density
    
    # Try a few candidate angle sets and pick the best
    candidates = [
        [gamma0, gamma0 * 0.8],
        [0.3, 0.2],
        [0.4, 0.15],
        [0.25, 0.3],
        [0.35, 0.25],
        [0.5, 0.1],
        [0.15, 0.4],
    ]
    
    # Also try angles scaled by 1/sqrt(avg_degree)
    if avg_degree > 0:
        scale = 1.0 / np.sqrt(avg_degree)
        candidates.append([gamma0 * scale, beta0 * scale])
        candidates.append([0.5 * scale, 0.3 * scale])
    
    best_angles = None
    best_val = -1.0
    
    # Evaluate candidates
    for cand in candidates:
        gammas = cand[:p] + [cand[-1]] * max(0, p - len(cand))
        betas = [0.2 + 0.1 * i for i in range(p)]
        # Adjust betas based on density
        betas = [b * (0.8 + 0.4 * density) for b in betas]
        
        try:
            val = _expected_cut_fast(n, edges, gammas, betas)
            if val > best_val:
                best_val = val
                best_angles = (gammas, betas)
        except Exception:
            continue
    
    # Do a local refinement if time permits
    if best_angles is not None and _time_remaining() > 30:
        gammas, betas = best_angles
        # Simple hill climbing
        for _ in range(20):
            if _time_remaining() < 30:
                break
            improved = False
            for i in range(p):
                for delta in [-0.05, 0.05]:
                    new_gammas = list(gammas)
                    new_gammas[i] += delta
                    val = _expected_cut_fast(n, edges, new_gammas, betas)
                    if val > best_val + 1e-6:
                        gammas = new_gammas
                        best_val = val
                        improved = True
                    new_betas = list(betas)
                    new_betas[i] += delta
                    val = _expected_cut_fast(n, edges, gammas, new_betas)
                    if val > best_val + 1e-6:
                        betas = new_betas
                        best_val = val
                        improved = True
            if not improved:
                break
    
    if best_angles is None:
        # Fallback
        return np.tile([0.35, 0.25], (p, 1))
    
    gammas, betas = best_angles
    return np.column_stack([gammas, betas])


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of nodes
        edges: list of (u, v) tuples
        p: number of QAOA layers (always 2 for this task)
    
    Returns:
        np.array of shape (p, 2), rows are (gamma, beta) in radians
    """
    try:
        features = _graph_features(n, edges)
        result = _select_angles(n, edges, p, features)
        # Ensure correct shape and type
        result = np.asarray(result, dtype=np.float64)
        if result.shape != (p, 2):
            result = np.tile([0.35, 0.25], (p, 1))
        return result
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
