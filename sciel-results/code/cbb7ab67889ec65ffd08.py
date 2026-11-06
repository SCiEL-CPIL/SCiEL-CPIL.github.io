"""Improved program for R(K2,2,2) maximization via simulated annealing."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _build_adj_masks(adj, n):
    """Build bitmask adjacency lists from boolean numpy array."""
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in range(i + 1, n):
            if adj[i, j]:
                mask |= (1 << j)
        masks[i] = mask
    return masks

def _mask_to_adj(masks, n):
    """Convert bitmask list to boolean numpy array."""
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if (masks[i] >> j) & 1:
                adj[i, j] = True
                adj[j, i] = True
    return adj

class K222Detector:
    """
    Detects K2,2,2 subgraphs in a graph using bitmask operations.
    K2,2,2 = complete tripartite with parts of size 2.
    Equivalent to finding 3 disjoint non-edges (a,b),(c,d),(e,f) such that
    all cross-pairs are edges.
    """
    
    def __init__(self, n):
        self.n = n
        self.all_ones = (1 << n) - 1
    
    def has_k222(self, masks):
        """Check if graph with given adjacency masks contains K2,2,2."""
        n = self.n
        if n < 6:
            return False
        
        # Find all non-edges
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (masks[i] >> j) & 1:
                    non_edges.append((i, j))
        
        if len(non_edges) < 3:
            return False
        
        len_ne = len(non_edges)
        
        # For each pair of disjoint non-edges, check if they form part of K2,2,2
        for i1 in range(len_ne):
            a, b = non_edges[i1]
            # Common neighbors of a and b
            common_ab = masks[a] & masks[b]
            common_ab &= ~(1 << a) & ~(1 << b)
            
            for i2 in range(i1 + 1, len_ne):
                c, d = non_edges[i2]
                # Disjoint check
                if c == a or c == b or d == a or d == b:
                    continue
                
                # c and d must be common neighbors of a and b
                if not ((common_ab >> c) & 1) or not ((common_ab >> d) & 1):
                    continue
                
                # Now need e,f in common_ab ∩ N(c) ∩ N(d), with (e,f) a non-edge
                candidates = common_ab & masks[c] & masks[d]
                candidates &= ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
                
                # Check if any pair in candidates is a non-edge
                m = candidates
                while m:
                    lsb = m & -m
                    e = lsb.bit_length() - 1
                    m ^= lsb
                    # For each e in candidates, check if there's f > e in candidates that's a non-edge
                    rest = candidates & ~((1 << (e + 1)) - 1)  # bits above e
                    while rest:
                        lsb2 = rest & -rest
                        f = lsb2.bit_length() - 1
                        rest ^= lsb2
                        if not ((masks[e] >> f) & 1):
                            return True
        
        return False

def _paley_coloring(p):
    """Paley coloring of K_p where p is prime, p ≡ 1 mod 4."""
    if p % 4 != 1:
        return None
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            colors.append(0 if diff in squares else 1)
    return colors

def _colors_to_adj(colors, n):
    """Convert color list to boolean adjacency matrix (color 0 = edge)."""
    adj = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 0:
                adj[i, j] = True
                adj[j, i] = True
            idx += 1
    return adj

def _adj_to_colors(adj, n):
    """Convert adjacency matrix to color list (edge = 0, non-edge = 1)."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if adj[i, j] else 1)
    return colors

def _is_valid(adj, n):
    """Check that neither color class contains K2,2,2."""
    detector = K222Detector(n)
    
    # Color 0 class
    masks_0 = _build_adj_masks(adj, n)
    if detector.has_k222(masks_0):
        return False
    
    # Color 1 class (complement)
    all_ones = (1 << n) - 1
    masks_1 = [0] * n
    for i in range(n):
        masks_1[i] = (~masks_0[i]) & all_ones & ~(1 << i)
    
    if detector.has_k222(masks_1):
        return False
    
    return True

def _simulated_annealing(adj, n, deadline, max_steps=200000):
    """
    Simulated annealing to find valid coloring.
    Objective: minimize violations (0 = valid, 1+ = invalid).
    """
    detector = K222Detector(n)
    masks_0 = _build_adj_masks(adj, n)
    all_ones = (1 << n) - 1
    
    def compute_violations(masks):
        v0 = 1 if detector.has_k222(masks) else 0
        masks_comp = [0] * n
        for i in range(n):
            masks_comp[i] = (~masks[i]) & all_ones & ~(1 << i)
        v1 = 1 if detector.has_k222(masks_comp) else 0
        return v0 + v1
    
    current_viol = compute_violations(masks_0)
    if current_viol == 0:
        return _mask_to_adj(masks_0, n)
    
    best_adj = _mask_to_adj(masks_0, n)
    best_viol = current_viol
    
    # Annealing schedule
    T_start = 1.0
    T_end = 0.01
    alpha = 0.99995
    
    T = T_start
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Pick random edge to flip
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        # Flip edge (i, j) in masks_0
        bit_i_j = 1 << j
        bit_j_i = 1 << i
        
        if masks_0[i] & bit_i_j:
            # Edge exists, remove it
            masks_0[i] &= ~bit_i_j
            masks_0[j] &= ~bit_j_i
        else:
            # No edge, add it
            masks_0[i] |= bit_i_j
            masks_0[j] |= bit_j_i
        
        new_viol = compute_violations(masks_0)
        
        # Accept if better, or with probability exp(-delta/T)
        delta = new_viol - current_viol
        if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            current_viol = new_viol
            if current_viol < best_viol:
                best_viol = current_viol
                best_adj = _mask_to_adj(masks_0, n)
                if best_viol == 0:
                    break
        else:
            # Revert
            if masks_0[i] & bit_i_j:
                masks_0[i] &= ~bit_i_j
                masks_0[j] &= ~bit_j_i
            else:
                masks_0[i] |= bit_i_j
                masks_0[j] |= bit_j_i
        
        T *= alpha
    
    return best_adj

def _try_extend(adj_current, current_n, target_n, deadline):
    """
    Try to extend current_n vertex coloring to target_n vertices.
    Returns (success, adj) where adj is the extended coloring.
    """
    n = target_n
    adj_new = np.zeros((n, n), dtype=bool)
    
    # Copy existing coloring
    adj_new[:current_n, :current_n] = adj_current
    
    # Initialize new edges with random colors
    for i in range(current_n, n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj_new[i, j] = True
                adj_new[j, i] = True
    for i in range(current_n):
        for j in range(current_n, n):
            if random.random() < 0.5:
                adj_new[i, j] = True
                adj_new[j, i] = True
    
    # Run simulated annealing
    adj_result = _simulated_annealing(adj_new, n, deadline, max_steps=50000)
    
    if _is_valid(adj_result, n):
        return True, adj_result
    
    return False, adj_result

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_n = 1
    best_adj = np.zeros((1, 1), dtype=bool)
    best_colors = []
    
    # Try Paley colorings as starting points
    paley_primes = [17, 29, 13, 5]
    for p in paley_primes:
        if p % 4 != 1:
            continue
        cols = _paley_coloring(p)
        if cols is None:
            continue
        
        adj_test = _colors_to_adj(cols, p)
        if _is_valid(adj_test, p):
            if p > best_n:
                best_n = p
                best_adj = adj_test
                best_colors = cols
    
    # If no valid Paley coloring found, use random small graph
    if best_n <= 1:
        for trial in range(10):
            n_test = random.randint(5, 10)
            adj_test = (np.random.rand(n_test, n_test) > 0.5).astype(bool)
            np.fill_diagonal(adj_test, False)
            adj_test = ((adj_test + adj_test.T) > 0).astype(bool)
            if _is_valid(adj_test, n_test):
                best_n = n_test
                best_adj = adj_test
                best_colors = _adj_to_colors(adj_test, n_test)
                break
    
    # Try to extend to larger n
    current_n = best_n
    current_adj = best_adj.copy()
    
    for target_n in range(current_n + 1, 31):
        if time.time() > deadline:
            break
        
        success, adj_ext = _try_extend(current_adj, current_n, target_n, deadline)
        
        if success:
            current_n = target_n
            current_adj = adj_ext
            best_n = target_n
            best_adj = adj_ext
            best_colors = _adj_to_colors(adj_ext, target_n)
        else:
            # If extension failed, try a few more times with different random seeds
            for retry in range(3):
                if time.time() > deadline:
                    break
                success, adj_ext = _try_extend(current_adj, current_n, target_n, deadline)
                if success:
                    current_n = target_n
                    current_adj = adj_ext
                    best_n = target_n
                    best_adj = adj_ext
                    best_colors = _adj_to_colors(adj_ext, target_n)
                    break
    
    # Final fallback: ensure we have a valid result
    if best_n < 2:
        best_n = 2
        best_adj = np.zeros((2, 2), dtype=bool)
        best_adj[0, 1] = True
        best_adj[1, 0] = True
        best_colors = [0]
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
