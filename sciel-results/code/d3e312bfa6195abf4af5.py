import time
import random
import numpy as np

def _has_k222_count_np(A, n):
    """
    Counts the number of K2,2,2 subgraphs in the graph defined by adjacency matrix A.
    A is an n x n numpy array of 0s and 1s.
    Optimized: uses matrix operations and early exits.
    """
    count = 0
    for c in range(n - 1):
        for d in range(c + 1, n):
            if A[c, d] == 0:
                continue
            common = A[:, c] & A[:, d]
            common[c] = 0
            common[d] = 0
            m_size = np.count_nonzero(common)
            if m_size < 4:
                continue
            # For each pair (a,b) in common, check if they share >= 2 common neighbors in common
            # Vectorized: for each a in common, count how many b in common also connect to a
            # and share >= 2 neighbors with a within common
            common_idx = np.where(common)[0]
            for i in range(m_size - 1):
                a = common_idx[i]
                # b must be in common_idx[i+1:] and A[a,b]==1
                # Also need |N(a) ∩ N(b) ∩ common| >= 2
                neighbors_a_in_common = A[a, :] & common
                # For each candidate b, check |N(a) ∩ N(b) ∩ common| >= 2
                # Vectorized over all b at once:
                # Build matrix: for each b, count |N(a) ∩ N(b) ∩ common|
                # = sum over v in common of A[a,v] * A[b,v]
                # This is: (A[a, :] & common) @ A[b, :]  ... but we need per-b
                # Use: for all b, compute dot product of (A[a,:] & common) with A[b,:]
                # That's (A[a,:] & common) @ A
                shared_counts = (A[a, :] & common) @ A
                # Now for b in common_idx[i+1:], we need A[a,b]==1 and shared_counts[b] >= 2
                # But shared_counts[b] counts |N(a) ∩ N(b) ∩ common| which includes a itself if A[b,a]==1
                # Actually shared_counts[b] = sum_{v in common} A[a,v]*A[b,v]
                # We need this >= 2, and A[a,b]==1, and b != a, b != c, b != d
                # Since b is in common, b != c and b != d already
                # A[a,b] for b in common: we need A[a,b]==1
                # So: for b in common_idx[i+1:], check A[a,b]==1 and shared_counts[b]>=2
                valid_b = common_idx[i+1:]
                if len(valid_b) == 0:
                    continue
                mask = (A[a, valid_b] == 1) & (shared_counts[valid_b] >= 2)
                count += np.count_nonzero(mask)
    return count

def _is_valid_fast_np(A, n):
    return _has_k222_count_np(A, n) == 0

def _build_adj_np(colors, n):
    A = np.zeros((n, n), dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                A[i, j] = 1
                A[j, i] = 1
            idx += 1
    return A

def _get_color_list_np(A, n):
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(int(A[i, j]))
    return colors

def _paley(n, p):
    """Paley coloring using prime p >= n."""
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)
    return colors

def _find_violation_edges(A, n):
    """
    Find edges that are part of monochromatic K2,2,2 subgraphs.
    Returns a list of (i, j) pairs.
    """
    violating = set()
    for c in range(n - 1):
        for d in range(c + 1, n):
            if A[c, d] == 0:
                continue
            common = A[:, c] & A[:, d]
            common[c] = 0
            common[d] = 0
            m_size = np.count_nonzero(common)
            if m_size < 4:
                continue
            common_idx = np.where(common)[0]
            for i in range(m_size - 1):
                a = common_idx[i]
                valid_b = common_idx[i+1:]
                if len(valid_b) == 0:
                    continue
                shared_counts = (A[a, :] & common) @ A
                mask = (A[a, valid_b] == 1) & (shared_counts[valid_b] >= 2)
                if np.count_nonzero(mask) > 0:
                    # This K2,2,2 exists: {c,d} x {a, b} for each b in mask
                    violating.add((c, d))
                    violating.add((a, c))
                    violating.add((a, d))
                    for b_idx in np.where(mask)[0]:
                        b = valid_b[b_idx]
                        violating.add((a, b))
                        violating.add((b, c))
                        violating.add((b, d))
    return list(violating)

def _simulated_annealing_np(colors, n, time_limit=30, initial_temp=1.0, cooling_rate=0.9999):
    """
    Simulated annealing with targeted edge flipping.
    Prioritizes flipping edges in existing violations.
    """
    start_time = time.time()
    A = _build_adj_np(colors, n)
    current_cost = _has_k222_count_np(A, n)
    
    if current_cost == 0:
        return colors, True
    
    best_cost = current_cost
    best_colors = colors[:]
    temp = initial_temp
    
    # Precompute edge-to-vertex mapping
    edge_to_ij = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_ij.append((i, j))
            idx += 1
    
    # Build reverse mapping
    ij_to_idx = {}
    for idx, (i, j) in enumerate(edge_to_ij):
        ij_to_idx[(i, j)] = idx
        ij_to_idx[(j, i)] = idx
    
    # Get initial violation edges for targeted search
    violation_edges = _find_violation_edges(A, n)
    if not violation_edges:
        return colors, True
    
    step_count = 0
    while time.time() - start_time < time_limit:
        step_count += 1
        
        # 70% chance: pick an edge from violations, 30% random
        if violation_edges and random.random() < 0.7:
            vi, vj = random.choice(violation_edges)
            idx = ij_to_idx.get((vi, vj))
            if idx is None:
                idx = ij_to_idx.get((vj, vi))
            if idx is None:
                idx = random.randint(0, len(colors) - 1)
        else:
            idx = random.randint(0, len(colors) - 1)
        
        i, j = edge_to_ij[idx]
        
        # Flip edge (i, j)
        old_val = A[i, j]
        new_val = 1 - old_val
        A[i, j] = new_val
        A[j, i] = new_val
        colors[idx] = new_val
        
        new_cost = _has_k222_count_np(A, n)
        delta = new_cost - current_cost
        
        if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_colors = colors[:]
                if best_cost == 0:
                    return best_colors, True
            # Update violation edges periodically
            if step_count % 50 == 0 or new_cost == 0:
                violation_edges = _find_violation_edges(A, n)
        else:
            # Revert
            A[i, j] = old_val
            A[j, i] = old_val
            colors[idx] = old_val
        
        temp *= cooling_rate
        if temp < 1e-10:
            temp = 1e-10
    
    return best_colors, best_cost == 0

def construct():
    deadline = time.time() + 140
    best_n = 0
    best_colors = None
    
    # Target n=29
    target_n = 29
    p = 29  # 29 is prime
    
    # 1. Try Paley(29)
    colors = _paley(target_n, p)
    if colors:
        A = _build_adj_np(colors, target_n)
        if _is_valid_fast_np(A, target_n):
            return {'n': target_n, 'colors': colors}
    
    # 2. Try Simulated Annealing from Paley
    colors_copy = colors[:]
    _, success = _simulated_annealing_np(colors_copy, target_n, time_limit=50, initial_temp=1.5, cooling_rate=0.9999)
    if success:
        return {'n': target_n, 'colors': colors_copy}
    
    # 3. Random Restarts with Simulated Annealing
    for attempt in range(30):
        if time.time() > deadline - 10:
            break
        colors = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
        remaining = deadline - 10 - time.time()
        tl = max(3, min(8, remaining / max(1, 30 - attempt)))
        _, success = _simulated_annealing_np(colors, target_n, time_limit=tl, initial_temp=0.8, cooling_rate=0.9999)
        if success:
            return {'n': target_n, 'colors': colors}
    
    # Fallback: Try smaller n with Paley constructions
    for n in range(28, 10, -1):
        if time.time() > deadline - 5:
            break
        p_candidate = None
        for pp in [17, 19, 23, 29, 31]:
            if pp >= n:
                p_candidate = pp
                break
        if not p_candidate:
            continue
        colors = _paley(n, p_candidate)
        if colors:
            A = _build_adj_np(colors, n)
            if _is_valid_fast_np(A, n):
                return {'n': n, 'colors': colors}
    
    # Absolute fallback
    return {'n': 1, 'colors': [0]}
