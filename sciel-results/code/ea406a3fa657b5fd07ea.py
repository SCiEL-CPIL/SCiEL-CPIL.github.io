import time
import random
import numpy as np

def _has_k222_count_np(A, n):
    """
    Counts the number of K2,2,2 subgraphs in the graph defined by adjacency matrix A.
    A is an n x n numpy array of 0s and 1s.
    """
    count = 0
    for c in range(n - 1):
        for d in range(c + 1, n):
            if A[c, d] == 0:
                continue
            common = A[:, c] & A[:, d]
            common_verts = np.where(common)[0]
            m_size = len(common_verts)
            if m_size < 4:
                continue
            for i in range(m_size - 1):
                a = common_verts[i]
                for j in range(i + 1, m_size):
                    b = common_verts[j]
                    neighbors = (A[:, a] & A[:, b]) & common
                    if np.sum(neighbors) >= 2:
                        count += 1
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

def _simulated_annealing_np(colors, n, time_limit=30, initial_temp=1.0, cooling_rate=0.9999):
    """
    Simulated annealing with QUBO-inspired cost function.
    The cost is the number of monochromatic K_{2,2,2} subgraphs.
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
    
    while time.time() - start_time < time_limit:
        # Pick a random edge to flip
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
    _, success = _simulated_annealing_np(colors_copy, target_n, time_limit=40, initial_temp=1.0, cooling_rate=0.99995)
    if success:
        return {'n': target_n, 'colors': colors_copy}
    
    # 3. Random Restarts with Simulated Annealing
    for attempt in range(20):
        if time.time() > deadline:
            break
        colors = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
        _, success = _simulated_annealing_np(colors, target_n, time_limit=5, initial_temp=0.5, cooling_rate=0.9999)
        if success:
            return {'n': target_n, 'colors': colors}
    
    # Fallback: Return best found or smaller valid n
    for n in range(28, 10, -1):
        if time.time() > deadline:
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
