import time
import random
import numpy as np

def _has_k222_count(adj, n):
    """
    Counts the number of K2,2,2 subgraphs using bitsets.
    adj[i] is an integer bitset where bit j is 1 if edge (i,j) exists.
    """
    count = 0
    for c in range(n):
        for d in range(c + 1, n):
            common = adj[c] & adj[d]
            if common == 0:
                continue
            # Get vertices in common
            verts_in_common = []
            temp = common
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                verts_in_common.append(v)
                temp ^= lsb
            
            m_size = len(verts_in_common)
            if m_size < 4:
                continue
            
            for i in range(m_size - 1):
                a = verts_in_common[i]
                for j in range(i + 1, m_size):
                    b = verts_in_common[j]
                    # Check if a and b have at least 2 common neighbors in common
                    neighbors = adj[a] & adj[b] & common
                    # Check if neighbors has at least 2 bits set
                    if neighbors & (neighbors - 1):
                        count += 1
    return count

def _is_valid(adj, n):
    return _has_k222_count(adj, n) == 0

def _build_adj_bitsets(colors, n):
    adj = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            idx += 1
    return adj

def _get_color_list(adj, n):
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(1 if (adj[i] & (1 << j)) else 0)
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

def _find_violating_edges(adj, n):
    """Find edges that are part of monochromatic K2,2,2 subgraphs."""
    violating = set()
    for c in range(n):
        for d in range(c + 1, n):
            common = adj[c] & adj[d]
            if common == 0:
                continue
            verts_in_common = []
            temp = common
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                verts_in_common.append(v)
                temp ^= lsb
            
            m_size = len(verts_in_common)
            if m_size < 4:
                continue
            
            for i in range(m_size - 1):
                a = verts_in_common[i]
                for j in range(i + 1, m_size):
                    b = verts_in_common[j]
                    neighbors = adj[a] & adj[b] & common
                    if neighbors & (neighbors - 1):
                        # Found a K2,2,2: {c,d} x {a,b} with a,b in common
                        violating.add((c, d))
                        violating.add((a, c))
                        violating.add((a, d))
                        violating.add((a, b))
                        violating.add((b, c))
                        violating.add((b, d))
    return list(violating)

def _local_search_with_repair(colors, n, time_limit=30, initial_temp=1.0, cooling_rate=0.9999):
    """
    Local search with targeted repair of violating edges.
    Uses simulated annealing with bias toward flipping edges in violations.
    """
    start_time = time.time()
    adj = _build_adj_bitsets(colors, n)
    current_cost = _has_k222_count(adj, n)
    
    if current_cost == 0:
        return colors, True
    
    best_cost = current_cost
    best_colors = colors[:]
    temp = initial_temp
    
    # Precompute edge index mapping
    edge_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_idx[(i, j)] = idx
            edge_to_idx[(j, i)] = idx
            idx += 1
    
    # Get initial violation edges
    violation_edges = _find_violating_edges(adj, n)
    if not violation_edges:
        return colors, True
    
    step_count = 0
    stuck_count = 0
    best_stuck = best_cost
    
    while time.time() - start_time < time_limit:
        step_count += 1
        
        # 70% chance: pick an edge from violations, 30% random
        if violation_edges and random.random() < 0.7:
            vi, vj = random.choice(violation_edges)
            idx = edge_to_idx.get((vi, vj))
            if idx is None:
                idx = edge_to_idx.get((vj, vi))
            if idx is None:
                idx = random.randint(0, len(colors) - 1)
        else:
            idx = random.randint(0, len(colors) - 1)
        
        # Flip edge
        old_val = colors[idx]
        colors[idx] = 1 - old_val
        
        # Rebuild adj and check cost
        new_adj = _build_adj_bitsets(colors, n)
        new_cost = _has_k222_count(new_adj, n)
        delta = new_cost - current_cost
        
        if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_cost = new_cost
            adj = new_adj
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_colors = colors[:]
                stuck_count = 0
                if best_cost == 0:
                    return best_colors, True
            else:
                stuck_count += 1
            
            # Update violation edges periodically
            if step_count % 20 == 0 or new_cost == 0:
                violation_edges = _find_violating_edges(adj, n)
        else:
            # Revert
            colors[idx] = old_val
        
        # Perturbation if stuck
        if stuck_count > 100:
            # Kick: flip 3-5 random edges
            kick_size = random.randint(3, 5)
            for _ in range(kick_size):
                kidx = random.randint(0, len(colors) - 1)
                colors[kidx] = 1 - colors[kidx]
            adj = _build_adj_bitsets(colors, n)
            current_cost = _has_k222_count(adj, n)
            violation_edges = _find_violating_edges(adj, n)
            stuck_count = 0
        
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
        adj = _build_adj_bitsets(colors, target_n)
        if _is_valid(adj, target_n):
            return {'n': target_n, 'colors': colors}
    
    # 2. Try local search with repair from Paley
    colors_copy = colors[:]
    remaining = deadline - time.time() - 5
    if remaining > 10:
        _, success = _local_search_with_repair(colors_copy, target_n, time_limit=min(50, remaining), initial_temp=1.5, cooling_rate=0.9999)
        if success:
            return {'n': target_n, 'colors': colors_copy}
    
    # 3. Multiple random restarts with local search
    num_restarts = 20
    for attempt in range(num_restarts):
        if time.time() > deadline - 10:
            break
        colors = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
        remaining = deadline - 10 - time.time()
        tl = max(3, min(8, remaining / max(1, num_restarts - attempt)))
        _, success = _local_search_with_repair(colors, target_n, time_limit=tl, initial_temp=0.8, cooling_rate=0.9999)
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
            adj = _build_adj_bitsets(colors, n)
            if _is_valid(adj, n):
                return {'n': n, 'colors': colors}
    
    # Absolute fallback
    return {'n': 1, 'colors': [0]}
