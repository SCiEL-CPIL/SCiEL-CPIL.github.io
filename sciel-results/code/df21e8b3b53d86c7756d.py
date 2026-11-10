"""Combined: Fast bitmask C6 + numpy greedy + aggressive multi-edge perturbation."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np

DEADLINE = 140.0  # Safety margin before 150s kill

def _check_c6_bitmask(adj, u, v, n=40):
    """
    Check if adding edge (u,v) creates a C6 using bitmask operations.
    Path: u - x1 - x2 - x3 - x4 - v (all distinct, x's not u or v)
    """
    # x1 in N(v) \ {u}
    x1_mask = adj[v] & ~(1 << u)
    while x1_mask:
        x1 = (x1_mask & (-x1_mask)).bit_length() - 1
        x1_mask &= x1_mask - 1
        
        # x4 in N(u) \ {v, x1}
        x4_mask = adj[u] & ~(1 << v) & ~(1 << x1)
        while x4_mask:
            x4 = (x4_mask & (-x4_mask)).bit_length() - 1
            x4_mask &= x4_mask - 1
            
            # x2 in N(x1) \ {v, x4, u}
            x2_mask = adj[x1] & ~(1 << v) & ~(1 << x4) & ~(1 << u)
            while x2_mask:
                x2 = (x2_mask & (-x2_mask)).bit_length() - 1
                x2_mask &= x2_mask - 1
                
                # x3 in N(x4) \ {u, x2, v, x1}
                x3_mask = adj[x4] & ~(1 << u) & ~(1 << x2) & ~(1 << v) & ~(1 << x1)
                while x3_mask:
                    x3 = (x3_mask & (-x3_mask)).bit_length() - 1
                    x3_mask &= x3_mask - 1
                    
                    # Check if x2-x3 is an edge
                    if adj[x2] & (1 << x3):
                        return False  # C6 found
    return True  # No C6 through this edge

def _count_edges_bitmask(adj, n=40):
    count = 0
    for i in range(n):
        count += bin(adj[i]).count('1')
    return count // 2

def _get_edges_bitmask(adj, n=40):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def _is_c6_free_bitmask(adj_bits, n=40):
    """Full check: verify no C6 exists in the graph."""
    for i in range(n):
        for j in range(i+1, n):
            if adj_bits[i] & (1 << j):
                if not _check_c6_bitmask(adj_bits, i, j, n):
                    return False
    return True

def _has_c6_numpy(adj_np, u, v, n=40):
    """
    Check if adding edge (u, v) would create a C6 using numpy adjacency matrix.
    Path: u -> a -> b -> c -> d -> v where all vertices distinct.
    """
    # Get neighbors efficiently
    neighbors_u = np.nonzero(adj_np[u])[0]
    for a in neighbors_u:
        if a == v:
            continue
        neighbors_a = np.nonzero(adj_np[a])[0]
        for b in neighbors_a:
            if b == u or b == v or b == a:
                continue
            neighbors_b = np.nonzero(adj_np[b])[0]
            for c in neighbors_b:
                if c == u or c == v or c == a or c == b:
                    continue
                neighbors_c = np.nonzero(adj_np[c])[0]
                for d in neighbors_c:
                    if d == u or d == v or d == a or d == b or d == c:
                        continue
                    if adj_np[d][v] == 1:
                        return True
    return False

def _greedy_numpy(adj_np, n=40, max_time=3.0, seed=None):
    """
    Greedy edge addition using numpy. Start from empty or given graph.
    """
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() - start_time > max_time:
            break
        if adj_np[u][v] == 1:
            continue
        # Check if adding this edge creates a C6
        if not _has_c6_numpy(adj_np, u, v, n):
            adj_np[u][v] = 1
            adj_np[v][u] = 1
    
    return adj_np

def _numpy_to_bitmask(adj_np, n=40):
    """Convert numpy adjacency matrix to bitmask representation."""
    adj = [0] * n
    for i in range(n):
        for j in range(n):
            if adj_np[i][j]:
                adj[i] |= (1 << j)
    return adj

def _bitmask_to_numpy(adj, n=40):
    """Convert bitmask adjacency to numpy matrix."""
    adj_np = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if adj[i] & (1 << j):
                adj_np[i][j] = 1
    return adj_np

def _spectral_priority(adj, n=40):
    """
    Compute priority scores for non-edges based on degree and common neighbors.
    Returns list of (edge, score) sorted by score descending.
    """
    scores = []
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] & (1 << j)):
                deg_i = bin(adj[i]).count('1')
                deg_j = bin(adj[j]).count('1')
                common = bin(adj[i] & adj[j]).count('1')
                # Prefer moderate degree, few common neighbors
                score = (deg_i + deg_j) * 0.1 - common * 0.5
                if common <= 1:
                    score += 1.0
                scores.append(((i, j), score))
    scores.sort(key=lambda x: -x[1])
    return scores

def _local_search_aggressive(adj, n=40, max_time=15.0):
    """
    Aggressive local search:
    1. Try to add edges (1-add)
    2. If stuck, remove 2-4 edges and greedily refill
    3. Track best solution found
    """
    start_time = time.time()
    current = _count_edges_bitmask(adj, n)
    best_adj = adj[:]
    best_count = current
    
    stuck_counter = 0
    max_stuck = 8
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Phase 1: Try to add edges
        # Get non-edges sorted by spectral priority
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        
        if non_edges:
            # Score and sort
            scored = []
            for (u, v) in non_edges:
                deg_u = bin(adj[u]).count('1')
                deg_v = bin(adj[v]).count('1')
                common = bin(adj[u] & adj[v]).count('1')
                score = (deg_u + deg_v) * 0.1 - common * 0.5
                if common <= 1:
                    score += 1.0
                scored.append(((u, v), score))
            scored.sort(key=lambda x: -x[1])
            
            # Try top candidates first, then randomize rest
            top_k = min(30, len(scored))
            candidates = [e for e, s in scored[:top_k]]
            rest = [e for e, s in scored[top_k:]]
            random.shuffle(rest)
            candidates.extend(rest)
            
            for u, v in candidates:
                if time.time() - start_time > max_time:
                    return best_adj, best_count
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if _check_c6_bitmask(adj, u, v, n):
                    current += 1
                    improved = True
                    if current > best_count:
                        best_count = current
                        best_adj = adj[:]
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        if improved:
            stuck_counter = 0
            continue
        
        # Stuck - try aggressive perturbation
        stuck_counter += 1
        if stuck_counter > max_stuck:
            break
        
        # Get current edges and non-edges
        cur_edges = []
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    cur_edges.append((i, j))
                else:
                    non_edges.append((i, j))
        
        if len(cur_edges) < 2 or len(non_edges) < 2:
            break
        
        # Determine perturbation size
        if stuck_counter >= 4:
            num_remove = random.choice([3, 4])
        elif stuck_counter >= 2:
            num_remove = random.choice([2, 3])
        else:
            num_remove = 2
        
        # Try multiple perturbations
        max_perturbations = 200
        found_valid = False
        
        for _ in range(max_perturbations):
            if time.time() - start_time > max_time:
                return best_adj, best_count
            
            if len(cur_edges) < num_remove or len(non_edges) < num_remove:
                break
            
            # Pick edges to remove and add
            remove_edges = random.sample(cur_edges, num_remove)
            add_edges = random.sample(non_edges, num_remove)
            
            # Save state
            saved_adj = adj[:]
            
            # Apply removals
            for u, v in remove_edges:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            
            # Apply additions with C6 check
            valid = True
            for u, v in add_edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not _check_c6_bitmask(adj, u, v, n):
                    valid = False
                    break
            
            if valid:
                # Valid perturbation - now try to greedily add more edges
                # Convert to numpy for faster greedy refill
                adj_np = _bitmask_to_numpy(adj, n)
                adj_np = _greedy_numpy(adj_np, n, max_time=min(2.0, max_time - (time.time() - start_time)))
                adj = _numpy_to_bitmask(adj_np, n)
                
                new_count = _count_edges_bitmask(adj, n)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = adj[:]
                    stuck_counter = 0
                    found_valid = True
                    break
                else:
                    # Revert
                    adj = saved_adj[:]
            else:
                # Revert
                adj = saved_adj[:]
        
        if not found_valid and stuck_counter >= max_stuck:
            break

    return best_adj, best_count

def _known_good(n=40):
    """8 disjoint K5s = 100 edges, C6-free."""
    adj = [0] * n
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a+1, 5):
                adj[base+a] |= (1 << (base+b))
                adj[base+b] |= (1 << (base+a))
    return adj

def construct():
    start_time = time.time()
    n = 40
    best_adj = None
    best_edges = 0
    
    # Phase 1: Start from 8xK5 baseline
    adj = _known_good(n)
    best_adj = adj[:]
    best_edges = _count_edges_bitmask(adj, n)
    
    # Phase 2: Local search from 8xK5
    remaining = DEADLINE - (time.time() - start_time) - 5
    if remaining > 5:
        adj_ls, count_ls = _local_search_aggressive(adj, n, max_time=min(12.0, remaining))
        if count_ls > best_edges:
            best_edges = count_ls
            best_adj = adj_ls[:]
    
    # Phase 3: Multiple random greedy restarts + local search
    attempt = 0
    while time.time() - start_time < DEADLINE - 5:
        attempt += 1
        remaining = DEADLINE - (time.time() - start_time) - 5
        if remaining < 3:
            break
        
        # Random greedy with numpy
        adj_np = np.zeros((n, n), dtype=np.int8)
        adj_np = _greedy_numpy(adj_np, n, max_time=min(2.0, remaining / 3), seed=1000 + attempt)
        adj_g = _numpy_to_bitmask(adj_np, n)
        count_g = _count_edges_bitmask(adj_g, n)
        
        if count_g > best_edges:
            best_edges = count_g
            best_adj = adj_g[:]
        
        # Apply local search if promising
        if count_g >= best_edges - 5 and remaining > 5:
            mt_ls = min(8.0, remaining / 2)
            adj_l, count_l = _local_search_aggressive(adj_g, n, max_time=mt_ls)
            if count_l > best_edges:
                best_edges = count_l
                best_adj = adj_l[:]
    
    # Final verification
    if best_adj is None:
        best_adj = _known_good(n)
    
    if not _is_c6_free_bitmask(best_adj, n):
        best_adj = _known_good(n)
    
    return _get_edges_bitmask(best_adj, n)
# EVOLVE-BLOCK-END
