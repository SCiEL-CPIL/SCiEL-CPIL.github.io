"""Program for the graph construction task."""
import time
import random

DEADLINE = 115  # safety margin before 120s hard kill

def _compute_score_fast(n, adj_masks, degs):
    """
    Compute the largest degree sum over triangles using bitmask intersection.
    adj_masks: list of integers, adj_masks[i] has bit j set if i-j is an edge.
    degs: list of degrees.
    """
    max_sum = 0
    for u in range(n):
        adj_u = adj_masks[u]
        for v in range(u + 1, n):
            if (adj_u >> v) & 1:
                # Common neighbors of u and v
                common = adj_u & adj_masks[v]
                # Iterate through set bits in common
                m = common
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    s = degs[u] + degs[v] + degs[w]
                    if s > max_sum:
                        max_sum = s
                    m ^= lsb
    return max_sum

def _initial_construction():
    """Base construction: K_{8,8} plus one internal edge."""
    n = 16
    edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    return edges

def _graph_to_state(edges):
    """Convert edge list to adj_masks and degs."""
    n = 16
    adj_masks = [0] * n
    degs = [0] * n
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        degs[u] += 1
        degs[v] += 1
    return adj_masks, degs

def _state_to_edges(adj_masks):
    """Convert adj_masks back to edge list."""
    n = 16
    edges = []
    for i in range(n):
        mask = adj_masks[i]
        for j in range(i + 1, n):
            if (mask >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    n = 16
    m_target = 65
    
    # Precompute all potential pairs for fast swap generation
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_score = float('inf')
    best_edges = None
    
    # Helper to try local search from a given state
    def try_local_search(initial_edges, time_budget_remaining):
        nonlocal best_score, best_edges
        
        adj_masks, degs = _graph_to_state(initial_edges)
        current_score = _compute_score_fast(n, adj_masks, degs)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = _state_to_edges(adj_masks)
            
        # Local search loop
        # We will try random swaps.
        # To be efficient, we pick a random edge to remove and random non-edge to add.
        edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in initial_edges)
        current_edges_list = list(edge_set)
        
        # For faster swapping, maintain a list of edges and a set of non-edges?
        # Or just pick random indices.
        # Number of non-edges = 120 - 65 = 55.
        
        # We'll use a list of current edges and a list of non-edges
        # Rebuilding non-edges is O(n^2).
        
        # Simpler: just pick random pairs and check.
        # With n=16, checking membership in a set is fast.
        
        stuck_count = 0
        max_stuck = 2000 # Increased stuck limit due to faster evaluation
        
        while time.time() - start_time < time_budget_remaining:
            # Pick random edge to remove
            rem_u, rem_v = random.choice(current_edges_list)
            # Pick random non-edge to add
            # We need to ensure it's not already an edge
            add_u, add_v = random.choice(all_pairs)
            if (min(add_u, add_v), max(add_u, add_v)) in edge_set:
                continue
            
            # Perform swap on a copy? No, just modify and revert.
            # Modify adj_masks and degs
            # Remove rem
            adj_masks[rem_u] &= ~(1 << rem_v)
            adj_masks[rem_v] &= ~(1 << rem_u)
            degs[rem_u] -= 1
            degs[rem_v] -= 1
            edge_set.remove((min(rem_u, rem_v), max(rem_u, rem_v)))
            
            # Add add
            adj_masks[add_u] |= (1 << add_v)
            adj_masks[add_v] |= (1 << add_u)
            degs[add_u] += 1
            degs[add_v] += 1
            edge_set.add((min(add_u, add_v), max(add_u, add_v)))
            
            new_score = _compute_score_fast(n, adj_masks, degs)
            
            if new_score < current_score:
                current_score = new_score
                stuck_count = 0
                # Update best
                if current_score < best_score:
                    best_score = current_score
                    best_edges = _state_to_edges(adj_masks)
                # Update current_edges_list
                current_edges_list = list(edge_set)
            else:
                # Revert
                # Remove add
                adj_masks[add_u] &= ~(1 << add_v)
                adj_masks[add_v] &= ~(1 << add_u)
                degs[add_u] -= 1
                degs[add_v] -= 1
                edge_set.remove((min(add_u, add_v), max(add_u, add_v)))
                
                # Add rem
                adj_masks[rem_u] |= (1 << rem_v)
                adj_masks[rem_v] |= (1 << rem_u)
                degs[rem_u] += 1
                degs[rem_v] += 1
                edge_set.add((min(rem_u, rem_v), max(rem_u, rem_v)))
                
                stuck_count += 1
                if stuck_count > max_stuck:
                    break
            
    # Multiple restarts
    num_restarts = 1000 # Increased restarts, each is faster now
    for i in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # Generate initial graph
        # Strategy 1: Random
        # Strategy 2: K8,8 + random edge
        if i % 2 == 0:
            # Random graph
            random.shuffle(all_pairs)
            init_edges = [list(p) for p in all_pairs[:m_target]]
        else:
            # K8,8 + random edge within partition
            edges = [[u, v] for u in range(8) for v in range(8, 16)]
            # Pick a random pair within 0-7 or 8-15
            if random.random() < 0.5:
                u = random.randint(0, 7)
                v = random.randint(0, 7)
                while u == v:
                    v = random.randint(0, 7)
            else:
                u = random.randint(8, 15)
                v = random.randint(8, 15)
                while u == v:
                    v = random.randint(8, 15)
            edges.append([u, v])
            init_edges = edges
        
        try_local_search(init_edges, DEADLINE)
        
        # If we found score <= 23, we can stop early as it's likely optimal
        if best_score <= 23:
            break
            
    return {'n': n, 'edges': best_edges}
