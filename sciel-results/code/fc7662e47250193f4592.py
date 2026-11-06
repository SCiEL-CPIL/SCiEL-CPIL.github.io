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
        du = degs[u]
        for v in range(u + 1, n):
            if (adj_u >> v) & 1:
                # Common neighbors of u and v
                common = adj_u & adj_masks[v]
                # Iterate through set bits in common
                m = common
                base = du + degs[v]
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    s = base + degs[w]
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
            
        # Simulated annealing parameters
        T_start = 5.0
        T_end = 0.1
        cooling_rate = 0.999
        
        # Local search loop with simulated annealing
        edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in initial_edges)
        current_edges_list = list(edge_set)
        
        # Maintain a list of non-edges for faster random selection
        non_edges = [p for p in all_pairs if p not in edge_set]
        
        T = T_start
        step_count = 0
        best_T = T_start
        
        while time.time() - start_time < time_budget_remaining:
            step_count += 1
            T *= cooling_rate
            
            # Pick random edge to remove
            rem_u, rem_v = random.choice(current_edges_list)
            
            # Pick random non-edge to add
            if not non_edges:
                break
            add_u, add_v = random.choice(non_edges)
            
            # Perform swap
            # Remove rem
            adj_masks[rem_u] &= ~(1 << rem_v)
            adj_masks[rem_v] &= ~(1 << rem_u)
            degs[rem_u] -= 1
            degs[rem_v] -= 1
            edge_set.remove((rem_u, rem_v))
            
            # Add add
            adj_masks[add_u] |= (1 << add_v)
            adj_masks[add_v] |= (1 << add_u)
            degs[add_u] += 1
            degs[add_v] += 1
            edge_set.add((add_u, add_v))
            
            new_score = _compute_score_fast(n, adj_masks, degs)
            
            delta = new_score - current_score
            
            # Accept if improvement, or with probability exp(-delta/T) if worsening
            if delta < 0:
                accept = True
            else:
                # For discrete scores, use a simple acceptance criterion
                # Accept worsening moves with probability based on temperature
                if T > 0.01:
                    import math
                    # Scale delta to make it meaningful for discrete scores
                    # A delta of 1 in score is significant
                    accept_prob = math.exp(-delta / (T * 2.0))
                    accept = random.random() < accept_prob
                else:
                    accept = False
            
            if accept:
                current_score = new_score
                # Update best
                if current_score < best_score:
                    best_score = current_score
                    best_edges = _state_to_edges(adj_masks)
                    best_T = T
                
                # Update lists efficiently
                # rem is now non-edge
                # add is now edge
                # We need to update current_edges_list and non_edges
                # current_edges_list: remove rem, add add
                # non_edges: remove add, add rem
                
                # Find index of rem in current_edges_list
                try:
                    idx_rem = current_edges_list.index((rem_u, rem_v))
                    current_edges_list[idx_rem] = (add_u, add_v)
                except ValueError:
                    # Should not happen if logic is correct, but safe fallback
                    current_edges_list = list(edge_set)
                
                # Find index of add in non_edges
                try:
                    idx_add = non_edges.index((add_u, add_v))
                    non_edges[idx_add] = (rem_u, rem_v)
                except ValueError:
                    non_edges = [p for p in all_pairs if p not in edge_set]
                    
            else:
                # Revert
                # Remove add
                adj_masks[add_u] &= ~(1 << add_v)
                adj_masks[add_v] &= ~(1 << add_u)
                degs[add_u] -= 1
                degs[add_v] -= 1
                edge_set.remove((add_u, add_v))
                
                # Add rem
                adj_masks[rem_u] |= (1 << rem_v)
                adj_masks[rem_v] |= (1 << rem_u)
                degs[rem_u] += 1
                degs[rem_v] += 1
                edge_set.add((rem_u, rem_v))
                
                # If temperature is very low and we're stuck, break
                if T < 0.05 and step_count > 500:
                    break
            
    # Multiple restarts
    num_restarts = 1000
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
