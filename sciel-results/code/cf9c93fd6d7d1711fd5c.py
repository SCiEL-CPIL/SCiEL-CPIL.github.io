import time
import random
import math

DEADLINE = 115

def _compute_score(n, adj_masks, degs):
    """
    Compute the largest degree sum over triangles using bitmask intersection.
    """
    max_sum = 0
    for u in range(n):
        au = adj_masks[u]
        for v in range(u + 1, n):
            if (au >> v) & 1:
                common = au & adj_masks[v]
                m = common
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    s = degs[u] + degs[v] + degs[w]
                    if s > max_sum:
                        max_sum = s
                    m ^= lsb
    return max_sum

def _state_to_edges(adj_masks):
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
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_pairs_set = set(all_pairs)
    
    best_score = float('inf')
    best_adj_masks = None
    
    # Helper to compute a "smooth" score estimate for guiding swaps
    # We use a soft-max approximation: max(x) ~ sum(x * exp(x/T)) / sum(exp(x/T))
    # But for discrete swaps, we can just evaluate the exact score.
    # The "inspiration" suggests using a surrogate. 
    # Let's stick to the exact score for the decision, but perhaps use a 
    # heuristic to pick better swaps?
    
    # The main improvement from inspiration 1 is the bitmask speedup (already in current)
    # and the structure of the local search (restarts, specific initialization).
    
    # The main improvement from inspiration 2 is the "Max-Cut relaxation" idea.
    # This is hard to implement directly in a discrete swap search without a lot of overhead.
    # However, we can use a degree-balance heuristic.
    # If we remove an edge (u,v) and add (x,y), the degrees change.
    # We want to minimize the max triangle sum.
    # Triangles are formed by edges.
    
    # Let's refine the local search to be more aggressive and use the time budget better.
    
    def _init_k88_plus_one():
        adj_masks = [0] * n
        degs = [0] * n
        for u in range(8):
            for v in range(8, 16):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                degs[u] += 1
                degs[v] += 1
        # Add one internal edge
        if random.random() < 0.5:
            u, v = random.sample(range(8), 2)
        else:
            u, v = random.sample(range(8, 16), 2)
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        degs[u] += 1
        degs[v] += 1
        return adj_masks, degs

    def _init_random():
        adj_masks = [0] * n
        degs = [0] * n
        shuffled_pairs = all_pairs[:]
        random.shuffle(shuffled_pairs)
        edges = shuffled_pairs[:m_target]
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            degs[u] += 1
            degs[v] += 1
        return adj_masks, degs

    def _local_search(adj_masks, degs, time_budget_end):
        nonlocal best_score, best_adj_masks
        
        current_score = _compute_score(n, adj_masks, degs)
        if current_score < best_score:
            best_score = current_score
            best_adj_masks = adj_masks[:]
            
        current_edges_list = []
        current_edges_set = set()
        for i in range(n):
            mask = adj_masks[i]
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    current_edges_list.append((i, j))
                    current_edges_set.add((i, j))
                    
        T = 10.0
        T_min = 0.01
        cooling = 0.99999
        
        # Pre-compute list of non-edges for faster random selection?
        # 120 total pairs, 65 edges -> 55 non-edges.
        # Rebuilding non-edges list on every step is O(n^2) = 256 ops.
        # Picking random pair and checking set is O(1) but might fail often?
        # Probability of picking an edge is 65/120 ~ 54%. So ~46% success rate.
        # This is fine.
        
        while time.time() < time_budget_end:
            if not current_edges_list:
                break
                
            rem_u, rem_v = random.choice(current_edges_list)
            add_u, add_v = random.choice(all_pairs)
            
            if (add_u, add_v) in current_edges_set:
                continue
                
            # Apply swap
            adj_masks[rem_u] &= ~(1 << rem_v)
            adj_masks[rem_v] &= ~(1 << rem_u)
            degs[rem_u] -= 1
            degs[rem_v] -= 1
            current_edges_list.remove((rem_u, rem_v))
            current_edges_set.remove((rem_u, rem_v))
            
            adj_masks[add_u] |= (1 << add_v)
            adj_masks[add_v] |= (1 << add_u)
            degs[add_u] += 1
            degs[add_v] += 1
            current_edges_list.append((add_u, add_v))
            current_edges_set.add((add_u, add_v))
            
            new_score = _compute_score(n, adj_masks, degs)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj_masks = adj_masks[:]
            else:
                # Revert
                adj_masks[add_u] &= ~(1 << add_v)
                adj_masks[add_v] &= ~(1 << add_u)
                degs[add_u] -= 1
                degs[add_v] -= 1
                current_edges_list.remove((add_u, add_v))
                current_edges_set.remove((add_u, add_v))
                
                adj_masks[rem_u] |= (1 << rem_v)
                adj_masks[rem_v] |= (1 << rem_u)
                degs[rem_u] += 1
                degs[rem_v] += 1
                current_edges_list.append((rem_u, rem_v))
                current_edges_set.add((rem_u, rem_v))
                
            T *= cooling
            if T < T_min:
                T = 5.0

    # Main loop: multiple restarts
    # We will alternate between K8,8+1 and random starts.
    # K8,8+1 is a strong baseline (score 24). Random might find better structures.
    
    while time.time() - start_time < DEADLINE:
        # Determine initialization strategy
        if random.random() < 0.7:
            adj_masks, degs = _init_k88_plus_one()
        else:
            adj_masks, degs = _init_random()
            
        # Run local search for a short duration or until stuck?
        # Let's give each restart a fixed time slice, or run until time is low.
        # If we run until time is low, we might only do 1 restart.
        # Better to do many short restarts or one long one?
        # The current program does one long one with re-heating.
        # Inspiration 1 does multiple restarts.
        
        # Let's do a restart every few seconds or when stuck?
        # Simpler: just run the local search loop with the global deadline.
        # But we need to restart to escape local minima.
        
        # Strategy: Run local search for a fixed number of steps or time slice.
        # Let's use time slices of ~5 seconds.
        slice_end = min(time.time() + 5.0, DEADLINE)
        _local_search(adj_masks, degs, slice_end)
        
        if best_score <= 23:
            break

    if best_adj_masks is None:
        # Fallback
        adj_masks, degs = _init_k88_plus_one()
        best_adj_masks = adj_masks
        
    best_edges = _state_to_edges(best_adj_masks)
    return {'n': n, 'edges': best_edges}
