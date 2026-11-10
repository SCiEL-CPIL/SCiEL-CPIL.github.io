import time
import random
import itertools

def _calc_cost(adj, n):
    """
    Calculate the number of regular induced subgraphs of size >= 6.
    adj: list of ints (bitmasks)
    n: number of vertices
    """
    cost = 0
    # Precompute list of vertices for each mask? No, iterate combinations.
    # To speed up, we can iterate masks directly if n is small, but combinations is cleaner.
    # For n=20, 1M subsets.
    
    # Optimization: Check sizes 6 to n
    for k in range(6, n + 1):
        for combo in itertools.combinations(range(n), k):
            # Build mask
            mask = 0
            for v in combo:
                mask |= 1 << v
            
            # Check if regular
            # Get degree of first vertex
            first_v = combo[0]
            target_deg = (adj[first_v] & mask).bit_count()
            
            is_regular = True
            for v in combo[1:]:
                deg = (adj[v] & mask).bit_count()
                if deg != target_deg:
                    is_regular = False
                    break
            
            if is_regular:
                cost += 1
                
    return cost

def _get_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # Try n from 20 down to 10
    for n in range(20, 9, -1):
        # Try a few random restarts for each n
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize random graph
            adj = [0] * n
            # Use a density around 0.5
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= 1 << j
                        adj[j] |= 1 << i
            
            current_cost = _calc_cost(adj, n)
            if current_cost == 0:
                return {'n': n, 'edges': _get_edges(adj, n)}
            
            # Local Search / Simulated Annealing
            T = 1.0
            min_T = 0.01
            steps = 2000
            
            for step in range(steps):
                if time.time() > deadline:
                    break
                
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                # Flip edge
                if adj[u] & (1 << v):
                    adj[u] ^= 1 << v
                    adj[v] ^= 1 << u
                    was_present = True
                else:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    was_present = False
                
                new_cost = _calc_cost(adj, n)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    if current_cost == 0:
                        return {'n': n, 'edges': _get_edges(adj, n)}
                else:
                    # Accept with probability
                    diff = new_cost - current_cost
                    if random.random() < pow(2.718, -diff / T):
                        current_cost = new_cost
                    else:
                        # Revert
                        if was_present:
                            adj[u] |= 1 << v
                            adj[v] |= 1 << u
                        else:
                            adj[u] ^= 1 << v
                            adj[v] ^= 1 << u
                
                T *= 0.99
                if T < min_T:
                    T = min_T
        
    # Fallback to known valid small graph
    return {'n': 5, 'edges': []}
