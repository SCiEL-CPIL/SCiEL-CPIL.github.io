import random
import time
import itertools

def construct():
    start_time = time.time()
    time_limit = 110
    
    # Try n=20 first, as it's known to exist
    # Then 21, 22
    # Actually, try 22, 21, 20
    # But 20 is safest.
    
    for n in [20, 21, 22]:
        if time.time() - start_time > time_limit:
            break
            
        # Try to find valid graph for n
        valid_graph = find_valid_graph(n, time_limit - (time.time() - start_time))
        if valid_graph is not None:
            return valid_graph
            
    # Fallback
    return {'n': 5, 'edges': []}

def find_valid_graph(n, time_budget):
    # Local search with restarts
    best_penalty = float('inf')
    best_adj = None
    
    for restart in range(20):
        if time.time() - time_budget > 0: # Bad check, use absolute time
            break
            
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        penalty = count_violations(adj, n)
        
        if penalty == 0:
            return format_graph(adj, n)
            
        # Local search
        current_adj = adj[:]
        current_penalty = penalty
        temp = 1.0
        cooling = 0.99
        
        for step in range(1000):
            if time.time() - start_time_global > 110: # Need global time
                break
                
            # Pick random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            
            # Flip edge
            if current_adj[u] & (1 << v):
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
            new_penalty = count_violations(current_adj, n)
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                if new_penalty == 0:
                    return format_graph(current_adj, n)
            else:
                # Accept with probability exp(-delta/temp)
                delta = new_penalty - current_penalty
                if random.random() < pow(2.718, -delta/temp):
                    pass # Keep change
                else:
                    # Revert
                    if current_adj[u] & (1 << v):
                        current_adj[u] &= ~(1 << v)
                        current_adj[v] &= ~(1 << u)
                    else:
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                        
            temp *= cooling
            
        if current_penalty < best_penalty:
            best_penalty = current_penalty
            best_adj = current_adj[:]
            
    return None

def count_violations(adj, n):
    penalty = 0
    vertices = list(range(n))
    # Only check size 6 for speed?
    # If we only check size 6, we might miss larger ones.
    # But if we find 0 size-6 violations, we check larger.
    
    # Let's check all sizes >= 6
    for k in range(6, n+1):
        for subset in itertools.combinations(vertices, k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Check if regular
            # Get degrees
            degs = []
            for v in subset:
                degs.append((adj[v] & mask).bit_count())
                
            if len(set(degs)) == 1:
                penalty += 1
                # Early exit if penalty is high?
                # No, we need the count for SA.
                # But if penalty > 100, we can stop and return large number?
                # For SA, relative change matters.
                
    return penalty

def format_graph(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
