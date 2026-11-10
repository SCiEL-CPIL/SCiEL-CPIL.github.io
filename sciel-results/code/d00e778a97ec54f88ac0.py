import time
import random
from itertools import combinations

START_TIME = time.time()
DEADLINE = START_TIME + 145.0

def precompute_subsets(n):
    """Precompute subsets and their masks for sizes 6 to n."""
    subsets_by_size = {}
    for size in range(6, n + 1):
        subs = []
        for subset in combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            subs.append((subset, mask))
        subsets_by_size[size] = subs
    return subsets_by_size

def find_violation(n, adj, subsets_by_size):
    """
    Finds a regular induced subgraph of size >= 6.
    Uses precomputed subsets and masks.
    """
    for size in range(6, n + 1):
        if size not in subsets_by_size:
            continue
        for subset, mask in subsets_by_size[size]:
            # Calculate degrees
            # Early exit optimization:
            # Compute first degree
            u0 = subset[0]
            d0 = (adj[u0] & mask).bit_count()
            
            # Check others
            is_regular = True
            for u in subset[1:]:
                if (adj[u] & mask).bit_count() != d0:
                    is_regular = False
                    break
            
            if is_regular:
                return (size, subset)
    return None

def construct():
    best_n = 5
    best_edges = []
    
    for target_n in [22, 21, 20, 19, 18]:
        if time.time() > DEADLINE - 10:
            break
            
        # Precompute subsets for this n
        if time.time() > DEADLINE - 5:
            break
        subsets_by_size = precompute_subsets(target_n)
        
        found = False
        
        for restart in range(100):
            if time.time() > DEADLINE - 5:
                break
                
            adj = [0] * target_n
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
            for step in range(500):
                if time.time() > DEADLINE - 5:
                    break
                    
                viol = find_violation(target_n, adj, subsets_by_size)
                if viol is None:
                    best_n = target_n
                    edges = []
                    for u in range(target_n):
                        for v in range(u + 1, target_n):
                            if (adj[u] >> v) & 1:
                                edges.append([u, v])
                    found = True
                    break
                
                size, subset = viol
                candidate_edges = [(u, v) for u in subset for v in subset if u < v]
                if not candidate_edges:
                    break
                u, v = random.choice(candidate_edges)
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
            if found:
                break
                
        if found:
            break
            
    return {'n': best_n, 'edges': best_edges}
