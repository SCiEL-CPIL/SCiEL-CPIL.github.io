import time
import random
import itertools

DEADLINE = 150
SAFETY_MARGIN = 5

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_n = 0
    best_edges = []
    
    # Try n from 22 down to 10
    # Allocate time based on n
    time_per_n = 30.0
    
    for n in range(22, 9, -1):
        if time.time() > deadline:
            break
        
        n_start_time = time.time()
        found = False
        
        # More attempts for larger n? Or same?
        # Larger n has more combinations, so each check is slower.
        # But we want to find them if they exist.
        
        attempts = 0
        # Dynamic attempt limit based on remaining time
        while time.time() < deadline and (time.time() - n_start_time) < time_per_n:
            attempts += 1
            
            # Initialize random graph
            adj = [0] * n
            # p=0.5 is standard. Maybe vary p?
            p = random.uniform(0.3, 0.7)
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < p:
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
            
            # Local Search
            current_adj = adj[:]
            valid = False
            
            # Limit local search steps
            max_local_steps = 500
            
            for step in range(max_local_steps):
                if time.time() > deadline:
                    break
                    
                violation = find_violation(current_adj, n)
                if violation is None:
                    valid = True
                    break
                
                # Pick a random edge within the violation to flip
                # This is a simple heuristic.
                # To improve, we could try to flip an edge that breaks the regularity.
                # Flipping an edge (u,v) changes degrees of u and v by +/- 1.
                # If all degrees were d, now they are d+1, d-1, d, ...
                # This breaks regularity unless d+1 == d-1 (impossible) or the other degrees adjust?
                # No, only u and v change. So if k>2, regularity is broken.
                # So any flip inside the violation breaks the regularity of THAT violation.
                # The risk is creating a NEW violation.
                
                u, v = random.choice(list(itertools.combinations(violation, 2)))
                
                # Flip
                current_adj[u] ^= 1 << v
                current_adj[v] ^= 1 << u
                
                # Optional: If we are stuck, randomize
                if step % 100 == 99:
                    # Random kick
                    for _ in range(5):
                        ru, rv = random.randint(0, n-1), random.randint(0, n-1)
                        if ru != rv:
                            current_adj[ru] ^= 1 << rv
                            current_adj[rv] ^= 1 << ru

            if valid:
                edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if (current_adj[u] >> v) & 1:
                            edges.append([u, v])
                            
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            # We found a valid graph for this n.
            # Since we iterate downwards, this is the maximum n we've found so far.
            # We can break the outer loop.
            break
            
    if best_n == 0:
        # Fallback
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}

def find_violation(adj, n):
    # Check subsets of size 6 to n
    # Optimization: Check smaller k first?
    # If a graph has a regular induced subgraph of size 7, it might not have one of size 6.
    # But checking size 6 is faster (fewer combos).
    # However, if we find a size 6 violation, we stop.
    # If we don't find size 6, we must check size 7, etc.
    
    for k in range(6, n + 1):
        for combo in itertools.combinations(range(n), k):
            if is_regular(combo, adj):
                return combo
    return None

def is_regular(vertices, adj):
    mask = 0
    for v in vertices:
        mask |= 1 << v
    
    degs = []
    for v in vertices:
        d = (adj[v] & mask).bit_count() - 1
        degs.append(d)
        
    return len(set(degs)) == 1
