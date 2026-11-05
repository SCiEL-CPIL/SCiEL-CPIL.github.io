"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def _paley_tournament(p):
    """Return orientation list for Paley tournament on p vertices."""
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            orientation.append(0 if (j - i) % p in squares else 1)
    return orientation

def _count_bad_quadruples(orientation, n):
    """Count the number of 4-sets that lack a common dominator.
    
    A 4-set {a,b,c,d} has a common dominator if there exists a vertex v (not necessarily in the set)
    such that v -> a, v -> b, v -> c, v -> d.
    
    Actually, re-reading the problem: "Every set of 4 vertices must have a vertex with arcs to all four."
    This means for every subset S of 4 vertices, there exists some vertex v (could be in S or not) 
    such that v has arcs to all vertices in S.
    
    Wait, let me re-read: "Every set of 4 vertices must have a vertex with arcs to all four."
    I think this means: for every 4-element subset {v1,v2,v3,v4}, there exists a vertex w 
    (possibly one of v1..v4) such that w -> vi for all i.
    
    Actually in tournament theory, a "dominator" of a set S is a vertex that has arcs to all 
    vertices in S. The vertex doesn't need to be in S.
    
    So the property is: for every 4-subset S, there exists a vertex w in V such that 
    w -> v for all v in S.
    
    Let me verify with the Paley tournament.
    """
    # Build adjacency matrix: adj[i][j] = True if i -> j
    m = n * (n - 1) // 2
    adj = [[False] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                adj[i][j] = True
            else:
                adj[j][i] = True
            idx += 1
    
    bad_count = 0
    # For each 4-subset, check if there's a common dominator
    for quad in itertools.combinations(range(n), 4):
        # Check if any vertex dominates all four
        has_dominator = False
        for w in range(n):
            if all(adj[w][v] for v in quad):
                has_dominator = True
                break
        if not has_dominator:
            bad_count += 1
    return bad_count

def _fast_check_property(orientation, n):
    """Check if the tournament satisfies the 4-domination property.
    
    Returns True if valid, False otherwise.
    """
    m = n * (n - 1) // 2
    adj = [[False] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                adj[i][j] = True
            else:
                adj[j][i] = True
            idx += 1
    
    # For each 4-subset, check if there's a common dominator
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        # Check if any vertex dominates all four
        found = False
        for w in range(n):
            if adj[w][a] and adj[w][b] and adj[w][c] and adj[w][d]:
                found = True
                break
        if not found:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    # Start with Paley tournament on 67 vertices (known to satisfy the property)
    p = 67
    squares = {x * x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 
                   for i in range(p) for j in range(i + 1, p)]
    
    n = p
    m = n * (n - 1) // 2
    
    # Build adjacency matrix
    adj = [[False] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                adj[i][j] = True
            else:
                adj[j][i] = True
            idx += 1
    
    # Verify the Paley tournament is valid
    if not _fast_check_property(orientation, n):
        # Fallback to 83
        p = 83
        squares = {x * x % p for x in range(1, p)}
        orientation = [0 if (j - i) % p in squares else 1 
                       for i in range(p) for j in range(i + 1, p)]
        return {'n': p, 'orientation': orientation}
    
    best_n = n
    best_orientation = orientation[:]
    
    # Try to reduce n by removing vertices
    # We'll try to find a smaller valid sub-tournament
    
    # Strategy: try all subsets of size n-1, n-2, etc.
    # But that's expensive. Instead, use a heuristic.
    
    # First, try greedy vertex removal
    for target_n in range(66, 30, -1):
        if time.time() > deadline:
            break
        
        # Try to find a valid sub-tournament of size target_n
        # Use a greedy approach: start with all vertices, try to remove one at a time
        
        current_n = 67
        current_indices = list(range(67))
        
        # Build sub-adjacency
        def get_sub_adj(indices):
            sub_n = len(indices)
            sub_adj = [[False] * sub_n for _ in range(sub_n)]
            for i in range(sub_n):
                for j in range(i + 1, sub_n):
                    vi = indices[i]
                    vj = indices[j]
                    sub_adj[i][j] = adj[vi][vj]
                    sub_adj[j][i] = adj[vj][vi]
            return sub_adj
        
        def check_sub(indices):
            sub_n = len(indices)
            if sub_n < 4:
                return True
            sub_adj = get_sub_adj(indices)
            for quad in itertools.combinations(range(sub_n), 4):
                a, b, c, d = quad
                found = False
                for w in range(sub_n):
                    if sub_adj[w][a] and sub_adj[w][b] and sub_adj[w][c] and sub_adj[w][d]:
                        found = True
                        break
                if not found:
                    return False
            return True
        
        # Greedy removal: try to remove vertices one by one
        for target_n in range(66, 30, -1):
            if time.time() > deadline:
                break
            
            # Try random restarts of greedy removal
            found_valid = False
            for attempt in range(100):
                if time.time() > deadline:
                    break
                
                # Start with all 67 vertices
                current = set(range(67))
                
                # Randomly order vertices for removal
                order = list(current)
                random.shuffle(order)
                
                # Try to remove vertices
                for v in order:
                    if len(current) <= target_n:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Try removing v
                    test_set = current - {v}
                    if len(test_set) < 4:
                        continue
                    
                    # Check if removing v keeps the property
                    # A vertex v can be removed if every 4-set that had v as the only dominator
                    # still has another dominator
                    
                    # Quick check: just verify the sub-tournament
                    indices = sorted(test_set)
                    if check_sub(indices):
                        current = test_set
                
                if len(current) <= target_n and check_sub(sorted(current)):
                    found_valid = True
                    break
            
            if found_valid:
                # Extract the valid sub-tournament
                indices = sorted(current)
                sub_n = len(indices)
                sub_orientation = []
                for i in range(sub_n):
                    for j in range(i + 1, sub_n):
                        vi = indices[i]
                        vj = indices[j]
                        # Find the index in the original orientation
                        if vi < vj:
                            orig_idx = vi * (67 - 1) - vi * (vi + 1) // 2 + (vj - vi - 1)
                            # Actually, let's just use adj
                            sub_orientation.append(0 if adj[vi][vj] else 1)
                        else:
                            sub_orientation.append(0 if adj[vj][vi] else 1)
                
                if sub_n < best_n:
                    best_n = sub_n
                    best_orientation = sub_orientation
                    n = sub_n
                    m = n * (n - 1) // 2
                    orientation = sub_orientation[:]
                
                if time.time() > deadline:
                    break
    
    return {'n': best_n, 'orientation': best_orientation}
# EVOLVE-BLOCK-END
