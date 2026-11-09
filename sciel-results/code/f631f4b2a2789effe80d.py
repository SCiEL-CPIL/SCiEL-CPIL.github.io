import itertools
import time

def check_validity(n, orientation):
    """
    orientation: list of bits for pairs (i,j) i<j.
    0 means i->j, 1 means j->i.
    """
    # Build adjacency: adj[i][j] = 1 if i->j else 0
    # We need in-neighbor bitmasks.
    # in_nei[v] is a bitmask of u such that u->v.
    
    # Map pair index to (i,j)
    # idx = i*n - i*(i+1)/2 + (j-i-1) ? No, standard lexicographic.
    # Let's just build a 2D array or use the list directly.
    
    # For speed, build in_nei[v] as an integer bitmask.
    in_nei = [0] * n
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            bit = orientation[idx]
            if bit == 0:
                # i -> j
                # i is in-neighbor of j
                in_nei[j] |= (1 << i)
            else:
                # j -> i
                # j is in-neighbor of i
                in_nei[i] |= (1 << j)
            idx += 1
            
    # Check all 4-sets
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        # Check if there is a vertex v that dominates all four.
        # v dominates u iff u is in in_nei[v]? No.
        # v dominates u means v->u.
        # So u is an OUT-neighbor of v.
        # My in_nei[v] contains u if u->v.
        # I need out_nei[v] or check if v is in in_nei[u].
        
        # Let's re-evaluate.
        # Condition: Exists v such that v->a, v->b, v->c, v->d.
        # This means a,b,c,d are all in OUT_NEI(v).
        # Equivalently, v is in IN_NEI(a), v is in IN_NEI(b), etc.
        # So we need IN_NEI(a) & IN_NEI(b) & IN_NEI(c) & IN_NEI(d) != 0.
        
        mask = in_nei[a] & in_nei[b] & in_nei[c] & in_nei[d]
        if mask == 0:
            return False
            
    return True

def make_paley(p):
    squares = {x*x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i+1, p):
            # i -> j iff j-i is a square
            if (j-i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def construct():
    # Candidates in increasing order of n
    candidates = [31, 43, 47, 59, 67, 83]
    
    # We can also try to search for a valid tournament for n < 67 if Paley fails.
    # But let's first see which Paley works.
    
    best_n = 83
    best_orient = make_paley(83)
    
    for p in candidates:
        orient = make_paley(p)
        if check_validity(p, orient):
            return {'n': p, 'orientation': orient}
            
    # If none worked (should not happen), return the largest known valid one
    return {'n': best_n, 'orientation': best_orient}
