"""Rewritten program using local search to find a small tournament where every 4-set has a common dominator."""
import numpy as np
from itertools import combinations
import random

def count_bad_quadruples(adj, n):
    """Count the number of 4-sets that do NOT have a common dominator.
    A 4-set {a,b,c,d} has a common dominator if there exists some vertex v (possibly in the set or not)
    such that v->a, v->b, v->c, v->d all hold.
    
    Wait, re-reading the problem: "Every set of 4 vertices must have a vertex with arcs to all four."
    This means for every subset S of 4 vertices, there exists a vertex v (which could be in S or outside S)
    such that v has arcs to all vertices in S.
    
    Actually, let me re-read: "Every set of 4 vertices must have a vertex with arcs to all four."
    The vertex could be any vertex in the tournament, not necessarily in the set of 4.
    
    So for a 4-set S = {a,b,c,d}, we need: exists v in {0,...,n-1} such that v->a AND v->b AND v->c AND v->d.
    
    This is equivalent to: the intersection of in-neighbors of a, b, c, d is non-empty.
    """
    # For each vertex v, let out[v] be the set of vertices that v points to.
    # A 4-set S has a common dominator if there exists v such that S subset of out[v].
    
    # Alternative: for each 4-set S, check if any v dominates all of S.
    # This is expensive for large n. Let's think of a smarter way.
    
    # For each vertex v, the number of 4-sets it dominates is C(out_degree(v), 4).
    # But we need to count 4-sets that are dominated by AT LEAST ONE vertex.
    # This is a union count, which is hard.
    
    # Let's just iterate over all 4-sets and check.
    # For n=31, C(31,4) = 31465. For each 4-set, we check up to n vertices. That's ~1M ops, fine.
    
    bad_count = 0
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        # Check if any vertex v dominates all four
        found = False
        for v in range(n):
            if (adj[v, a] == 1 and adj[v, b] == 1 and adj[v, c] == 1 and adj[v, d] == 1):
                found = True
                break
        if not found:
            bad_count += 1
    return bad_count


def construct():
    # We want to minimize n such that there exists a tournament on n vertices where every 4-set has a common dominator.
    # Erdos lower bound: f(4) >= 31. Known: 67 (Paley).
    # Let's try to find a valid tournament for small n using local search.
    
    # Start with n = 31 (the lower bound) and try to find a valid tournament.
    # If we can't, try n = 32, 33, etc.
    
    random.seed(42)
    np.random.seed(42)
    
    for n in range(31, 70):
        # Try multiple random restarts
        for restart in range(20):
            # Initialize random tournament
            # adj[i,j] = 1 means i -> j
            adj = np.zeros((n, n), dtype=np.uint8)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 0
                    else:
                        adj[i, j] = 0
                        adj[j, i] = 1
            
            bad = count_bad_quadruples(adj, n)
            
            if bad == 0:
                # Found a valid tournament!
                orientation = []
                for i in range(n):
                    for j in range(i + 1, n):
                        # 0 means i -> j, 1 means j -> i
                        orientation.append(int(1 - adj[i, j]))
                return {'n': n, 'orientation': orientation}
            
            # Local search: flip edges that reduce bad count
            # Simulated annealing
            T = 1.0
            cooling = 0.999
            min_T = 0.01
            max_iter = 5000
            
            for it in range(max_iter):
                if bad == 0:
                    orientation = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            orientation.append(int(1 - adj[i, j]))
                    return {'n': n, 'orientation': orientation}
                
                if T < min_T:
                    break
                
                # Pick a random edge to flip
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                
                # Flip edge (i, j)
                old_ij = adj[i, j]
                adj[i, j] = 1 - old_ij
                adj[j, i] = 1 - adj[i, j]
                
                new_bad = count_bad_quadruples(adj, n)
                
                # Accept if better or with probability exp(-delta/T)
                delta = new_bad - bad
                if delta < 0 or random.random() < np.exp(-delta / T):
                    bad = new_bad
                else:
                    # Revert
                    adj[i, j] = old_ij
                    adj[j, i] = 1 - old_ij
                
                T *= cooling
        
        # If we couldn't find a valid tournament for this n, try larger n
    
    # Fallback: return the Paley tournament on 83 vertices if local search fails
    p = 83
    squares = {x * x % p for x in range(1, p)}
    return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}
