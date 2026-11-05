"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import deque

_START = None

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _has_triangle(n, adj):
    # adj: list of bitmasks
    for i in range(n):
        ai = adj[i]
        # check if any two neighbors of i are adjacent
        # iterate over set bits of ai
        x = ai
        while x:
            lsb = x & (-x)
            j = lsb.bit_length() - 1
            x ^= lsb
            # check if any neighbor of j (that is also neighbor of i) exists
            # i.e., adj[j] & ai should have no bits other than i? 
            # Actually triangle i-j-k means j in N(i), k in N(i), and j~k.
            # So we need to check if (adj[j] & ai) has any bit set other than i?
            # No, i is not in adj[i]. So we just check if adj[j] & ai != 0.
            # But wait, j is in ai. So adj[j] contains i.
            # We want to know if there is k in ai such that k in adj[j].
            # Since i is not in adj[i], i is not in ai.
            # So (adj[j] & ai) will contain k if k~j and k~i.
            # This is correct.
            if adj[j] & ai:
                return True
    return False

def _max_independent_set_size(n, adj):
    # Exact branch and bound for maximum independent set.
    # adj: list of bitmasks.
    # We want to find if alpha >= 10. If so, return 10 (or more).
    # We can just check if there is an independent set of size 10.
    # This is a decision problem: does alpha >= 10?
    # We can use a simple DFS with pruning.
    
    # Order vertices by degree (descending) to prune faster?
    # Or just use the standard Bron-Kerbosch style for independent set (clique in complement).
    # But complement might be dense.
    # Let's stick to a simple recursive search for an independent set of size 10.
    
    # If n < 10, return n.
    if n < 10:
        return n
        
    # State: current independent set size, available vertices (bitmask)
    # We want to reach size 10.
    # If available vertices count < (10 - current_size), prune.
    
    # Precompute popcount? Python 3.10+ has int.bit_count()
    
    # Recursive function
    def dfs(available, current_size):
        if current_size >= 10:
            return True
        if available == 0:
            return False
        if current_size + available.bit_count() < 10:
            return False
        
        # Pick a vertex v from available
        # Heuristic: pick vertex with highest degree in the induced subgraph on 'available'?
        # Or just pick the lowest index for simplicity, but heuristics help.
        # Let's pick the vertex v in available that minimizes the number of neighbors in available?
        # This is the "minimum degree" heuristic for finding large independent sets (or maximum cliques).
        # Actually, for independent set, we want to remove as few vertices as possible?
        # No, if we pick v, we remove N(v) from available.
        # We want N(v) to be small? So we want v to have low degree in the subgraph.
        
        # Let's just pick the first available vertex for now, maybe optimize later if needed.
        # But performance is key.
        
        # Find v
        # v = (available & -available).bit_length() - 1
        # Let's try to pick the vertex with minimum degree in the available set.
        # This might be expensive to compute every time.
        # Let's just pick the first one.
        
        lsb = available & (-available)
        v = lsb.bit_length() - 1
        
        # Branch 1: include v
        # New available = available \ {v} \ N(v)
        # N(v) is adj[v]. But we only care about neighbors in 'available'.
        # new_available = available & ~adj[v] & ~(1 << v)
        
        if dfs(available & ~adj[v] & ~(1 << v), current_size + 1):
            return True
            
        # Branch 2: exclude v
        # New available = available \ {v}
        if dfs(available & ~(1 << v), current_size):
            return True
            
        return False

    # Initial call
    # available = all vertices
    full_mask = (1 << n) - 1
    if dfs(full_mask, 0):
        return 10
    else:
        # If no independent set of size 10, the max size is < 10.
        # We need to return the actual max size?
        # The task says "no independent set of 10 vertices".
        # So if alpha < 10, it's valid.
        # The function _max_independent_set_size should return the size.
        # But for validity check, we just need to know if it's >= 10.
        # However, the name implies it returns the size.
        # Let's just return 9 if it fails to find 10? 
        # No, that's inaccurate. But for the purpose of `construct`, we only care if it's valid.
        # If we are searching, we might want the exact size to compare candidates?
        # Actually, any valid graph with n>26 is better.
        # So we just need a validity checker.
        # Let's rename or just use it as a checker.
        # If it returns 10, it's invalid. If < 10, it's valid.
        # Since we don't know the exact size if it's < 10, let's just return 9 for simplicity in the search loop?
        # Or we can implement a proper max independent set algorithm.
        # But branch and bound for max IS is similar.
        # For now, returning 9 if <10 is sufficient for the "valid" check.
        return 9

def _is_valid(n, adj):
    if _has_triangle(n, adj):
        return False
    if _max_independent_set_size(n, adj) >= 10:
        return False
    return True

def _greedy_circulant(n, deadline):
    # Try to construct a circulant graph C(n, S) that is triangle-free and alpha < 10.
    # S is a set of shifts.
    # Triangle-free condition for circulant graph:
    # No a, b, c in S such that a+b=0, a+c=0, b+c=0 (mod n) ?
    # Actually, triangle i, i+a, i+b.
    # Edges: (i, i+a), (i, i+b), (i+a, i+b).
    # (i+a, i+b) is an edge iff (i+b - (i+a)) mod n is in S or -S.
    # So b-a mod n must be in S or -S.
    # So for any a, b in S, (b-a) mod n must NOT be in S and NOT be in -S.
    # This means S must be a "sum-free" set in a specific sense?
    # Actually, it means S contains no solutions to x + y = z (mod n) with x,y,z in S?
    # If a, b, c in S and a+b=c, then vertices 0, a, c form a triangle?
    # 0~a (a in S), 0~c (c in S), a~c (c-a=b in S). Yes.
    # So S must be sum-free modulo n.
    # Also, we need alpha < 10.
    
    # Greedy construction of S:
    # Start with S = empty.
    # Try adding shifts.
    # We want to maximize n, so we start with high n (e.g., 40, 39) and try to find S.
    # Or start with small n and increase?
    # The task says maximize n.
    # Known best is 39.
    # Let's try n=40, 39, 38...
    
    for n_target in range(40, 26, -1):
        if _now() > deadline:
            break
        # Try to find a sum-free set S for Z_n such that alpha < 10.
        # Greedy:
        # S = {}
        # candidates = [1, 2, ..., n-1]
        # We can restrict S to be symmetric? i.e., if a in S, n-a in S?
        # Usually circulant graphs are undirected, so S is symmetric.
        # Let's assume S is symmetric.
        # So we pick a, and n-a is also in S.
        # We need to ensure sum-free.
        
        # Let's just do a random search for S for a fixed n.
        # Or a greedy deterministic one.
        
        # Let's try a few random S sets for each n.
        for attempt in range(100):
            if _now() > deadline:
                break
            random.shuffle(range(1, n_target // 2 + 1))
            S = set()
            valid = True
            for a in range(1, n_target // 2 + 1):
                # Check if we can add a and n-a to S
                # Check triangle condition:
                # For all x, y in S, (x-y) mod n not in S and not in -S.
                # Since S is symmetric, -S = S.
                # So (x-y) mod n not in S.
                
                # Let's check if adding a violates this.
                # New elements: a, n-a.
                # Check pairs (a, x), (a, n-x), (n-a, x), (n-a, n-x)
                # Actually, just check all pairs in S U {a, n-a}.
                
                candidate_S = S | {a, n_target - a}
                
                ok = True
                # Check sum-free property
                # For all x, y in candidate_S, (x+y) mod n not in candidate_S?
                # No, triangle condition is: if a, b in S, then (b-a) mod n not in S.
                # This is equivalent to: no a, b, c in S such that a+b=c.
                # Let's check that.
                for x in candidate_S:
                    for y in candidate_S:
                        z = (x + y) % n_target
                        if z in candidate_S:
                            # Found a triangle
                            ok = False
                            break
                    if not ok:
                        break
                
                if ok:
                    S = candidate_S
                else:
                    # Cannot add a
                    pass
            
            # If we have a valid S, check alpha
            # Build adjacency list
            adj = [0] * n_target
            for i in range(n_target):
                for s in S:
                    j = (i + s) % n_target
                    adj[i] |= (1 << j)
                    # Since undirected, we add both ways?
                    # The loop above adds i->j.
                    # We need j->i as well.
                    # But if we iterate all i and all s in S, and S is symmetric,
                    # then j->i will be added when we process j and s' = n-s.
                    # So it's fine.
            
            if _is_valid(n_target, adj):
                # Found a valid graph!
                edges = []
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                return {'n': n_target, 'edges': edges}
        
        # If no valid S found for this n, try smaller n.
        # But we might miss some.
        # Let's add a random restart with different heuristics?
        # For now, just move to next n.

    # If we didn't find anything better than 26, return the seed.
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

def construct():
    # Deadline: 120s wall-clock. Keep safety margin.
    deadline = 115.0
    
    # First, try to find a better graph using greedy circulant search.
    result = _greedy_circulant(40, deadline)
    
    # If the search took too long or failed, the function returns the seed or a found graph.
    # _greedy_circulant always returns a valid graph (at least the seed).
    # But we should double check if it's valid?
    # The task says "both conditions are checked by exact branch and bound".
    # Our _is_valid does that.
    # So the returned graph is valid.
    
    return result

# EVOLVE-BLOCK-END
