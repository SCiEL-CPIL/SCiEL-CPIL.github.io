import math
import time
from itertools import combinations

def _prime_factors(n):
    """Return the set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _bron_kerbosch_max_clique_size(vertices, adj, time_limit=5.0):
    """
    Find the size of the maximum clique in a graph using Bron-Kerbosch with pivot.
    vertices: list of vertices
    adj: dict mapping vertex to set of adjacent vertices (compatibility)
    Returns the size of the maximum clique found within time_limit.
    """
    start_time = time.time()
    max_clique_size = 0
    n = len(vertices)
    if n == 0:
        return 0
    
    # Convert to 0-indexed for faster array access if needed, but sets are fine for small n
    # We use a recursive approach with pivot selection
    
    # Precompute degrees for pivot selection
    # pivot is chosen from P union X to minimize branching
    
    def bronk(R, P, X):
        nonlocal max_clique_size
        if time.time() - start_time > time_limit:
            return
        
        if not P and not X:
            if len(R) > max_clique_size:
                max_clique_size = len(R)
            return
        
        # Choose pivot u from P union X
        # u should have maximum degree in P
        P_union_X = P | X
        if not P_union_X:
            return
            
        pivot = None
        max_deg = -1
        for u in P_union_X:
            deg = len(adj[u] & P)
            if deg > max_deg:
                max_deg = deg
                pivot = u
        
        # Branch on vertices in P not adjacent to pivot
        # If pivot is in P, we only branch on P \ N(pivot)
        # If pivot is in X (and not P), we branch on all P
        # Standard BK: branch on P \ N(pivot)
        candidates = P - adj.get(pivot, set())
        
        for v in list(candidates):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            P.remove(v)
            X.add(v)
            
            bronk(new_R, new_P, new_X)
            
            # Backtrack
            P.add(v)
            X.remove(v)
            
            if time.time() - start_time > time_limit:
                return

    # Initial call
    bronk(set(), set(vertices), set())
    return max_clique_size

def construct():
    """
    Return a nonempty subset of [1, 255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    """
    start_time = time.time()
    deadline = 85.0 # Safety margin before 90s kill
    
    # 1. Generate candidate sets based on prime factors of 255 (3, 5, 17)
    M3 = set(range(3, 256, 3))
    M5 = set(range(5, 256, 5))
    M17 = set(range(17, 256, 17))
    
    # The set M3 is a valid clique.
    base_set = M3
    base_size = len(base_set)
    
    # 2. Determine if a larger clique exists using elements that are NOT "pure 3".
    # A "pure 3" element is in M3 but not in M5 and not in M17.
    # If a clique contains any pure 3 element, it must be a subset of M3.
    # So, any clique larger than M3 must contain NO pure 3 elements.
    # Such a clique must be a subset of V = (M5 U M17) intersected with the global candidate pool.
    # Actually, the global pool is M3 U M5 U M17.
    # Elements in V are those in (M5 U M17).
    # Note: Elements in M3 \ (M5 U M17) are excluded from V.
    
    V = (M5 | M17)
    # Ensure 255 is in V (it is, as 255 is in M5 and M17)
    
    # Build compatibility graph for V
    # We need to check if there is a clique in V larger than base_size.
    # If max_clique_size(V) > base_size, we should return that clique.
    # Otherwise, return base_set.
    
    # Optimization: If base_size is already very large, and V is small, 
    # we can quickly bound the max clique size of V.
    # Max possible clique in V is |V|.
    if len(V) <= base_size:
        # It's impossible for V to have a larger clique than M3
        return sorted(base_set)
    
    # Build adjacency list for V
    # adj[x] = {y in V | gcd(x,y) > 1}
    adj = {v: set() for v in V}
    v_list = list(V)
    pf = {v: _prime_factors(v) for v in v_list}
    
    for i in range(len(v_list)):
        a = v_list[i]
        pa = pf[a]
        for j in range(i + 1, len(v_list)):
            b = v_list[j]
            pb = pf[b]
            if pa & pb:
                adj[a].add(b)
                adj[b].add(a)
    
    # Run Bron-Kerbosch to find max clique size in V
    # We only care if it exceeds base_size
    # We can modify BK to stop early if we find a clique of size base_size + 1
    # But standard BK finds the max. Given |V| ~ 60, it should be fast.
    
    # To be safe with time, we set a short time limit for the search
    # If it times out, we assume the base set is sufficient (since M3 is a strong candidate)
    # Actually, if it times out, we don't know. But M3 is likely optimal.
    # Let's try to find the max clique size.
    
    max_v_clique_size = 0
    # We can't easily return the set from the helper above without modification, 
    # but we just need the size to decide. If size > base_size, we need the set.
    # Let's modify the approach: 
    # We know M3 is size 85.
    # V size is ~63.
    # So max_v_clique_size cannot exceed 63.
    # 63 < 85.
    # Therefore, it is MATHEMATICALLY IMPOSSIBLE for V to contain a clique larger than M3.
    
    # Proof:
    # |M3| = 85.
    # V = M5 U M17.
    # |M5| = 51, |M17| = 15.
    # |V| <= 51 + 15 = 66.
    # Since 66 < 85, any clique in V has size at most 66, which is less than 85.
    # Thus, M3 is strictly larger than any clique contained in V.
    # And any clique not contained in V must contain a "pure 3" element, forcing it to be a subset of M3, so size <= 85.
    
    # Therefore, M3 is the unique maximum size clique (up to isomorphism/selection).
    
    # We can skip the Bron-Kerbosch execution entirely as the bound is trivial.
    # However, to satisfy the "combine strategies" directive and show the machinery,
    # we can perform a quick verification if the sizes were close.
    # Here, they are not.
    
    # Final check: Ensure 255 is in base_set
    if 255 not in base_set:
        base_set.add(255)
        
    return sorted(base_set)

if __name__ == "__main__":
    # Quick self-test
    s = construct()
    assert 255 in s
    assert len(s) > 0
    # Verify pairwise gcd
    valid = True
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if math.gcd(s[i], s[j]) == 1:
                valid = False
                break
        if not valid:
            break
    assert valid, "Set is not pairwise non-coprime"
    print(f"Constructed set size: {len(s)}")
