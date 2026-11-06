"""Rewritten program using graph-based clique finding for smooth pair sums."""
import time
from math import isqrt
from itertools import combinations

def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    if n % 2 == 0:
        factors.add(2)
        while n % 2 == 0:
            n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            factors.add(f)
            while n % f == 0:
                n //= f
        f += 2
    if n > 1:
        factors.add(n)
    return factors

def _is_smooth(n, allowed_primes):
    """Check if n has only prime factors in allowed_primes."""
    for p in allowed_primes:
        while n % p == 0:
            n //= p
    return n == 1

def _max_clique_branch_bound(adj, vertices, current_clique, best_clique):
    """
    Branch and bound maximum clique.
    adj: list of sets (adjacency)
    vertices: list of candidate vertices to consider
    current_clique: list of vertices in current clique
    best_clique: list (modified in place) - best clique found so far
    """
    if not vertices:
        if len(current_clique) > len(best_clique):
            best_clique[:] = current_clique[:]
        return
    
    # Upper bound: current_clique + all remaining vertices
    if len(current_clique) + len(vertices) <= len(best_clique):
        return
    
    # Order vertices by degree in the subgraph (greedy coloring bound)
    # Simple heuristic: process vertices in order, use bound
    # Use a simple bound: we can add at most len(vertices) more
    # Better: use greedy coloring on the subgraph induced by vertices
    
    # Simple branch and bound: try each vertex
    # Order by degree descending for better pruning
    vlist = sorted(vertices, key=lambda v: len(adj[v] & set(vertices)), reverse=True)
    
    for v in vlist:
        if len(current_clique) + len(vlist) <= len(best_clique):
            break
        
        # v must be connected to all in current_clique
        if not all(v in adj[c] for c in current_clique):
            continue
        
        new_clique = current_clique + [v]
        # New candidates: vertices after v that are adjacent to v
        remaining = [u for u in vlist if u in adj[v]]
        
        _max_clique_branch_bound(adj, remaining, new_clique, best_clique)
        
        # Remove v from consideration for siblings
        # (already handled by iterating through vlist)
        
        vlist.remove(v)

def _find_clique(adj, target_size, candidates, deadline):
    """
    Find a clique of size target_size in the graph.
    adj: dict mapping vertex -> set of neighbors
    candidates: list of candidate vertices
    Returns a list of target_size vertices forming a clique, or None.
    """
    best = []
    _max_clique_branch_bound(adj, candidates, [], best)
    if len(best) >= target_size:
        return best[:target_size]
    return None

def _build_compatibility_graph(candidates, allowed_primes):
    """
    Build adjacency where two candidates are connected if their sum is smooth.
    Returns adj dict.
    """
    n = len(candidates)
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            s = candidates[i] + candidates[j]
            if _is_smooth(s, allowed_primes):
                adj[i].add(j)
                adj[j].add(i)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    n = 12
    max_val = 10000
    
    best_set = None
    best_count = float('inf')
    
    def evaluate(arr):
        primes = set()
        for i in range(n):
            for j in range(i + 1, n):
                primes |= _prime_factors(arr[i] + arr[j])
        return primes
    
    # Strategy: Try small prime sets, build compatibility graph, find max clique
    # Focus on small prime sets first since they give fewer distinct primes
    
    prime_sets_to_try = [
        {2},
        {2, 3},
        {2, 5},
        {2, 3, 5},
        {2, 3, 7},
        {2, 5, 7},
        {2, 3, 5, 7},
        {2, 3, 5, 11},
        {2, 3, 7, 11},
        {2, 3, 5, 7, 11},
    ]
    
    for ps in prime_sets_to_try:
        if time.time() > deadline - 5:
            break
        
        # Generate candidate numbers: focus on numbers that are likely to pair well
        # For smooth sums, numbers that are themselves smooth or near-smooth are good
        # Generate candidates: all numbers up to some limit, filtered by smoothness of potential sums
        
        # Key insight: if we want all pair sums to be ps-smooth, 
        # then for any a, b in our set, a+b must be ps-smooth.
        # So for a fixed a, valid b's are those where a+b is ps-smooth.
        
        # Generate a pool of candidates: numbers 1..max_val where the number itself
        # has "nice" properties. Let's use numbers up to 2000 to keep graph manageable.
        
        # Actually, let's be smarter: generate candidates based on smooth numbers
        # A number x is a good candidate if x + y is smooth for many y in our range.
        
        # Simpler approach: take a pool of candidates and find clique
        # Pool: numbers 1 to 1000 (manageable graph size)
        
        pool_size = 500
        candidates = list(range(1, pool_size + 1))
        
        # Filter: keep only candidates that can form at least n-1 smooth sums
        # with other candidates (pre-filter)
        filtered = []
        for c in candidates:
            count = 0
            for c2 in candidates:
                if c != c2 and _is_smooth(c + c2, ps):
                    count += 1
                    if count >= n - 1:
                        break
            if count >= n - 1:
                filtered.append(c)
        
        if len(filtered) < n:
            continue
        
        # Limit graph size for tractability
        if len(filtered) > 150:
            # Take a subset: prefer smaller numbers
            filtered = sorted(filtered)[:150]
        
        # Build compatibility graph
        adj = _build_compatibility_graph(filtered, ps)
        
        # Find maximum clique
        clique = _find_clique(adj, n, list(range(len(filtered))), deadline)
        
        if clique is not None:
            result = [filtered[i] for i in clique]
            primes = evaluate(result)
            if len(primes) < best_count:
                best_count = len(primes)
                best_set = result
                if len(ps) == len(primes):
                    # We achieved the theoretical minimum for this prime set
                    # Try to see if we can do better with smaller prime sets
                    # But since we're iterating from small to large, this is good
                    break
        else:
            # Try a different pool: maybe larger numbers work better
            # Try candidates that are multiples of small numbers
            for mult in [2, 3, 4, 5, 6]:
                if time.time() > deadline - 5:
                    break
                mult_candidates = [m * mult for m in range(1, max_val // mult + 1)][:100]
                if len(mult_candidates) < n:
                    continue
                
                adj2 = _build_compatibility_graph(mult_candidates, ps)
                clique2 = _find_clique(adj2, n, list(range(len(mult_candidates))), deadline)
                if clique2 is not None:
                    result = [mult_candidates[i] for i in clique2]
                    primes = evaluate(result)
                    if len(primes) < best_count:
                        best_count = len(primes)
                        best_set = result
                    break
    
    # Fallback: structured approach with multiples
    if best_set is None or best_count > 5:
        for m in [2, 3, 4, 6, 8, 12, 16, 24]:
            if time.time() > deadline - 3:
                break
            vals = list(range(m, m + n * m, m))
            if vals[-1] <= max_val:
                primes = evaluate(vals)
                if len(primes) < best_count:
                    best_count = len(primes)
                    best_set = list(vals)
    
    # Local search improvement
    if best_set is not None and time.time() < deadline - 3:
        current = list(best_set)
        current_primes = evaluate(current)
        current_count = len(current_primes)
        
        import random
        rng = random.Random(42)
        
        while time.time() < deadline - 1:
            improved = False
            for i in range(n):
                if time.time() >= deadline - 1:
                    break
                old_val = current[i]
                for trial in range(50):
                    if time.time() >= deadline - 1:
                        break
                    new_val = rng.randint(1, max_val)
                    if new_val == old_val or new_val in current:
                        continue
                    
                    new_current = list(current)
                    new_current[i] = new_val
                    
                    # Recompute only affected sums
                    new_primes = set()
                    for a in range(n):
                        for b in range(a + 1, n):
                            new_primes |= _prime_factors(new_current[a] + new_current[b])
                    
                    if len(new_primes) < current_count:
                        current = new_current
                        current_primes = new_primes
                        current_count = len(new_primes)
                        improved = True
                        if current_count < best_count:
                            best_count = current_count
                            best_set = list(current)
                        break
                    elif len(new_primes) == current_count and rng.random() < 0.05:
                        current = new_current
                        current_primes = new_primes
                        break
            
            if not improved:
                break
    
    if best_set is None:
        # Last resort fallback
        best_set = list(range(1, n + 1))
    
    return best_set
