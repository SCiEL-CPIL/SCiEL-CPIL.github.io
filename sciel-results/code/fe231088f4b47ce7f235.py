import time
import random

def construct():
    deadline = time.time() + 118
    rng = random.Random(42)

    def has_k4_fast(n, A):
        """Fast K4 check using bitmasks."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                common = A[a] & A[b] & (1 << n) - 1
                for c in range(b + 1, n):
                    if not (common >> c) & 1:
                        continue
                    common2 = common & A[c] & (1 << n) - 1
                    for d in range(c + 1, n):
                        if (common2 >> d) & 1:
                            return True
        return False

    def count_independent_sets(n, A, target=6):
        """Count independent sets of size >= target using complement clique counting."""
        # Build complement adjacency
        comp = [0] * n
        full = (1 << n) - 1
        for i in range(n):
            comp[i] = full & ~(A[i] | (1 << i))
        
        found = [0]
        
        def bk(R, P, X):
            if found[0]:
                return
            if len(R) >= target:
                found[0] = 1
                return
            if not P:
                return
            if len(R) + P.bit_count() < target:
                return
            # Pivot: choose u from X (or P if X empty) to minimize candidates
            if X:
                u = (X & -X).bit_length() - 1
                candidates = P & ~comp[u]
            else:
                candidates = P
            c = candidates
            while c and not found[0]:
                low = c & -c
                v = low.bit_length() - 1
                R.append(v)
                bk(R, P & comp[v], X & comp[v])
                R.pop()
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], full, 0)
        return found[0]

    def valid(n, A):
        if has_k4_fast(n, A):
            return False
        if count_independent_sets(n, A, 6):
            return False
        return True

    def to_edges(n, A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (A[j] >> i) & 1:
                    edges.append([i, j])
        return edges

    # Start with Paley(17)
    q = 17
    squares = {x * x % q for x in range(1, q)}
    n0 = q
    A = [0] * n0
    for i in range(n0):
        for j in range(n0):
            if i != j and (j - i) % q in squares:
                A[i] |= (1 << j)
    assert valid(n0, A), "Paley(17) should be valid"
    best_n = n0
    best_A = list(A)

    def try_add_vertex(cur_n, cur_A, deadline):
        """Try to add one vertex with systematic neighborhood search.
        Key constraint: N(new_vertex) must be triangle-free in G."""
        new_n = cur_n + 1
        full_mask = (1 << cur_n) - 1
        
        # Precompute: for each pair (i,j) that is an edge, mark them
        # N(v) must not contain any triangle
        # Equivalent: for every triangle (a,b,c) in G, N(v) cannot contain all three
        
        # Build list of all triangles in G
        triangles = []
        for a in range(cur_n):
            for b in range(a + 1, cur_n):
                if not ((cur_A[b] >> a) & 1):
                    continue
                common = cur_A[a] & cur_A[b] & full_mask
                for c in range(b + 1, cur_n):
                    if (common >> c) & 1:
                        triangles.append((a, b, c))
        
        if not triangles:
            # No triangles, any neighborhood works for K4 constraint
            # Just need to check independent set constraint
            pass
        
        # Try neighborhoods of various sizes
        # Start with medium-sized neighborhoods (around cur_n/2)
        # Use random sampling with bias toward valid patterns
        
        best_result = None
        
        # Precompute for each vertex i: its adjacency bitmask
        adj = cur_A  # already bitmasks
        
        # For a candidate neighbor set S (as bitmask), check:
        # 1. S is triangle-free: no triangle (a,b,c) has all three in S
        # 2. Adding vertex with neighborhood S doesn't create independent 6-set
        #    The non-neighbors of v in the new graph = {v} ∪ (V \ S)
        #    We need: no independent 5-set in V\S that is also independent from v
        #    Since v is non-adjacent to V\S, any independent 5-set in V\S 
        #    that has no edges among themselves gives independent 6-set with v
        #    So: alpha(G[V\S]) < 5
        
        # Strategy: enumerate candidate S using random + systematic
        attempts = 0
        max_attempts = 500
        
        # First try: random neighborhoods
        while attempts < max_attempts and time.time() < deadline - 20:
            attempts += 1
            target_deg = cur_n // 2 + rng.randint(-cur_n // 5, cur_n // 5)
            target_deg = max(1, min(cur_n - 1, target_deg))
            
            neighbors = set(rng.sample(range(cur_n), target_deg))
            S = 0
            for v in neighbors:
                S |= (1 << v)
            
            # Check triangle-free: for each triangle, not all three in S
            tri_ok = True
            for (a, b, c) in triangles:
                if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1:
                    tri_ok = False
                    break
            if not tri_ok:
                continue
            
            # Build candidate graph
            cand_A = list(cur_A)
            cand_A.append(0)
            for i in range(cur_n):
                if (S >> i) & 1:
                    cand_A[i] |= (1 << new_n - 1)
                    cand_A[new_n - 1] |= (1 << i)
            
            # Full validation
            if has_k4_fast(new_n, cand_A):
                continue
            if count_independent_sets(new_n, cand_A, 6):
                continue
            
            return new_n, cand_A
        
        # Systematic: try all subsets of size around cur_n/2 ± small range
        # This is expensive, so limit it
        if time.time() < deadline - 30:
            # Try a few systematic patterns
            for deg_offset in range(-2, 3):
                target_deg = cur_n // 2 + deg_offset
                if target_deg < 1 or target_deg > cur_n - 1:
                    continue
                # Try a few random samples of this exact size
                for _ in range(20):
                    if time.time() > deadline - 20:
                        break
                    neighbors = set(rng.sample(range(cur_n), target_deg))
                    S = 0
                    for v in neighbors:
                        S |= (1 << v)
                    
                    tri_ok = True
                    for (a, b, c) in triangles:
                        if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1:
                            tri_ok = False
                            break
                    if not tri_ok:
                        continue
                    
                    cand_A = list(cur_A)
                    cand_A.append(0)
                    for i in range(cur_n):
                        if (S >> i) & 1:
                            cand_A[i] |= (1 << new_n - 1)
                            cand_A[new_n - 1] |= (1 << i)
                    
                    if has_k4_fast(new_n, cand_A):
                        continue
                    if count_independent_sets(new_n, cand_A, 6):
                        continue
                    
                    return new_n, cand_A
        
        return None

    def local_improve(cur_n, cur_A, deadline):
        """Try to improve by flipping edges to enable growth."""
        best_A = list(cur_A)
        best_valid = valid(cur_n, best_A)
        
        for _ in range(50):
            if time.time() > deadline - 15:
                break
            
            # Try flipping one edge
            i = rng.randrange(cur_n)
            j = rng.randrange(cur_n)
            if i == j:
                continue
            
            trial_A = list(best_A)
            if (trial_A[j] >> i) & 1:
                trial_A[i] &= ~(1 << j)
                trial_A[j] &= ~(1 << i)
            else:
                trial_A[i] |= (1 << j)
                trial_A[j] |= (1 << i)
            
            if valid(cur_n, trial_A):
                best_A = trial_A
        
        return best_A

    # Main search loop
    while time.time() < deadline - 5:
        # Try to grow
        if best_n < 40:
            result = try_add_vertex(best_n, best_A, deadline)
            if result:
                best_n = result[0]
                best_A = result[1]
                continue
        
        # Local improvement to enable growth
        improved = local_improve(best_n, best_A, deadline)
        if improved != best_A:
            best_A = improved
            # Try growing again
            if best_n < 40 and time.time() < deadline - 10:
                result = try_add_vertex(best_n, best_A, deadline)
                if result:
                    best_n = result[0]
                    best_A = result[1]
                    continue
        
        # If stuck, try a different random seed perturbation
        if time.time() < deadline - 10:
            perturbed = list(best_A)
            for _ in range(rng.randint(2, 6)):
                i = rng.randrange(best_n)
                j = rng.randrange(best_n)
                if i != j:
                    if (perturbed[j] >> i) & 1:
                        perturbed[i] &= ~(1 << j)
                        perturbed[j] &= ~(1 << i)
                    else:
                        perturbed[i] |= (1 << j)
                        perturbed[j] |= (1 << i)
            
            if valid(best_n, perturbed):
                best_A = perturbed
                if best_n < 40:
                    result = try_add_vertex(best_n, best_A, deadline)
                    if result:
                        best_n = result[0]
                        best_A = result[1]

    return {'n': best_n, 'edges': to_edges(best_n, best_A)}
