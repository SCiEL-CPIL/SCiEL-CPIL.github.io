import time
import random
from math import gcd

def construct():
    start_time = time.time()
    timeout = 88.0  # Safety margin for 90s limit
    
    N = 256
    values = list(range(2, N + 1))
    n = len(values)
    
    # Precompute conflict: conflict[i][j] = True if lcm(values[i], values[j]) <= 256
    # Use 0-indexed: values[0]=2, values[1]=3, ..., values[254]=256
    conflict = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            # lcm(a, b) = a * b // gcd(a, b)
            # We want lcm > 256, so conflict if lcm <= 256
            # a * b // gcd(a, b) <= 256
            # a * b <= 256 * gcd(a, b)
            # Since a, b <= 256, a*b <= 65536. 256*gcd(a,b) can be up to 256*256=65536.
            # Check directly:
            g = gcd(values[i], values[j])
            if values[i] * values[j] <= 256 * g:
                conflict[i][j] = True
                conflict[j][i] = True
    
    weights = [1.0 / values[i] for i in range(n)]
    
    # 1. Structural Decomposition: Group numbers by their largest prime factor
    # A number is either a prime or has a largest prime factor p.
    # If two numbers share the same largest prime factor p, they are in the same "chain".
    # Note: Numbers like 12 (2^2*3) have largest prime factor 3. 18 (2*3^2) has largest prime factor 3.
    # They are in the same chain.
    # Primes p are in their own chain (or grouped by p).
    
    chains = {}
    for i in range(n):
        v = values[i]
        # Find largest prime factor
        largest_pf = v
        d = 2
        x = v
        while d * d <= x:
            while x % d == 0:
                x //= d
                largest_pf = d
            d += 1
        if x > 1:
            largest_pf = x
            
        if largest_pf not in chains:
            chains[largest_pf] = []
        chains[largest_pf].append(i)
    
    # 2. Solve WIS for each chain
    # Within a chain, conflicts are structured.
    # For a fixed largest prime p, numbers are of form k * p^e where p does not divide k.
    # Conflict condition: lcm(a, b) <= 256.
    # This is a specific graph. We can solve WIS on each chain exactly using DP or Branch and Bound.
    # Given chain sizes are small (max ~10-20), we can use a simple recursive DP with memoization on bitmask?
    # Or just greedy + local search within chain?
    # Let's try a simple exact solver for small graphs.
    # Since max chain size is small, we can even do brute force if < 20.
    # But let's use a standard WIS heuristic or DP if tree-like.
    # Actually, let's just use a local search on the whole graph but initialized by solving chains independently.
    
    initial_S = set()
    
    for p, indices in chains.items():
        if not indices:
            continue
        # Solve WIS on this subgraph
        m = len(indices)
        # Create local conflict matrix
        local_conflict = [[False] * m for _ in range(m)]
        for a in range(m):
            for b in range(a + 1, m):
                if conflict[indices[a]][indices[b]]:
                    local_conflict[a][b] = True
                    local_conflict[b][a] = True
        
        # Simple DP for WIS on small graph?
        # Or just greedy by weight descending?
        # Let's try a simple branch and bound for small m.
        # Or even simpler: since these are chains, maybe we can order them?
        # Let's use a simple local search on the subgraph.
        best_local = set()
        best_local_val = 0.0
        
        # Greedy init
        order = sorted(range(m), key=lambda i: weights[indices[i]], reverse=True)
        curr_local = set()
        for i in order:
            ok = True
            for j in curr_local:
                if local_conflict[i][j]:
                    ok = False
                    break
            if ok:
                curr_local.add(i)
        
        curr_val = sum(weights[indices[i]] for i in curr_local)
        best_local = curr_local
        best_local_val = curr_val
        
        # Local search on subgraph
        improved = True
        while improved and time.time() - start_time < timeout - 5:
            improved = False
            # Try swap
            for u in list(curr_local):
                for v in range(m):
                    if v in curr_local:
                        continue
                    # Check if v is compatible with curr_local - {u}
                    ok = True
                    for w in curr_local:
                        if w == u:
                            continue
                        if local_conflict[v][w]:
                            ok = False
                            break
                    if ok:
                        gain = weights[indices[v]] - weights[indices[u]]
                        if gain > 1e-12:
                            curr_local.remove(u)
                            curr_local.add(v)
                            curr_val += gain
                            if curr_val > best_local_val:
                                best_local_val = curr_val
                                best_local = set(curr_local)
                            improved = True
                            break
                if improved:
                    break
            
            # Try add
            if not improved:
                for v in range(m):
                    if v in curr_local:
                        continue
                    ok = True
                    for w in curr_local:
                        if local_conflict[v][w]:
                            ok = False
                            break
                    if ok:
                        curr_local.add(v)
                        curr_val += weights[indices[v]]
                        if curr_val > best_local_val:
                            best_local_val = curr_val
                            best_local = set(curr_local)
                        improved = True
                        break
        
        for i in best_local:
            initial_S.add(indices[i])
    
    # 3. Global Local Search
    # Start with initial_S
    S = set(initial_S)
    best_S = set(S)
    best_val = sum(weights[i] for i in S)
    
    def is_compatible(v, S_set):
        for u in S_set:
            if conflict[v][u]:
                return False
        return True
    
    def local_search_global(S_set, max_iters=10000):
        curr_val = sum(weights[i] for i in S_set)
        best_local_val = curr_val
        best_local_set = set(S_set)
        
        for _ in range(max_iters):
            if time.time() - start_time > timeout - 2:
                break
            
            improved = False
            # Try 1-swap
            # We want to find (u in S, v not in S) such that v is compatible with S\{u} and weight[v] > weight[u]
            # To speed up, we can iterate u in S and find best v.
            # Or iterate v not in S and find best u?
            
            # Let's try random restarts of the local search if stuck?
            # For now, deterministic sweep.
            
            # Optimization: Precompute list of candidates
            # This is O(|S| * (N-|S|)) which is 200*200 = 40k checks. Fast enough.
            
            best_swap_gain = -1.0
            best_u = None
            best_v = None
            
            # Iterate u in S
            for u in list(S_set):
                # Find best v not in S compatible with S\{u}
                # We want max weight[v]
                # Sort candidates by weight? No, just check all.
                for v in range(n):
                    if v in S_set:
                        continue
                    # Check compatibility with S\{u}
                    ok = True
                    for w in S_set:
                        if w == u:
                            continue
                        if conflict[v][w]:
                            ok = False
                            break
                    if ok:
                        gain = weights[v] - weights[u]
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_u = u
                            best_v = v
            
            if best_u is not None and best_swap_gain > 1e-12:
                S_set.remove(best_u)
                S_set.add(best_v)
                curr_val += best_swap_gain
                if curr_val > best_local_val:
                    best_local_val = curr_val
                    best_local_set = set(S_set)
                improved = True
                continue
            
            # Try add-only
            for v in range(n):
                if v in S_set:
                    continue
                if is_compatible(v, S_set):
                    S_set.add(v)
                    curr_val += weights[v]
                    if curr_val > best_local_val:
                        best_local_val = curr_val
                        best_local_set = set(S_set)
                    improved = True
                    break
            
            if not improved:
                # Try 2-swap? (remove u1, u2; add v1, v2)
                # This is expensive. O(|S|^2 * (N-|S|)^2).
                # Maybe just break and restart with random perturbation?
                # Or break local search.
                break
        
        return best_local_set, best_local_val
    
    best_S, best_val = local_search_global(S)
    
    # Additional restarts
    for _ in range(10):
        if time.time() - start_time > timeout - 10:
            break
        
        # Perturb best_S
        S_pert = set(best_S)
        # Remove 1-2 random elements
        for _ in range(random.randint(1, 3)):
            if S_pert:
                u = random.choice(list(S_pert))
                S_pert.remove(u)
        
        S_new, val_new = local_search_global(S_pert)
        if val_new > best_val:
            best_val = val_new
            best_S = S_new
    
    return [values[i] for i in best_S]
