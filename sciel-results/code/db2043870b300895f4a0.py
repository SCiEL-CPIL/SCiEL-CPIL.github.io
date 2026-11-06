import time
import itertools

_START = time.time()

def _spf_sieve(limit):
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

# We need to factor sums up to 20000
_SPF = _spf_sieve(20000)

# Primes to consider for the set P
# We start with small primes as they generate more smooth numbers
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

def _is_smooth(x, p_set_mask):
    """
    Check if x is smooth with respect to the primes set by p_set_mask.
    p_set_mask is an integer bitmask where bit i is set if _PRIMES[i] is in the set.
    """
    if x <= 1:
        return True
    t = x
    while t > 1:
        p = _SPF[t]
        # Find the index of p in _PRIMES
        # Since _PRIMES is small and sorted, we can use a dict for lookup
        idx = _P2IDX.get(p)
        if idx is None:
            return False
        if not (p_set_mask & (1 << idx)):
            return False
        while t % p == 0:
            t //= p
    return True

# Precompute index for primes for fast lookup
_P2IDX = {p: i for i, p in enumerate(_PRIMES)}

def _find_clique(candidates, adj, k, deadline):
    """
    Find a clique of size k in the graph defined by candidates and adj.
    adj[i] is a set of indices j such that candidates[i] and candidates[j] form a smooth sum.
    """
    n = len(candidates)
    if n < k:
        return None
    
    # Order candidates by degree descending to help pruning
    # deg[i] = len(adj[i])
    degs = [len(a) for a in adj]
    order = sorted(range(n), key=lambda i: -degs[i])
    
    # Map old indices to new indices in ordered list? 
    # Or just use the original indices but iterate in order?
    # For simplicity in backtracking, let's stick to original indices 
    # but choose the next vertex from the available set using a heuristic.
    
    # To speed up, we can precompute the adjacency as bitsets if n is small, 
    # but for n ~ 1000, sets are fine.
    
    # Backtracking with forward checking
    # path: list of indices in current clique
    # avail: set of indices that can be added to the clique (connected to all in path)
    
    best = []
    
    def dfs(path, avail):
        if time.time() > deadline:
            return None
        
        if len(path) == k:
            return list(path)
            
        # Pruning: if not enough candidates left
        if len(avail) < k - len(path):
            return None
            
        # Heuristic: pick a vertex with high degree within 'avail'
        # To avoid O(n^2) scanning every time, we can just iterate.
        # For better performance, we can order the candidates in 'avail' by their degree in the subgraph induced by 'avail'.
        # This is expensive to compute dynamically. 
        # A simpler heuristic: try vertices with high global degree first.
        
        # We can also use a "pivot" strategy from Bron-Kerbosch to reduce recursion, 
        # but for finding *one* clique of fixed size, simple backtracking with good ordering is often sufficient.
        
        # Let's sort avail by global degree descending
        sorted_avail = sorted(avail, key=lambda i: -degs[i])
        
        for v in sorted_avail:
            # Check if v is still in avail (it is, by definition of the loop variable from the set)
            # Add v to path
            new_path = path + [v]
            
            # Calculate new available set:
            # Must be in 'avail' (already connected to previous path vertices)
            # AND must be connected to v
            new_avail = avail.intersection(adj[v])
            
            # Remove v from new_avail (distinct indices)
            new_avail.discard(v)
            
            res = dfs(new_path, new_avail)
            if res is not None:
                return res
                
        return None

    initial_avail = set(range(n))
    res = dfs([], initial_avail)
    
    if res is not None:
        # Verify
        for i in range(k):
            for j in range(i+1, k):
                u, v = res[i], res[j]
                if not (u in adj[v]):
                    return None
        return res
        
    return None

def construct():
    deadline = time.time() + 80.0 # 10s safety margin
    
    # Try increasing sizes of prime sets
    # Start with k=3
    for k in range(3, 12):
        if time.time() > deadline - 10:
            break
            
        # Generate combinations of k primes from _PRIMES
        # To save time, we prioritize combinations with smaller primes
        # We can just iterate through all combinations of the first, say, 10 primes
        # But we should start with the smallest ones.
        
        # Optimization: Instead of all combinations, we can try 
        # P = first k primes, then swap one, etc.
        # But for now, let's just iterate combinations of the first 15 primes.
        
        # To be efficient, we can just try the set of the first k primes first.
        # If that fails, we might need to try others, but usually the smallest primes work best
        # because they generate the most smooth numbers.
        
        # Let's try the set of the first k primes.
        P = _PRIMES[:k]
        p_mask = 0
        for p in P:
            idx = _P2IDX[p]
            p_mask |= (1 << idx)
            
        # Generate candidates: P-smooth numbers <= 10000
        candidates = []
        for x in range(1, 10001):
            if _is_smooth(x, p_mask):
                candidates.append(x)
                
        if len(candidates) < 12:
            continue
            
        # Build adjacency list
        # adj[i] is a set of j such that candidates[i] + candidates[j] is P-smooth
        n = len(candidates)
        adj = [set() for _ in range(n)]
        
        # To speed up, we can break early if we have enough connections? No, we need the full graph for clique search.
        # However, we can limit the candidate pool if it's too large.
        # For k=3, n is small (~100). For k=4, n is larger (~1000).
        # If n > 2000, we might restrict to the smallest 2000 P-smooth numbers,
        # as small numbers are more likely to have smooth sums.
        
        if n > 2000:
            # Sort candidates and take the smallest 2000
            candidates = sorted(candidates)[:2000]
            n = len(candidates)
            adj = [set() for _ in range(n)]
            
        for i in range(n):
            ci = candidates[i]
            for j in range(i + 1, n):
                cj = candidates[j]
                s = ci + cj
                if _is_smooth(s, p_mask):
                    adj[i].add(j)
                    adj[j].add(i)
                    
        # Find clique
        # Use a sub-deadline
        sub_deadline = min(time.time() + 5.0, deadline - 10)
        res_idx = _find_clique(candidates, adj, 12, sub_deadline)
        
        if res_idx is not None:
            return [candidates[i] for i in res_idx]
            
        # If the first k primes didn't work, we could try other combinations,
        # but given the time constraint, we move to k+1.
        # The probability that a larger prime set works is higher, but the number of candidates grows.
        # However, the "smoothness" constraint becomes looser, making it easier to find smooth sums.
        
    # Fallback: Return a known valid set (even if not optimal)
    # The set [1, 2, 3, 5, 6, 9, 10, 15, 18, 25, 27, 30]
    # Sums:
    # 1+2=3, 1+3=4, 1+5=6, 1+6=7, 1+9=10, 1+10=11, 1+15=16, 1+18=19, 1+25=26, 1+27=28, 1+30=31
    # 2+3=5, 2+5=7, 2+6=8, 2+9=11, 2+10=12, 2+15=17, 2+18=20, 2+25=27, 2+27=29, 2+30=32
    # Primes involved: 2,3,5,7,11,13,17,19,23,29,31,37,43?
    # Let's check 1+25=26=2*13. 13 is prime.
    # 1+30=31. 31 is prime.
    # 2+15=17.
    # 2+18=20.
    # 2+27=29.
    # 3+5=8.
    # 3+6=9.
    # 3+9=12.
    # 3+10=13.
    # 3+15=18.
    # 3+18=21=3*7.
    # 3+25=28.
    # 3+27=30.
    # 3+30=33=3*11.
    # 5+6=11.
    # 5+9=14.
    # 5+10=15.
    # 5+15=20.
    # 5+18=23.
    # 5+25=30.
    # 5+27=32.
    # 5+30=35=5*7.
    # 6+9=15.
    # 6+10=16.
    # 6+15=21.
    # 6+18=24.
    # 6+25=31.
    # 6+27=33.
    # 6+30=36.
    # 9+10=19.
    # 9+15=24.
    # 9+18=27.
    # 9+25=34=2*17.
    # 9+27=36.
    # 9+30=39=3*13.
    # 10+15=25.
    # 10+18=28.
    # 10+25=35.
    # 10+27=37.
    # 10+30=40.
    # 15+18=33.
    # 15+25=40.
    # 15+27=42.
    # 15+30=45.
    # 18+25=43.
    # 18+27=45.
    # 18+30=48.
    # 25+27=52.
    # 25+30=55.
    # 27+30=57.
    
    # Primes: 2,3,5,7,11,13,17,19,23,29,31,37,43.
    # Count: 13.
    # This is a valid fallback.
    
    return [1, 2, 3, 5, 6, 9, 10, 15, 18, 25, 27, 30]
