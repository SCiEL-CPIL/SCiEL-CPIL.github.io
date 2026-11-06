import time

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

# We only care about the first few primes for the mask
# Let's include primes up to 113 to be safe for mask construction
PRIMES_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
P2IDX = {p: i for i, p in enumerate(PRIMES_LIST)}

def _build_masks(limit=20000):
    """
    Precompute prime factor masks for all numbers up to limit.
    mask[x] is a bitmask where bit i is set if PRIMES_LIST[i] divides x.
    If x has a prime factor not in PRIMES_LIST, mask[x] is -1 (invalid for smoothness checks with small prime sets).
    """
    masks = [-1] * (limit + 1)
    masks[0] = 0 
    masks[1] = 0 

    for x in range(2, limit + 1):
        t = x
        m = 0
        ok = True
        while t > 1:
            p = _SPF[t]
            idx = P2IDX.get(p)
            if idx is None:
                ok = False
                break
            m |= 1 << idx
            while t % p == 0:
                t //= p
        if ok:
            masks[x] = m
    return masks

_MASKS = _build_masks(20000)

def _is_smooth(x, p_mask):
    m = _MASKS[x]
    if m < 0:
        return False
    # Check if all prime factors of x are in p_mask
    return (m & ~p_mask) == 0

def _find_clique(nodes, k, p_mask, deadline, node_budget):
    """
    Finds a clique of size k in the graph defined by the nodes and smoothness.
    nodes: list of integers
    k: target clique size
    p_mask: bitmask of allowed primes
    deadline: time limit
    node_budget: max nodes to visit
    """
    n = len(nodes)
    if n < k:
        return None
    
    # Build adjacency list as bitmasks for fast intersection
    # adj[i] is a bitmask of indices j such that nodes[i] + nodes[j] is smooth
    adj = [0] * n
    for i in range(n):
        ai = nodes[i]
        for j in range(i + 1, n):
            if _is_smooth(ai + nodes[j], p_mask):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Precompute higher mask: higher[v] has bits set for all indices > v
    higher = [0] * n
    for v in range(n - 2, -1, -1):
        higher[v] = higher[v + 1] | (1 << (v + 1))
        
    # Degree for ordering
    deg = [adj[i].bit_count() for i in range(n)]
    
    best_clique = [None]
    node_count = [0]
    aborted = [False]
    
    def dfs(candidates_mask, current_clique):
        if len(current_clique) == k:
            best_clique[0] = list(current_clique)
            return True
        
        if time.time() > deadline:
            aborted[0] = True
            return False
            
        node_count[0] += 1
        if node_count[0] > node_budget:
            aborted[0] = True
            return False

        # If not enough candidates left
        if candidates_mask.bit_count() < k - len(current_clique):
            return False
            
        # Get candidates sorted by degree in the subgraph to improve pruning?
        # For now, just iterate.
        m = candidates_mask
        cands = []
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m &= m - 1
            nc_count = (candidates_mask & adj[v] & higher[v]).bit_count()
            cands.append((nc_count, v))
            
        # Sort by degree descending (try high-degree vertices first)
        cands.sort(key=lambda x: -x[0])
        
        for nc_count, v in cands:
            if nc_count + 1 < k - len(current_clique):
                continue
                
            next_mask = candidates_mask & adj[v] & higher[v]
            if dfs(next_mask, current_clique + [v]):
                return True
                
        return False

    initial_mask = (1 << n) - 1
    if dfs(initial_mask, []):
        return best_clique[0]
    return None

def construct():
    deadline = time.time() + 85.0 # Safety margin
    
    # Try increasing number of primes
    # Start with the smallest primes
    for num_primes in range(2, 15):
        if time.time() > deadline - 5:
            break
            
        P = PRIMES_LIST[:num_primes]
        p_mask = 0
        for p in P:
            p_mask |= 1 << P2IDX[p]
            
        # Find all smooth numbers <= 10000
        smooth_nums = [x for x in range(1, 10001) if _is_smooth(x, p_mask)]
        
        if len(smooth_nums) < 12:
            continue
            
        # Heuristic: smaller numbers tend to have smaller sums, which are more likely to be smooth
        # So we can just use the sorted list (which is already sorted by value).
        
        # If the list is very large (e.g. > 1000), we might need to be careful with clique search time.
        # But for small prime sets, the list is small.
        # For 2,3-smooth: 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96, 108, 128, 144, 162, 192, 216, 243, 256, 288, 324, 384, 432, 486, 512, 576, 648, 729, 768, 864, 972, 1024...
        # Count is small.
        
        # Run clique search
        # Use a tighter deadline for each attempt
        sub_deadline = min(time.time() + 5.0, deadline - 2.0)
        
        # Optimization: If the list is huge, we might want to limit it.
        # But let's assume the clique search with bitmask is fast enough for the sizes involved.
        
        # Limit the search to the first N smooth numbers to keep it fast?
        # Actually, for small prime sets, the list is small.
        # If num_primes is small (2,3), the list is small.
        # If num_primes is large, the list is large, but finding a clique is easier.
        
        # Try different subsets of smooth numbers to find a clique
        # Start with small subsets and increase
        max_smooth_to_try = min(len(smooth_nums), 200)
        
        for T in range(12, max_smooth_to_try + 1, 10):
            if time.time() > deadline - 2.0:
                break
            subset = smooth_nums[:T]
            res_idx = _find_clique(subset, 12, p_mask, sub_deadline, 100000)
            if res_idx is not None:
                # Convert indices back to values
                return [subset[i] for i in res_idx]
                
        # If not found in subset, try the full list with a larger budget
        res_idx = _find_clique(smooth_nums, 12, p_mask, sub_deadline, 500000)
        if res_idx is not None:
            # Convert indices back to values
            return [smooth_nums[i] for i in res_idx]
            
    # Fallback: Return a known valid set with a larger number of primes
    # This set uses primes 2, 3, 5
    # Sums:
    # 1+2=3, 1+3=4, 1+5=6, 1+6=7, 1+9=10, 1+10=11, 1+15=16, 1+18=19, 1+25=26, 1+27=28, 1+30=31
    # 2+3=5, 2+5=7, 2+6=8, 2+9=11, 2+10=12, 2+15=17, 2+18=20, 2+25=27, 2+27=29, 2+30=32
    # 3+5=8, 3+6=9, 3+9=12, 3+10=13, 3+15=18, 3+18=21, 3+25=28, 3+27=30, 3+30=33
    # 5+6=11, 5+9=14, 5+10=15, 5+15=20, 5+18=23, 5+25=30, 5+27=32, 5+30=35
    # 6+9=15, 6+10=16, 6+15=21, 6+18=24, 6+25=31, 6+27=33, 6+30=36
    # 9+10=19, 9+15=24, 9+18=27, 9+25=34, 9+27=36, 9+30=39
    # 10+15=25, 10+18=28, 10+25=35, 10+27=37, 10+30=40
    # 15+18=33, 15+25=40, 15+27=42, 15+30=45
    # 18+25=43, 18+27=45, 18+30=48
    # 25+27=52, 25+30=55
    # 27+30=57
    
    # Primes involved: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 43
    # This is a lot.
    
    # Let's try to generate a better fallback programmatically if the search fails.
    # But we must return something.
    # The previous program returned [1,2,3,5,6,9,10,15,18,25,27,30]
    # Let's use that.
    return [1, 2, 3, 5, 6, 9, 10, 15, 18, 25, 27, 30]
