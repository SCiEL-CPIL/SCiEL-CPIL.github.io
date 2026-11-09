import time
import random
from itertools import combinations

# Precompute smallest prime factor for numbers up to 20000 (max sum = 2*10000)
MAX_SUM = 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:
        for j in range(i*i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

# Precompute prime factors for all numbers up to MAX_SUM
factors_cache = [set() for _ in range(MAX_SUM + 1)]
for i in range(2, MAX_SUM + 1):
    n = i
    while n > 1:
        p = spf[n]
        factors_cache[i].add(p)
        while n % p == 0:
            n //= p

def count_distinct_primes(arr):
    """Count distinct prime factors across all pairwise sums of distinct elements."""
    all_primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            all_primes |= factors_cache[s]
    return len(all_primes)

def is_smooth(n, target_primes):
    """Check if n is smooth with respect to target_primes."""
    if n <= 1:
        return True
    return factors_cache[n].issubset(target_primes)

def find_clique_in_smooth_graph(target_primes, k=12, deadline=None, pool_size=300, seed=None):
    """
    Find k distinct numbers <= 10000 such that all pairwise sums are smooth w.r.t target_primes.
    Uses bitset-based clique search for efficiency.
    """
    if deadline and time.time() > deadline:
        return None
    
    if seed is not None:
        random.seed(seed)
    
    # Generate candidates: numbers <= 10000 that are "compatible"
    # Pick numbers that are themselves smooth w.r.t target_primes first
    candidates = []
    for num in range(1, 10001):
        if is_smooth(num, target_primes):
            candidates.append(num)
    
    if len(candidates) < k:
        return None
    
    # Limit pool for feasibility
    if len(candidates) > pool_size:
        random.shuffle(candidates)
        candidates = candidates[:pool_size]
    
    n_cand = len(candidates)
    adj = [0] * n_cand
    
    # Build adjacency: edge if sum is smooth
    for i in range(n_cand):
        for j in range(i+1, n_cand):
            if is_smooth(candidates[i] + candidates[j], target_primes):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
        if deadline and time.time() > deadline:
            return None

    # Bron-Kerbosch with bitsets to find clique of size k
    found_clique = []
    
    def bron_kerbosch(R, P, X):
        if deadline and time.time() > deadline:
            return False
        if len(R) >= k:
            found_clique.extend(R[:k])
            return True
        
        if P == 0 and X == 0:
            return False
            
        all_nodes = P | X
        if all_nodes == 0:
            return False
            
        # Pivot selection: pick u in P ∪ X with max degree in P
        pivot = 0
        max_deg = -1
        temp = all_nodes
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            deg = (adj[idx] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = idx
            temp ^= lsb
            
        # Iterate over P \ N(pivot)
        candidates_to_try = P & ~(adj[pivot])
        
        while candidates_to_try:
            lsb = candidates_to_try & (-candidates_to_try)
            v = lsb.bit_length() - 1
            candidates_to_try ^= lsb
            
            R.append(v)
            if bron_kerbosch(R, P & adj[v], X & adj[v]):
                R.pop()
                return True
            R.pop()
            P ^= lsb
            X |= lsb
            if deadline and time.time() > deadline:
                return False
            
        return False

    initial_P = (1 << n_cand) - 1
    if bron_kerbosch([], initial_P, 0):
        return [candidates[i] for i in found_clique]
        
    return None

def local_search_minimize(arr, target_primes, deadline, max_steps=5000, pool=None):
    """
    Local search to minimize the number of distinct primes NOT in target_primes
    appearing in pairwise sums. If target_primes is None, minimize total distinct primes.
    """
    if pool is None:
        pool = list(range(1, 10001))
    
    current_arr = arr[:]
    current_score = count_distinct_primes(current_arr) if target_primes is None else _count_bad_primes(current_arr, target_primes)
    
    best_score = current_score
    best_arr = current_arr[:]
    
    steps = 0
    while time.time() < deadline and steps < max_steps:
        steps += 1
        idx = random.randint(0, len(current_arr)-1)
        old_val = current_arr[idx]
        new_val = random.choice(pool)
        if new_val in current_arr and new_val != old_val:
            continue
        current_arr[idx] = new_val
        new_score = count_distinct_primes(current_arr) if target_primes is None else _count_bad_primes(current_arr, target_primes)
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_arr = current_arr[:]
        else:
            current_arr[idx] = old_val
    
    return best_arr, best_score

def _count_bad_primes(arr, target_primes):
    """Count distinct primes NOT in target_primes across all pairwise sums."""
    bad_primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            for p in factors_cache[s]:
                if p not in target_primes:
                    bad_primes.add(p)
    return len(bad_primes)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_score = 999
    best_arr = list(range(1, 13))
    
    # Strategy: Try to find a clique in the smooth graph for small sets of primes.
    # If we find 12 numbers such that all pairwise sums are smooth w.r.t {2,3,5,7},
    # then the answer is 4.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Try k=4 first: {2,3,5,7}
    # Then k=5: {2,3,5,7,11}
    # etc.
    
    for k in range(4, 9):
        if time.time() > deadline:
            break
            
        # Try a few combinations of primes of size k
        # We prefer smaller primes.
        
        # Option 1: First k primes
        target_primes = frozenset(primes[:k])
        
        # Try multiple random seeds/pools for clique search
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
            seed = 42 + k * 100 + attempt
            clique = find_clique_in_smooth_graph(target_primes, 12, deadline, pool_size=400, seed=seed)
            if clique:
                score = count_distinct_primes(clique)
                if score < best_score:
                    best_score = score
                    best_arr = clique[:]
                if score == k:
                    return best_arr
                found = True
        
        # Option 2: Random subsets of size k
        random.seed(42 + k * 10)
        for _ in range(3):
            if time.time() > deadline:
                break
            selected = random.sample(primes[:15], k)
            target_primes = frozenset(selected)
            clique = find_clique_in_smooth_graph(target_primes, 12, deadline, pool_size=400)
            if clique:
                score = count_distinct_primes(clique)
                if score < best_score:
                    best_score = score
                    best_arr = clique[:]
                if score == k:
                    return best_arr

    # Strategy 2: Modular arithmetic constraint satisfaction
    # Fix a small prime set P and restrict candidate numbers to residues modulo M
    # that guarantee pairwise sums are P-smooth
    
    for target_primes in [frozenset([2,3,5,7]), frozenset([2,3,5,7,11])]:
        if time.time() > deadline:
            break
        
        M = 1
        for p in target_primes:
            M *= p
        
        # Find residues r where numbers ≡ r mod M tend to have smooth sums
        # For a residue r, check if there exist 12 numbers ≡ r mod M with smooth pairwise sums
        residue_candidates = {}
        for num in range(1, 10001):
            r = num % M
            if r not in residue_candidates:
                residue_candidates[r] = []
            residue_candidates[r].append(num)
        
        # For each residue, check if we can find a clique within that residue class
        # Numbers in the same residue class have sums ≡ 2r mod M
        # Check if 2r mod M is "compatible" with smoothness
        
        for r, nums in residue_candidates.items():
            if time.time() > deadline:
                break
            if len(nums) < 12:
                continue
            
            # Check if a typical sum from this residue is smooth
            # Take first few pairs
            test_smooth = True
            for i in range(min(5, len(nums))):
                for j in range(i+1, min(5, len(nums))):
                    if not is_smooth(nums[i] + nums[j], target_primes):
                        test_smooth = False
                        break
                if not test_smooth:
                    break
            
            if not test_smooth:
                continue
            
            # Try to find a clique within this residue class
            pool = nums[:200] if len(nums) > 200 else nums
            random.shuffle(pool)
            pool = pool[:min(200, len(pool))]
            
            n_pool = len(pool)
            if n_pool < 12:
                continue
            
            adj = [0] * n_pool
            for i in range(n_pool):
                for j in range(i+1, n_pool):
                    if is_smooth(pool[i] + pool[j], target_primes):
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Quick greedy clique search
            clique = []
            remaining = set(range(n_pool))
            for _ in range(12):
                if not remaining:
                    break
                # Pick node with max degree in remaining
                best_node = None
                best_deg = -1
                for node in remaining:
                    deg = (adj[node] & (1 << len(remaining) - 1) if False else sum(1 for m in remaining if m != node and (adj[node] >> m) & 1))
                    if deg > best_deg:
                        best_deg = deg
                        best_node = node
                if best_node is None:
                    break
                clique.append(best_node)
                remaining.discard(best_node)
                # Only keep neighbors of best_node
                remaining = {m for m in remaining if (adj[best_node] >> m) & 1}
            
            if len(clique) >= 12:
                candidate = [pool[i] for i in clique[:12]]
                score = count_distinct_primes(candidate)
                if score < best_score:
                    best_score = score
                    best_arr = candidate[:]
                if score == len(target_primes):
                    return best_arr

    # Strategy 3: General local search with good initial guess
    if best_score == 999:
        best_arr = list(range(1, 13))
        best_score = count_distinct_primes(best_arr)
    
    # Final improvement: local search from best found
    random.seed(42)
    improved_arr, improved_score = local_search_minimize(best_arr, None, deadline, max_steps=10000)
    if improved_score < best_score:
        best_score = improved_score
        best_arr = improved_arr[:]
    
    return best_arr
