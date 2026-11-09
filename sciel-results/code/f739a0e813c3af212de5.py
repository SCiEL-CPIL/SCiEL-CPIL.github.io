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

def find_clique_in_smooth_graph(target_primes, k=12, deadline=None, max_pool=2000, n_restarts=8):
    """
    Find k distinct numbers <= 10000 such that all pairwise sums are smooth w.r.t target_primes.
    Uses bitset-based clique search for efficiency with multiple random restarts.
    """
    if deadline and time.time() > deadline:
        return None
    
    # Generate all smooth candidates
    all_smooth = []
    for num in range(1, 10001):
        if is_smooth(num, target_primes):
            all_smooth.append(num)
    
    if len(all_smooth) < k:
        return None
    
    best_clique = None
    
    for restart in range(n_restarts):
        if deadline and time.time() > deadline:
            break
        
        # For first restart, use structured selection; for others, random
        if restart == 0:
            # Structured: take evenly spaced samples from the smooth list
            pool_size = min(max_pool, len(all_smooth))
            step = len(all_smooth) / pool_size
            candidates = [all_smooth[int(i * step)] for i in range(pool_size)]
            # Ensure distinctness
            candidates = list(dict.fromkeys(candidates))
        else:
            random.seed(42 + restart * 137)
            pool_size = min(max_pool, len(all_smooth))
            candidates = random.sample(all_smooth, pool_size)
        
        n_cand = len(candidates)
        if n_cand < k:
            continue
        
        # Build adjacency bitsets
        adj = [0] * n_cand
        
        for i in range(n_cand):
            ci = candidates[i]
            for j in range(i+1, n_cand):
                cj = candidates[j]
                if is_smooth(ci + cj, target_primes):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            
            if deadline and time.time() > deadline:
                return best_clique
        
        # Bron-Kerbosch with bitsets
        found_clique = []
        
        def bron_kerbosch(R, P, X):
            if deadline and time.time() > deadline:
                return False
            if len(R) >= k:
                found_clique.extend(R[:k])
                return True
            
            if P == 0 and X == 0:
                return False
            
            if P | X == 0:
                return False
            
            # Find pivot with max degree in P
            all_nodes = P | X
            pivot = -1
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
            
            if pivot < 0:
                return False
            
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
            
            return False
        
        initial_P = (1 << n_cand) - 1
        if bron_kerbosch([], initial_P, 0):
            clique = [candidates[i] for i in found_clique[:k]]
            if len(set(clique)) == k:
                return clique
            best_clique = clique
    
    return best_clique

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin
    
    best_score = 999
    best_arr = list(range(1, 13))  # fallback
    
    # List of first few primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Strategy 1: Try to find a clique in the smooth graph for small sets of primes.
    # If we find 12 numbers such that all pairwise sums are smooth w.r.t {2,3,5,7},
    # then the answer is 4.
    
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
        
        clique = find_clique_in_smooth_graph(target_primes, 12, deadline, max_pool=2000, n_restarts=8)
        if clique:
            # Verify
            score = count_distinct_primes(clique)
            if score < best_score:
                best_score = score
                best_arr = clique[:]
            # If score == k, we found an optimal solution for this k.
            # Since we are iterating k upwards, the first valid clique with score k is likely the global minimum
            # (assuming k=4 is possible). If k=4 works, we can return early.
            if score == k:
                return best_arr
                
        # Option 2: Random subsets of size k
        random.seed(42 + k)
        for _ in range(5):
            if time.time() > deadline:
                break
            selected = random.sample(primes[:15], k)
            target_primes = frozenset(selected)
            clique = find_clique_in_smooth_graph(target_primes, 12, deadline, max_pool=1500, n_restarts=4)
            if clique:
                score = count_distinct_primes(clique)
                if score < best_score:
                    best_score = score
                    best_arr = clique[:]
                if score == k:
                    return best_arr

    # Strategy 2: Fallback to heuristic search if no perfect smooth clique found
    # Use the previous logic: powers of 2, multiples, etc.
    
    # Powers of 2
    powers_of_2 = [1 << i for i in range(12)]
    if max(powers_of_2) <= 10000:
        score = count_distinct_primes(powers_of_2)
        if score < best_score:
            best_score = score
            best_arr = powers_of_2[:]
            
    # Multiples
    for mult in range(1, 20):
        if time.time() > deadline:
            return best_arr
        for start_k in range(1, 30):
            if time.time() > deadline:
                return best_arr
            arr = [mult * (start_k + i) for i in range(12)]
            if len(set(arr)) < 12 or max(arr) > 10000:
                continue
            score = count_distinct_primes(arr)
            if score < best_score:
                best_score = score
                best_arr = arr[:]
                
    # Local search refinement on best_arr
    if best_score > 4:  # Only if we haven't found the theoretical minimum for small k
        random.seed(42)
        current_arr = best_arr[:]
        current_score = best_score
        
        # Simulated Annealing
        temp = 10.0
        cooling = 0.99
        min_temp = 0.1
        
        while temp > min_temp and time.time() < deadline:
            idx = random.randint(0, 11)
            old_val = current_arr[idx]
            new_val = random.randint(1, 10000)
            
            if new_val in current_arr and new_val != old_val:
                continue
                
            current_arr[idx] = new_val
            score = count_distinct_primes(current_arr)
            
            delta = score - current_score
            if delta < 0 or random.random() < pow(2.718, -delta / temp):
                current_score = score
                if score < best_score:
                    best_score = score
                    best_arr = current_arr[:]
            else:
                current_arr[idx] = old_val
                
            temp *= cooling

    return best_arr

if __name__ == "__main__":
    res = construct()
    print(res)
    print(count_distinct_primes(res))
