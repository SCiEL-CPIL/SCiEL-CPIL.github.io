import time
import random
import math

# Precompute smallest prime factor for numbers up to 20000 (max sum = 2*10000)
MAX_SUM = 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:
        for j in range(i*i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

# Precompute prime factors for all numbers up to MAX_SUM
# factors[n] is a set of primes
factors = [set() for _ in range(MAX_SUM + 1)]
for i in range(2, MAX_SUM + 1):
    n = i
    while n > 1:
        p = spf[n]
        factors[i].add(p)
        while n % p == 0:
            n //= p

def get_prime_counts(arr):
    """
    Returns a dictionary mapping prime -> count of sums containing that prime.
    """
    counts = {}
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            for p in factors[s]:
                counts[p] = counts.get(p, 0) + 1
    return counts

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    # Initial solution: Powers of 2
    arr = [1 << i for i in range(12)]
    counts = get_prime_counts(arr)
    best_score = len(counts)
    
    # --- Strategy 1: Bitset-based clique search for 7-smooth sums ---
    # We look for a clique of size 12 in the graph where an edge exists
    # if the sum of the two nodes is 7-smooth (factors only in {2,3,5,7}).
    # This guarantees a prime count <= 4.
    
    # 1. Generate candidate pool: Numbers that are "compatible" with many others.
    # A number x is compatible with y if x+y is 7-smooth.
    # Let's pick a pool of numbers that are themselves 7-smooth or close to it,
    # or just a large random sample to build the graph.
    # Actually, to find a 12-clique, we need a dense enough graph.
    # If we pick all 7-smooth numbers <= 10000, their sums might not be 7-smooth.
    # Example: 2+3=5 (ok), 2+5=7 (ok), 3+5=8 (ok), 5+5=10 (ok).
    # 2+7=9 (ok), 3+7=10 (ok), 5+7=12 (ok), 7+7=14 (ok).
    # It seems sums of 7-smooth numbers are often 7-smooth?
    # Let's check: 2^a 3^b 5^c 7^d + 2^e 3^f 5^g 7^h.
    # This is not always true. 2+2=4 (ok). 2+6=8 (ok). 2+10=12 (ok).
    # 2+14=16 (ok). 2+18=20 (ok). 2+22=24 (ok).
    # 2+26=28 (ok). 2+30=32 (ok).
    # 3+3=6 (ok). 3+5=8 (ok).
    # It appears that the set of 7-smooth numbers is "sum-closed" in many small cases,
    # but not generally. However, if we restrict to a subset, we can find a clique.
    
    # Let's generate all 7-smooth numbers <= 10000.
    smooth_7 = set()
    def gen_smooth(limit, primes, idx, current):
        if current > limit:
            return
        if current > 1:
            smooth_7.add(current)
        for i in range(idx, len(primes)):
            gen_smooth(limit, primes, i, current * primes[i])
    
    gen_smooth(10000, [2, 3, 5, 7], 0, 1)
    
    # If we have enough 7-smooth numbers, we can try to find a clique among them.
    # But wait, the sum of two 7-smooth numbers isn't necessarily 7-smooth.
    # Example: 2^10 = 1024. 3^10 is too big.
    # Let's just build the graph on the set of 7-smooth numbers and find a clique.
    
    candidates_smooth = sorted(smooth_7)
    # If we don't have enough, add more numbers.
    # But 7-smooth numbers are quite dense.
    # Count: approx 10000 / (log 10000)^4 * constant.
    # Let's just use the set we have.
    
    # Build adjacency bitsets for the smooth candidates.
    # We need to map values to indices.
    if len(candidates_smooth) >= 12:
        # To save memory and time, limit the pool to a manageable size if it's huge.
        # But 7-smooth numbers up to 10000 are not that many.
        # Let's check count.
        # If it's too big for O(N^2) bitset construction, we might need to prune.
        # N ~ 1000? O(N^2) = 10^6 operations. Fast.
        
        n_cand = len(candidates_smooth)
        # Map value to index
        val_to_idx = {v: i for i, v in enumerate(candidates_smooth)}
        
        # Build adjacency bitsets.
        # adj[i] is a bitset (integer) where bit j is 1 if candidates[i] + candidates[j] is 7-smooth.
        # Note: i != j.
        adj = [0] * n_cand
        
        # To speed up, we can precompute a set of 7-smooth numbers for lookup.
        # We already have smooth_7 set.
        
        for i in range(n_cand):
            vi = candidates_smooth[i]
            # We only need to check j > i for symmetry, but bitsets are easier with full mask.
            # Let's just iterate j.
            mask = 0
            for j in range(n_cand):
                if i == j:
                    continue
                s = vi + candidates_smooth[j]
                if s in smooth_7:
                    mask |= (1 << j)
            adj[i] = mask
            
        # Now find a clique of size 12.
        # Use a simple backtracking or greedy with bitset intersection.
        # Bron-Kerbosch variant or just simple recursive search.
        # Given the density, a greedy approach might work, or a simple DFS.
        
        found_clique = None
        
        def find_clique(start_idx, current_clique, remaining_mask):
            # current_clique is a list of indices
            # remaining_mask is a bitset of candidates that can be added to current_clique
            if len(current_clique) == 12:
                return current_clique
            
            # Heuristic: pick the vertex with the highest degree in the remaining graph?
            # Or just pick the first one.
            # To be efficient, let's try vertices in order of index, but maybe prioritize those with high connectivity.
            
            # Get list of candidates from mask
            # For speed, we can iterate through set bits.
            # But simple loop is okay if N is small.
            
            # Optimization: If the number of bits in remaining_mask is less than needed, return None
            if bin(remaining_mask).count('1') < 12 - len(current_clique):
                return None
            
            # Try each candidate in remaining_mask
            # We can order them by degree in the subgraph induced by remaining_mask
            # But for now, let's just iterate.
            
            # To make it faster, let's try a few starting points.
            # We'll call this recursively.
            
            # Let's iterate through the bits in remaining_mask
            # We can use a trick to iterate set bits
            m = remaining_mask
            while m:
                # Get lowest set bit
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                m ^= lsb
                
                # Check if this vertex can be added
                # It must be connected to all vertices in current_clique
                # This is guaranteed by how we maintain remaining_mask?
                # If we maintain remaining_mask as the intersection of neighbors of all current_clique vertices,
                # then yes.
                
                # New remaining mask: intersection of adj[idx] and current remaining_mask
                new_mask = remaining_mask & adj[idx]
                # Remove idx from new_mask (can't pick same vertex twice)
                new_mask &= ~(1 << idx)
                
                res = find_clique(idx + 1, current_clique + [idx], new_mask)
                if res:
                    return res
            
            return None
        
        # Start search from a few promising vertices
        # Vertices with high degree are better.
        degrees = [bin(adj[i]).count('1') for i in range(n_cand)]
        # Sort indices by degree descending
        order = sorted(range(n_cand), key=lambda x: -degrees[x])
        
        # Try starting with top 50 vertices
        for start_idx in order[:50]:
            if time.time() > deadline:
                break
            # Initial mask: all vertices except start_idx
            init_mask = (1 << n_cand) - 1
            init_mask &= ~(1 << start_idx)
            
            # The first vertex must be connected to all subsequent ones.
            # So the subsequent ones must be in adj[start_idx].
            init_mask &= adj[start_idx]
            
            res = find_clique(start_idx, [start_idx], init_mask)
            if res:
                found_clique = res
                break
        
        if found_clique:
            # Convert indices back to values
            clique_vals = [candidates_smooth[i] for i in found_clique]
            # Verify distinctness and bounds
            if len(set(clique_vals)) == 12 and all(1 <= v <= 10000 for v in clique_vals):
                new_counts = get_prime_counts(clique_vals)
                if len(new_counts) < best_score:
                    best_score = len(new_counts)
                    arr = clique_vals
                    counts = new_counts
                    if best_score <= 4:
                        # Can't do better than 4 if we use primes {2,3,5,7}. 
                        # Could we do 3? {2,3,5} or {2,3,7} or {2,5,7} or {3,5,7}.
                        # If we find a 4, we might try to improve to 3 later, but 4 is a huge jump from ~9.
                        # Let's return early if we get a very good score?
                        # No, let's finish the other strategies if time permits, but 4 is likely optimal or near-optimal.
                        # Actually, if we get 4, we can return immediately to save time?
                        # The task says "minimize". 4 is very low.
                        # Let's return.
                        return arr

    # --- Strategy 2: Hill Climbing (refined from current program) ---
    # If the clique search didn't find a 12-clique (or didn't run), we fall back to hill climbing.
    # We can also use the smooth numbers as a candidate pool for hill climbing.
    
    # Add some random candidates to the pool
    smooth_candidates = set(candidates_smooth)
    random.seed(42)
    for _ in range(200):
        smooth_candidates.add(random.randint(1, 10000))
        
    candidates = list(smooth_candidates)
    
    # If we didn't find a clique, arr is still the initial powers of 2 or the best from previous steps.
    # If we found a clique, we already returned. So we only get here if no 12-clique was found in 7-smooth graph.
    
    while time.time() < deadline:
        improved = False
        random.shuffle(candidates)
        
        for i in range(12):
            old_val = arr[i]
            
            # Calculate primes to remove (from sums involving old_val)
            primes_to_remove = {}
            for j in range(12):
                if i == j: 
                    continue
                s = old_val + arr[j]
                for p in factors[s]:
                    primes_to_remove[p] = primes_to_remove.get(p, 0) + 1
            
            for c in candidates:
                if c == old_val:
                    continue
                # Check distinctness
                if c in arr:
                    if arr.index(c) != i:
                        continue
                    else:
                        pass 
                
                # Calculate primes to add (from sums involving c)
                primes_to_add = {}
                for j in range(12):
                    if i == j: 
                        continue
                    s = c + arr[j]
                    for p in factors[s]:
                        primes_to_add[p] = primes_to_add.get(p, 0) + 1
                
                # Update counts
                new_counts = counts.copy()
                
                # Remove old contributions
                for p, cnt in primes_to_remove.items():
                    new_counts[p] -= cnt
                    if new_counts[p] == 0:
                        del new_counts[p]
                
                # Add new contributions
                for p, cnt in primes_to_add.items():
                    if p in new_counts:
                        new_counts[p] += cnt
                    else:
                        new_counts[p] = cnt
                
                new_score = len(new_counts)
                
                if new_score < best_score:
                    best_score = new_score
                    counts = new_counts
                    arr[i] = c
                    improved = True
        
        if not improved:
            # Add new random candidates to explore
            for _ in range(20):
                candidates.append(random.randint(1, 10000))
            
    return arr
