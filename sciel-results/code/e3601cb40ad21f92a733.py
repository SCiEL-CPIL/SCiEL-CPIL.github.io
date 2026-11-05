import time
from itertools import combinations
from collections import Counter

# Precompute prime factors for all numbers up to 20000
MAX_SUM = 20000
_prime_factors_cache = [set() for _ in range(MAX_SUM + 1)]
for s in range(2, MAX_SUM + 1):
    n = s
    d = 2
    while d * d <= n:
        if n % d == 0:
            _prime_factors_cache[s].add(d)
            while n % d == 0:
                n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        _prime_factors_cache[s].add(n)

def get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = get_primes(20000)
PRIME_SET = set(PRIMES)

def count_distinct_primes(nums):
    all_primes = set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            s = nums[i] + nums[j]
            all_primes.update(_prime_factors_cache[s])
    return len(all_primes)

def find_clique_in_graph(nodes, adj, k):
    """Find a clique of size k in the graph. Returns the clique or None."""
    if len(nodes) < k:
        return None
    
    # Order nodes by degree for better pruning
    nodes = sorted(nodes, key=lambda x: len(adj[x]), reverse=True)
    
    clique = []
    
    def backtrack(remaining):
        if len(clique) == k:
            return True
        if len(remaining) < k - len(clique):
            return False
        
        for idx, node in enumerate(remaining):
            if all(node in adj[c] for c in clique):
                clique.append(node)
                if backtrack(remaining[idx+1:]):
                    return True
                clique.pop()
        
        return False
    
    if backtrack(nodes):
        return clique[:]
    return None

def generate_smooth_numbers(max_val, primes):
    """Generate all numbers <= max_val whose prime factors are in primes."""
    nums = {1}
    for p in primes:
        current = list(nums)
        for x in current:
            y = x
            while y * p <= max_val:
                y *= p
                nums.add(y)
                if y > max_val:
                    break
    return sorted(nums)

def construct():
    start_time = time.time()
    deadline = 85.0
    
    best_set = list(range(1, 13))
    best_score = count_distinct_primes(best_set)
    
    # Strategy: Find 12 numbers such that all pairwise sums have prime factors from a small set S
    # We search for the smallest S that allows a K12 clique in the compatibility graph
    
    # Start with small prime sets and expand
    candidate_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    
    found = False
    for num_primes in range(2, len(candidate_primes) + 1):
        if time.time() > deadline - 5:
            break
        
        # Try all combinations of num_primes from candidate_primes
        # Start with the smallest primes first
        primes_list = candidate_primes[:num_primes]
        prime_set = set(primes_list)
        
        # Generate smooth numbers
        smooth_nums = generate_smooth_numbers(10000, primes_list)
        
        # For efficiency, limit candidates
        max_candidates = min(len(smooth_nums), 300)
        candidates = smooth_nums[:max_candidates]
        
        if len(candidates) < 12:
            continue
        
        # Build compatibility graph
        adj = {a: set() for a in candidates}
        for i in range(len(candidates)):
            a = candidates[i]
            for j in range(i + 1, len(candidates)):
                b = candidates[j]
                s = a + b
                if s <= MAX_SUM:
                    if _prime_factors_cache[s].issubset(prime_set):
                        adj[a].add(b)
                        adj[b].add(a)
        
        # Find clique of size 12
        clique = find_clique_in_graph(candidates, adj, 12)
        
        if clique:
            score = count_distinct_primes(clique)
            if score < best_score:
                best_score = score
                best_set = clique[:]
                found = True
                if best_score <= num_primes:
                    return best_set[:]
        
        # If we found a good solution, we can stop early
        if found and best_score <= 4:
            return best_set[:]
    
    # Local search from a good starting point
    if time.time() < deadline - 10:
        # Start with smooth numbers
        smooth_nums = generate_smooth_numbers(10000, [2, 3, 5, 7, 11, 13, 17, 19])
        if len(smooth_nums) >= 12:
            current = smooth_nums[:12]
        else:
            current = smooth_nums + list(range(1, 13 - len(smooth_nums)))
        
        current_score = count_distinct_primes(current)
        if current_score < best_score:
            best_score = current_score
            best_set = current[:]
        
        # Local search
        for _ in range(5000):
            if time.time() > deadline - 5:
                break
            
            i = __import__('random').randint(0, 11)
            old_val = current[i]
            new_val = __import__('random').choice(smooth_nums)
            if new_val in current:
                continue
            
            current[i] = new_val
            new_score = count_distinct_primes(current)
            
            if new_score <= current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
            else:
                current[i] = old_val
    
    return best_set

if __name__ == "__main__":
    start = time.time()
    res = construct()
    print(f"Result: {res}")
    print(f"Score: {count_distinct_primes(res)}")
    print(f"Time: {time.time() - start}")
