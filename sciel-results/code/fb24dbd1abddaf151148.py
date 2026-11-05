import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    # Parameters
    n = 12
    max_val = 10000
    
    # Helper: Generate p-smooth numbers up to limit
    def generate_smooth(limit, primes):
        smooth = set([1])
        for p in primes:
            new_smooth = set()
            for s in smooth:
                val = s
                while val * p <= limit:
                    val *= p
                    new_smooth.add(val)
            smooth.update(new_smooth)
        return smooth

    # Helper: Check if a set of numbers has all pairwise sums in smooth_set
    def is_valid_set(nums, smooth_set):
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i] + nums[j] not in smooth_set:
                    return False
        return True

    # Helper: Find a clique of size k in the graph defined by smooth_set
    # Vertices: 1..max_val
    # Edge (u, v) if u+v in smooth_set
    def find_clique(k, smooth_set, timeout=10.0):
        # Precompute neighbors
        # neighbors[i] = list of j such that i+j in smooth_set
        # To save memory/time, we only need to check candidates during search
        # But for speed, precomputing a set of valid sums is key.
        
        # We will use a recursive backtracking search.
        # To make it fast, we order candidates by "potential".
        # A simple strategy: Pick a starting point, then greedily add compatible nodes.
        # If stuck, backtrack.
        
        # Optimization: Only consider numbers up to max_val.
        # We can restrict the search space to numbers that have high "smooth-sum" density.
        
        # Let's try a randomized local search first for speed, then exact if needed?
        # No, exact backtracking with good pruning is better for small k.
        
        # Sort all numbers by how many smooth sums they can form?
        # degree(x) = count of y in [1, max_val] such that x+y in smooth_set
        degrees = [0] * (max_val + 1)
        for x in range(1, max_val + 1):
            for y in range(1, max_val + 1 - x):
                if x + y in smooth_set:
                    degrees[x] += 1
        
        # Candidate pool: numbers with high degree
        # We don't need all 10000, maybe top 1000?
        # But 1000 is still large.
        
        # Let's use a simple recursive search with a time limit.
        best_found = None
        
        def backtrack(current, candidates):
            nonlocal best_found
            if time.time() > timeout:
                return
            if len(current) == k:
                best_found = current.copy()
                return True
            if not candidates:
                return False
            
            # Heuristic: Pick the candidate with the most connections to 'current'
            # Or just pick the first one?
            # Better: Sort candidates by degree within the remaining graph?
            # For simplicity, we just iterate.
            
            for i in range(len(candidates)):
                x = candidates[i]
                # Check if x is compatible with all in current
                compatible = True
                for c in current:
                    if x + c not in smooth_set:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                # Recurse
                # New candidates must be > x (to avoid duplicates and maintain order)
                # And compatible with x
                new_candidates = [c for c in candidates[i+1:] if x + c in smooth_set]
                
                if backtrack(current + [x], new_candidates):
                    return True
        # This naive backtracking might be too slow if the first few branches are dead ends.
        # We need to be smarter.
        
        # Alternative: Randomized Restart Backtracking
        # Pick random starting nodes.
        
        # Let's try a different approach:
        # 1. Generate a large set of "promising" numbers (high degree).
        # 2. Use a greedy algorithm to build a clique, with randomization.
        
        promising = sorted(range(1, max_val + 1), key=lambda x: degrees[x], reverse=True)
        # Take top 500 candidates
        candidates = promising[:500]
        
        # Try to find clique in this subgraph
        # Use a simple recursive search with memoization or just backtracking
        # Since k=12 is small, and graph is dense-ish among top nodes, it should be fast.
        
        def search_clique_in_subgraph(candidates, k):
            if len(candidates) < k:
                return None
            if k == 1:
                return [candidates[0]]
            
            # Pick a pivot?
            # Let's just try all starting points
            for i in range(len(candidates)):
                x = candidates[i]
                # Find neighbors of x in candidates[i+1:]
                neighbors = [c for c in candidates[i+1:] if x + c in smooth_set]
                if len(neighbors) < k - 1:
                    continue
                rest = search_clique_in_subgraph(neighbors, k - 1)
                if rest:
                    return [x] + rest
            return None
        
        # This is still potentially slow.
        # Let's add a timeout to the whole construct function.
        
        # Given the complexity, let's just run a randomized local search which is robust.
        # Objective: Minimize number of distinct primes in sums.
        # If we hit 0 "extra" primes (i.e., all sums are 3-smooth), we are done.
        
        return None # Placeholder, logic below

    # --- Actual Logic ---
    
    # 1. Try to find a 2-prime solution (3-smooth sums)
    smooth_2 = generate_smooth(20000, [2, 3])
    
    # Use a randomized greedy approach with backtracking
    # We'll try to build a set of 12 numbers.
    
    def try_find_3_smooth():
        # Precompute valid partners for each number in a restricted range?
        # Let's just use random restarts.
        
        for _ in range(500): # 500 restarts
            if time.time() > deadline:
                break
            # Start with a random number
            current = [random.randint(1, max_val)]
            # Greedily add compatible numbers
            # To avoid getting stuck, we can shuffle candidates
            candidates = list(range(1, max_val + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                if x in current:
                    continue
                if all(x + c in smooth_2 for c in current):
                    current.append(x)
                    if len(current) == n:
                        return current
            # If we didn't reach n, restart
            
        return None

    sol = try_find_3_smooth()
    if sol:
        return sol

    # 2. Try to find a 3-prime solution (5-smooth sums)
    smooth_3 = generate_smooth(20000, [2, 3, 5])
    
    def try_find_5_smooth():
        for _ in range(500):
            if time.time() > deadline:
                break
            current = [random.randint(1, max_val)]
            candidates = list(range(1, max_val + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                if x in current:
                    continue
                if all(x + c in smooth_3 for c in current):
                    current.append(x)
                    if len(current) == n:
                        return current
        return None

    sol = try_find_5_smooth()
    if sol:
        return sol

    # 3. Fallback: Local Search to minimize distinct prime count
    # Initialize with random set
    best_set = list(range(1, n+1))
    best_score = float('inf')
    
    def get_prime_factors(num):
        factors = set()
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors.add(d)
                num //= d
            d += 1
        if num > 1:
            factors.add(num)
        return factors

    def evaluate_set(nums):
        primes = set()
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                primes.update(get_prime_factors(nums[i] + nums[j]))
        return len(primes)

    best_score = evaluate_set(best_set)
    
    # Simple local search
    for _ in range(10000):
        if time.time() > deadline:
            break
        i = random.randint(0, n-1)
        old_val = best_set[i]
        new_val = random.randint(1, max_val)
        if new_val in best_set:
            continue
        best_set[i] = new_val
        score = evaluate_set(best_set)
        if score < best_score:
            best_score = score
        else:
            best_set[i] = old_val
            
    return best_set
