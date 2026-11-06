import time
import random

def construct():
    start_time = time.time()
    timeout = 85.0
    
    # Target: 12 distinct positive integers <= 10000
    # Minimize distinct prime factors of all pairwise sums.
    
    # 1. Try to find a set of 12 integers where all pairwise sums are 7-smooth.
    #    If found, the distinct primes are {2, 3, 5, 7} -> Count 4.
    #    This is likely the optimal solution.
    
    target_primes = [2, 3, 5, 7]
    
    def is_smooth(n, primes):
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1

    # Precompute smooth numbers up to 400 (max sum for search space 1..200)
    max_search_val = 200
    smooth_nums = set()
    for n in range(2, max_search_val * 2 + 1):
        if is_smooth(n, target_primes):
            smooth_nums.add(n)
            
    def find_clique_7_smooth():
        # Build adjacency list for vertices 1..max_search_val
        adj = {v: set() for v in range(1, max_search_val + 1)}
        for u in range(1, max_search_val + 1):
            for v in range(u + 1, max_search_val + 1):
                if (u + v) in smooth_nums:
                    adj[u].add(v)
                    adj[v].add(u)
                    
        # Order vertices by degree descending to find large cliques faster
        vertices = list(range(1, max_search_val + 1))
        vertices.sort(key=lambda x: -len(adj[x]))
        
        def backtrack(clique, candidates):
            if len(clique) == 12:
                return clique
            if not candidates:
                return None
            if time.time() - start_time > timeout - 10:
                return None
                
            # Pruning: if not enough candidates, return
            if len(candidates) < 12 - len(clique):
                return None
                
            for v in candidates:
                new_clique = clique + [v]
                # New candidates: neighbors of v that are in candidates and greater than v
                new_candidates = [u for u in candidates if u in adj[v] and u > v]
                
                result = backtrack(new_clique, new_candidates)
                if result:
                    return result
            return None

        # Initial candidates sorted by degree
        initial_candidates = vertices
        return backtrack([], initial_candidates)

    best_set = None
    best_score = float('inf')
    
    # 1. Clique Search with strict time limit
    clique_deadline = start_time + 15.0
    old_timeout = timeout
    timeout = clique_deadline
    
    result = find_clique_7_smooth()
    timeout = old_timeout
    
    if result:
        # Verify
        valid = True
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                if not is_smooth(result[i]+result[j], target_primes):
                    valid = False
                    break
            if not valid: break
        if valid:
            return result

    # 2. Local Search for general minimum primes
    def get_primes(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    def score_set(A):
        primes = set()
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                primes.update(get_primes(A[i] + A[j]))
        return len(primes)

    # Initial set
    current_A = list(range(1, 13))
    current_score = score_set(current_A)
    best_set = current_A[:]
    best_score = current_score
    
    # Local Search with improved heuristic
    for restart in range(100):
        if time.time() - start_time > timeout - 5:
            break
            
        if restart == 0:
            A = current_A[:]
        else:
            # Random restart with bias toward small numbers
            A = random.sample(range(1, 101), 12)
            A.sort()
            
        score = score_set(A)
        
        if score < best_score:
            best_score = score
            best_set = A[:]
            
        no_improve = 0
        # Hill climbing with swap moves
        while no_improve < 100 and time.time() - start_time < timeout - 2:
            k = random.randint(0, 11)
            old_val = A[k]
            improved = False
            
            # Try replacing A[k] with values from a focused range
            for _ in range(30):
                new_val = random.randint(1, 150)
                if new_val in A and new_val != old_val:
                    continue
                if new_val == old_val:
                    continue
                
                A[k] = new_val
                new_score = score_set(A)
                
                if new_score < score:
                    score = new_score
                    improved = True
                    break
                elif new_score == score:
                    if random.random() < 0.03:
                        improved = True
                        break
                    else:
                        A[k] = old_val
                else:
                    A[k] = old_val
            
            if improved:
                no_improve = 0
                if score < best_score:
                    best_score = score
                    best_set = A[:]
            else:
                no_improve += 1
                
    return best_set
