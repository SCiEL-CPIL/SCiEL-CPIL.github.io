import time
import random
from math import isqrt

def is_smooth(n, max_prime):
    """Check if n has no prime factors greater than max_prime."""
    if n <= 1:
        return True
    # Simple trial division up to sqrt(n)
    # But we only care about primes > max_prime
    # So we can divide out all primes <= max_prime
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
        p += 1 if p == 2 else 2 # 2, 3, 5, 7...
        # Optimization: only check primes up to max_prime? 
        # No, if temp > 1 after dividing by all primes <= max_prime, 
        # then temp has a prime factor > max_prime (or temp is a prime > max_prime).
        # Wait, the loop above checks ALL primes up to sqrt(temp).
        # We need to stop checking primes once p > max_prime.
        # If we stop at p > max_prime, and temp > 1, then temp has a factor > max_prime.
        
    # Corrected logic:
    temp = n
    p = 2
    while p <= max_prime and p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
        if p == 2:
            p = 3
        else:
            p += 2
            
    # If temp > 1, it means there is a prime factor remaining.
    # If p <= max_prime, we haven't checked all primes up to max_prime?
    # The loop condition is p <= max_prime.
    # If the loop finishes because p > max_prime, and temp > 1, then temp has a prime factor > max_prime.
    # If the loop finishes because p*p > temp, and temp > 1, then temp is a prime.
    # If temp is a prime, is it <= max_prime?
    # We only divided by primes <= max_prime.
    # So if temp > 1, it must be a prime factor that was NOT divided out.
    # This happens if temp is a prime > max_prime, OR if temp is a composite of primes > max_prime?
    # No, if temp is composite, it has a factor <= sqrt(temp).
    # If we divided out all primes <= max_prime, and sqrt(temp) > max_prime, we might have missed a factor.
    # But if max_prime is small (e.g. 5), and temp is 49 (7*7), and sqrt(49)=7.
    # p goes 2, 3, 5. p becomes 7. 7 <= 5 is False. Loop stops.
    # temp is 49. 49 > 1. So it is not 5-smooth. Correct.
    
    return temp == 1

def get_smooth_candidates(limit, max_prime, count):
    """Generate 'count' numbers <= limit that are max_prime-smooth."""
    # We can just iterate and check, or generate recursively.
    # Iteration is fine for limit 10000.
    candidates = []
    for i in range(1, limit + 1):
        if is_smooth(i, max_prime):
            candidates.append(i)
            if len(candidates) >= count:
                break
    return candidates

def find_clique(candidates, target_size, max_prime):
    """Find a clique of size target_size in the compatibility graph."""
    n = len(candidates)
    if n < target_size:
        return None
    
    # Precompute compatibility
    # compat[i] is a set of indices j such that candidates[i] + candidates[j] is max_prime-smooth
    compat = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if is_smooth(candidates[i] + candidates[j], max_prime):
                compat[i].add(j)
                compat[j].add(i)
                
    # Backtracking search
    clique = []
    
    def backtrack(start, current_clique):
        if len(current_clique) == target_size:
            return current_clique[:]
        
        # Pruning: if not enough candidates left
        if len(current_clique) + (n - start) < target_size:
            return None
            
        for i in range(start, n):
            # Check if i is compatible with all in current_clique
            if all(i in compat[c] for c in current_clique):
                current_clique.append(i)
                result = backtrack(i + 1, current_clique)
                if result:
                    return result
                current_clique.pop()
                
        return None

    # Optimization: Order candidates by degree (number of compatible partners) descending
    # This often finds cliques faster
    degrees = [len(compat[i]) for i in range(n)]
    sorted_indices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # We need to map sorted indices back to original logic or just use the sorted list
    # Let's just use the sorted list of candidates
    sorted_candidates = [candidates[i] for i in sorted_indices]
    # Rebuild compat for sorted candidates? Or just use the original indices in the backtrack?
    # It's easier to just run backtrack on the original indices but maybe start with high degree nodes?
    # Actually, the standard Tomita algorithm or similar is better.
    # Given the small size (target 12, candidates ~100-1000), simple backtracking might be slow if not pruned well.
    # Let's stick to the simple backtracking but start from the most connected nodes.
    
    # To make it faster, we can restrict the search space.
    # Let's try the simple backtrack first.
    # If it's too slow, we might need to restrict candidates further.
    
    # Let's try a greedy approach first to see if it works quickly.
    # Greedy: Pick the node with max degree. Remove incompatible nodes. Repeat.
    # This doesn't guarantee finding a clique of size 12 if one exists, but it's fast.
    # We will use backtracking with a time limit or candidate limit.
    
    # For safety, let's limit candidates to the first 200 smooth numbers.
    # If we don't find it there, we try larger max_prime.
    
    return backtrack(0, [])

def construct():
    start_time = time.time()
    deadline = start_time + 45  # 45s budget
    
    target_size = 12
    
    # Try increasing max_prime
    # 5-smooth, 7-smooth, etc.
    for max_prime in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if time.time() > deadline:
            break
            
        # Generate candidates
        # We want numbers that are likely to form smooth sums.
        # Smooth numbers themselves are good candidates.
        candidates = get_smooth_candidates(10000, max_prime, 300)
        
        if len(candidates) < target_size:
            continue
            
        # Find clique
        # To speed up, we can shuffle candidates if randomization helps, 
        # but deterministic is better for reproducibility.
        # We'll just run the backtracking.
        
        # Optimization: If max_prime is small, the graph is sparse.
        # If max_prime is large, the graph is dense, and a clique is likely to exist.
        
        clique_indices = find_clique(candidates, target_size, max_prime)
        
        if clique_indices:
            result = [candidates[i] for i in clique_indices]
            # Verify distinctness and limit
            if len(set(result)) == target_size and all(x <= 10000 for x in result):
                return result
                
    # Fallback: Local search on random integers
    # This is a safety net.
    best_set = list(range(1, 13))
    best_score = 999
    
    # Calculate score for a set
    def get_score(s):
        primes = set()
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                n = s[i] + s[j]
                # Factor n
                temp = n
                p = 2
                while p * p <= temp:
                    if temp % p == 0:
                        primes.add(p)
                        while temp % p == 0:
                            temp //= p
                    p += 1 if p == 2 else 2
                if temp > 1:
                    primes.add(temp)
        return len(primes)
    
    current_score = get_score(best_set)
    best_score = current_score
    
    # Simple hill climbing
    for _ in range(5000):
        if time.time() > deadline:
            break
        i = random.randint(0, 11)
        old_val = best_set[i]
        # Try a new value
        # Pick a random value, or a value close to old_val
        # Let's try values that are multiples of small numbers
        new_val = random.randint(1, 10000)
        if new_val == old_val:
            continue
        if new_val in best_set:
            continue
            
        test_set = best_set[:]
        test_set[i] = new_val
        score = get_score(test_set)
        
        if score < best_score:
            best_set = test_set
            best_score = score
            if best_score <= 3:
                break

    return best_set
