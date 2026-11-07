import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Integers 2..40
    candidates = list(range(2, 41))
    
    # Precompute prime factorization vectors
    # Primes up to 40: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    def get_vector(n):
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)
        
    vectors = [get_vector(n) for n in candidates]
    
    # Function to check if a set of vectors has distinct subset sums
    # We maintain the set of sums incrementally in the search
    
    def search(k, start_idx, current_set_indices, current_sums):
        """
        Recursively try to find a valid set of size k.
        current_set_indices: list of indices of selected candidates
        current_sums: set of tuple sums of vectors for all subsets of current_set_indices
        Returns True if a valid set of size k is found, False otherwise.
        """
        if len(current_set_indices) == k:
            return True
            
        if time.time() > deadline:
            return False
            
        # Pruning: if not enough candidates left
        if len(current_set_indices) + (len(candidates) - start_idx) < k:
            return False
            
        for i in range(start_idx, len(candidates)):
            # Check if adding candidates[i] causes collision
            v = vectors[i]
            new_sums = set()
            # Optimization: check collision without creating full set if possible?
            # s + v in current_sums?
            # We need to check if {s + v for s in current_sums} intersects current_sums.
            # Actually, we just need to check if any s + v is in current_sums.
            # If yes, collision.
            # Also, s1 + v == s2 + v implies s1 == s2, so new sums are distinct among themselves.
            # So we only need to check intersection with existing sums.
            
            collision = False
            for s in current_sums:
                if tuple(x + y for x, y in zip(s, v)) in current_sums:
                    collision = True
                    break
            
            if not collision:
                # Add it
                new_sums = current_sums | {tuple(x + y for x, y in zip(s, v)) for s in current_sums}
                if search(k, i + 1, current_set_indices + [i], new_sums):
                    return True
                    
        return False

    # Try sizes from 18 down to 1
    # To improve search efficiency, we can shuffle candidates?
    # Deterministic order first.
    # If that fails, try different orderings?
    # For now, natural order.
    
    # We can also try to prioritize primes?
    # Let's stick to natural order first.
    
    for k in range(18, 11, -1):
        if search(k, 0, [], {tuple([0]*len(primes))}):
            # Found a set of size k
            # We need to return the actual integers?
            # The task says "Return a list of ints".
            # And "The raw value is the number of integers returned".
            # So we need to reconstruct the list.
            # My search function returns True/False.
            # I need to modify it to return the set or store it.
            
            pass
            
    # Refactor search to return the set
    def search_set(k, start_idx, current_set_indices, current_sums):
        if len(current_set_indices) == k:
            return current_set_indices[:]
        
        if time.time() > deadline:
            return None
            
        if len(current_set_indices) + (len(candidates) - start_idx) < k:
            return None
            
        for i in range(start_idx, len(candidates)):
            v = vectors[i]
            
            collision = False
            for s in current_sums:
                if tuple(x + y for x, y in zip(s, v)) in current_sums:
                    collision = True
                    break
            
            if not collision:
                new_sums = current_sums | {tuple(x + y for x, y in zip(s, v)) for s in current_sums}
                result = search_set(k, i + 1, current_set_indices + [i], new_sums)
                if result is not None:
                    return result
                    
        return None

    best_set = None
    for k in range(18, 11, -1):
        res = search_set(k, 0, [], {tuple([0]*len(primes))})
        if res is not None:
            best_set = res
            break
            
    if best_set is None:
        # Fall back to primes
        best_set = [i for i, n in enumerate(candidates) if vectors[i].count(1) == 1 and sum(vectors[i]) == 1]
        # Wait, vector for prime p has a single 1.
        # Vector for 1 is all 0s (not in candidates).
        # Primes: 2,3,5,7,11,13,17,19,23,29,31,37.
        # Indices:
        # 2: idx 0
        # 3: idx 1
        # ...
        # Let's just re-calculate primes
        primes_in_range = [n for n in candidates if all(n % d for d in range(2, int(n**0.5)+1))]
        best_set = [candidates.index(p) for p in primes_in_range]
        
    result = [candidates[i] for i in best_set]
    return result

# Test run
# print(construct())
