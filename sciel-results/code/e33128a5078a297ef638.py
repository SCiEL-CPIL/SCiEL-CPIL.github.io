import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 55.0
    candidates = list(range(2, 41))
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute exponent vectors for each candidate
    # Vector is a tuple of length num_primes
    def get_vec(n):
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        # n should be 1 if we covered all primes <= 40, which we did.
        return tuple(vec)
        
    vecs = {n: get_vec(n) for n in candidates}
    
    # Initial best set: all primes
    # This guarantees distinct products
    best_set = primes[:]
    best_size = len(best_set)
    
    # Helper to check if a list of elements is valid
    # Returns True if valid, and updates the set of vectors if valid (optional, but we return the set for efficiency in search)
    def check_validity(elements):
        # Start with empty product vector (all zeros)
        current_vectors = {tuple([0] * num_primes)}
        for x in elements:
            xv = vecs[x]
            new_vectors = set()
            valid = True
            for v in current_vectors:
                # Calculate new vector v + xv
                # Tuple addition is not native, so we do it manually or use a helper
                # Optimization: Pre-compute addition? No, vectors are small.
                new_v = tuple(v[i] + xv[i] for i in range(num_primes))
                if new_v in new_vectors:
                    valid = False
                    break
                new_vectors.add(new_v)
            
            if not valid:
                return False, None
            
            # Union with existing
            current_vectors.update(new_vectors)
            
        return True, current_vectors

    # Faster check for adding ONE element to an existing set of vectors
    def check_add_single(current_vectors, x):
        xv = vecs[x]
        new_vectors = set()
        # We need to check if any v + xv collides with existing or other new ones
        # Actually, it's easier to just generate the new set and check size
        # But we can early exit.
        
        # The new set will be current_vectors U {v+xv for v in current_vectors}
        # Collision if (v1 + xv) == v2 for some v1, v2 in current_vectors
        # OR (v1 + xv) == (v2 + xv) which implies v1 == v2 (impossible for distinct v)
        # So we just need to check if v + xv is in current_vectors for any v?
        # NO. If v1+xv == v2, that's a collision between a new product and an old product.
        # If v1+xv == v2+xv, that means v1==v2, which is fine.
        # So the condition is: For all v in current_vectors, (v + xv) NOT IN current_vectors.
        
        for v in current_vectors:
            new_v = tuple(v[i] + xv[i] for i in range(num_primes))
            if new_v in current_vectors:
                return False, None
        
        # If no collision with old, we can safely add.
        # The new set is current_vectors U {v+xv for v in current_vectors}
        # We don't strictly need to build it here if we just want True/False,
        # but for the search we need the new set.
        
        new_vectors = set(current_vectors)
        for v in current_vectors:
            new_v = tuple(v[i] + xv[i] for i in range(num_primes))
            new_vectors.add(new_v)
            
        return True, new_vectors

    random.seed(42)
    
    # Phase 1: Randomized Greedy with multiple restarts
    # We try to build a large set by adding elements in random order.
    
    for restart in range(500):
        if time.time() > deadline - 2.0:
            break
            
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        # Start with empty set, vector set is {0-vector}
        current_vectors = {tuple([0] * num_primes)}
        
        for x in shuffled:
            # Check if x can be added
            valid, new_vectors = check_add_single(current_vectors, x)
            if valid:
                current_set.append(x)
                current_vectors = new_vectors
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            # If we found a very good solution, we might want to stop early or tighten time
            if best_size >= 18:
                # Still try to find 19, but we have a strong baseline
                pass
    
    # Phase 2: Local Search (Swaps) to improve the best set found
    # Try to remove one element and add two? No, size must increase.
    # Standard local search: Try to swap one element out and one in? No, that keeps size same.
    # To increase size, we need to find a set of size k+1.
    # Strategy: Take the best set of size k. Try to add every possible x.
    # If all fail, try removing one element y from the set, then try to add every possible x.
    # If we can add x after removing y, we still have size k. 
    # This doesn't increase size directly.
    
    # Better Local Search: 
    # 1. Start with best_set.
    # 2. Try to add any x not in set.
    # 3. If fail, for each y in set:
    #    a. Remove y.
    #    b. Try to add any x not in (set - {y}).
    #    c. If added, size is same. 
    # This doesn't help increase size unless we can add 2 elements after removing 1?
    # That's a 1-2 swap.
    
    # Let's implement a simple "Hill Climbing" on the size.
    # Since we already did greedy, we might be stuck at a local maximum.
    # Let's try to restart the greedy process but seeding it with the best set found?
    # Or just run more restarts.
    
    # Alternative: Deterministic Greedy with different heuristics
    # Sort by number of prime factors (fewer is better for independence)
    def count_pf(n):
        c = 0
        for p in primes:
            while n % p == 0:
                c += 1
                n //= p
        return c
        
    sorted_by_pf = sorted(candidates, key=count_pf)
    
    # Try greedy with this order
    current_set = []
    current_vectors = {tuple([0] * num_primes)}
    for x in sorted_by_pf:
        valid, new_vectors = check_add_single(current_vectors, x)
        if valid:
            current_set.append(x)
            current_vectors = new_vectors
            
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set[:]
        
    # Try reverse order
    current_set = []
    current_vectors = {tuple([0] * num_primes)}
    for x in sorted_by_pf[::-1]:
        valid, new_vectors = check_add_single(current_vectors, x)
        if valid:
            current_set.append(x)
            current_vectors = new_vectors
            
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set[:]

    # Phase 3: Intensive Local Search around the best set
    # If we have a set of size K, we want to find a set of size K+1.
    # We can try to "shake" the set.
    
    if time.time() < deadline - 5.0:
        # Use the best set found so far as a seed for local improvements
        # We will try to add elements to the best set.
        # If we can't add any, we try to swap.
        
        # Re-compute vectors for best_set to be sure
        valid, best_vectors = check_validity(best_set)
        
        # Try to add any candidate not in best_set
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try adding
            for x in candidates:
                if x in best_set:
                    continue
                valid, new_vectors = check_add_single(best_vectors, x)
                if valid:
                    best_set.append(x)
                    best_vectors = new_vectors
                    best_size += 1
                    improved = True
                    break # Restart addition loop
            
            # If no addition possible, try swaps?
            # A swap keeps size constant. It might allow a future addition.
            # But checking swaps is expensive.
            # Let's try a 1-2 swap: Remove y, add x1, add x2.
            # This is complex. 
            # Instead, let's just run more random restarts if time permits.
            
            if not improved:
                # Try a few random restarts seeded near the current best?
                # Or just break and let the main loop handle it.
                break

    # Final safety check: ensure best_set is valid
    # The search logic maintains validity, but just in case
    # We trust the incremental check.
    
    # Ensure types are correct
    return [int(x) for x in best_set]

# Test locally if needed, but the function is self-contained.
if __name__ == "__main__":
    # Simple sanity check
    s = construct()
    print(f"Found set size: {len(s)}")
    print(f"Set: {s}")
    # Verify
    prods = {1}
    for x in s:
        new = {p * x for p in prods}
        if new & prods:
            print("INVALID: Collision found")
            break
        prods.update(new)
    else:
        print(f"Valid. Total products: {len(prods)}")
