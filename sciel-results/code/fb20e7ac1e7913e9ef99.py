import time
import random
from math import gcd

def construct():
    deadline = time.time() + 55
    candidates = list(range(2, 41))
    
    # Primes in the range
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Precompute exponent vectors for all candidates
    # Dimensions correspond to primes
    num_primes = len(primes)
    vecs = {}
    for n in candidates:
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vecs[n] = tuple(v)
        
    zero_vec = (0,) * num_primes
    
    def is_dissociated(vec_set):
        """
        Checks if the set of vectors is dissociated.
        This is equivalent to all subset sums being distinct.
        We generate all subset sums and check for duplicates.
        """
        sums = {zero_vec}
        for v in vec_set:
            new_sums = []
            for s in sums:
                ns = tuple(s[i] + v[i] for i in range(num_primes))
                if ns in sums:
                    return False
                new_sums.append(ns)
            sums.update(new_sums)
        return True
        
    def greedy_fill(initial_set):
        """
        Tries to add as many candidates as possible to initial_set
        maintaining dissociation.
        """
        current_set = list(initial_set)
        current_vecs = [vecs[x] for x in current_set]
        
        # Pre-compute current sums to avoid rebuilding from scratch for each check
        # But rebuilding is O(2^k * k). For k~15, 2^15 * 12 ~ 400k ops. Fast enough.
        
        # We will iterate through candidates in a heuristic order
        # Sort by number of prime factors (ascending) to add "simple" numbers first?
        # Or by norm?
        # Let's try sorting by the number of non-zero components in the vector
        def vec_complexity(v):
            return sum(1 for x in v if x != 0)
            
        remaining = [x for x in candidates if x not in current_set]
        remaining.sort(key=lambda x: vec_complexity(vecs[x]))
        
        for x in remaining:
            if time.time() > deadline - 5:
                break
            v = vecs[x]
            # Check if adding v keeps the set dissociated
            # We need to check if any existing sum s satisfies s + v = s' for some other sum s'
            # This is equivalent to checking if the new set of sums has duplicates.
            # Efficient check: 
            # Let S be the set of current sums.
            # New sums are S U (S + v).
            # Collision exists if (S + v) intersects S, OR if (S + v) has internal duplicates (impossible if v != 0 and S is dissociated? No, S is a set of sums of a dissociated set, so sums are distinct. S+v is a translation, so distinct. The only collision is between S and S+v).
            # So we just need to check if (S + v) intersects S.
            
            # Rebuild sums for current_set
            sums = {zero_vec}
            for u in current_vecs:
                new_sums = [tuple(s[i] + u[i] for i in range(num_primes)) for s in sums]
                sums.update(new_sums)
                
            # Check intersection
            collision = False
            for s in sums:
                ns = tuple(s[i] + v[i] for i in range(num_primes))
                if ns in sums:
                    collision = True
                    break
                    
            if not collision:
                current_set.append(x)
                current_vecs.append(v)
                
        return current_set

    # Initial best set: Primes
    best_set = primes[:]
    best_size = len(best_set)
    
    # Try to improve best_set with greedy fill
    improved_best = greedy_fill(best_set)
    if len(improved_best) > best_size:
        best_set = improved_best
        best_size = len(best_set)
    
    # Local Search
    # We will try swaps.
    # To make it faster, we'll use a randomized order for swaps and restarts.
    
    random.seed(42)
    
    while time.time() < deadline - 2:
        current_set = best_set[:]
        current_size = len(current_set)
        
        # Try to add elements greedily to current_set first
        current_set = greedy_fill(current_set)
        current_size = len(current_set)
        
        if current_size > best_size:
            best_set = current_set
            best_size = current_size
            continue
            
        # Perform swap-based local search
        improved = False
        # Shuffle candidates for randomness in search order
        candidates_list = candidates[:]
        random.shuffle(candidates_list)
        
        # Iterate through elements in current_set
        for i in range(len(current_set)):
            if time.time() > deadline - 2:
                break
            if improved:
                break
                
            x_to_remove = current_set[i]
            
            for y_to_add in candidates_list:
                if y_to_add in current_set:
                    continue
                    
                # Try swap
                trial_set = current_set[:]
                trial_set.remove(x_to_remove)
                trial_set.append(y_to_add)
                
                # Check if trial_set is dissociated
                if is_dissociated([vecs[x] for x in trial_set]):
                    # If valid, try to fill
                    filled_set = greedy_fill(trial_set)
                    
                    if len(filled_set) > current_size:
                        current_set = filled_set
                        current_size = len(current_set)
                        improved = True
                        break
                        # Note: We break out of y loop, continue to next x or restart?
                        # Usually, accepting a move and restarting the scan is better.
                        # Let's break out of both and restart the outer while loop?
                        # Or just break x loop and continue outer while?
                        # Breaking outer while loop is safer to re-evaluate from scratch.
                        # But we just updated best_set? No, current_set.
                        # Let's update best_set if improved.
                        if current_size > best_size:
                            best_set = current_set
                            best_size = current_size
                        # Restart local search from this new best
                        break
                        
        # If we improved best_set, the outer while loop will continue with the new best
        # If not, we might want to do a random restart from a different starting point
        # To avoid getting stuck, let's occasionally restart from primes
        if time.time() < deadline - 5 and random.random() < 0.05:
             # Random restart
             random_start = primes[:]
             random.shuffle(random_start)
             # Maybe add a random composite
             if random.random() < 0.5:
                 composites = [x for x in candidates if x not in primes]
                 random_start.append(random.choice(composites))
             current_set = greedy_fill(random_start)
             if len(current_set) > best_size:
                 best_set = current_set
                 best_size = len(current_set)
                 
    # Final check and cleanup
    # Ensure types are correct
    return [int(x) for x in best_set]
