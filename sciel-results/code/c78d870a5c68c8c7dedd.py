import random
import itertools
import time

def max_sidon_size(S):
    """
    Computes the size of the largest subset of S with all unordered pair sums unique.
    S is a list of 12 distinct integers.
    """
    n = len(S)
    
    def is_sidon(subset):
        sums = set()
        for i in range(len(subset)):
            for j in range(i, len(subset)):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    for k in range(n, 0, -1):
        for subset in itertools.combinations(S, k):
            if is_sidon(list(subset)):
                return k
    return 1

def count_collisions(S):
    """
    Counts the number of pairs of pairs (i,j) and (k,l) with i<=j, k<=l
    such that S[i]+S[j] == S[k]+S[l].
    """
    sums = {}
    count = 0
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                count += sums[s]
            sums[s] = sums.get(s, 0) + 1
    return count

def construct():
    # Seed the random generator for reproducibility if needed, 
    # but standard random is fine.
    best_set = None
    best_sidon_size = 13 # Max possible is 12, so 13 is infinity
    
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
    # Number of restarts
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize random set
        S = list(random.sample(range(129), 12))
        current_collisions = count_collisions(S)
        current_sidon = max_sidon_size(S)
        
        if current_sidon < best_sidon_size:
            best_sidon_size = current_sidon
            best_set = S[:]
            
        # Local Search
        # We want to MAXIMIZE collisions to MINIMIZE sidon size.
        # So we accept moves that increase collisions.
        no_improve_count = 0
        max_no_improve = 5000
        
        while no_improve_count < max_no_improve:
            if time.time() - start_time > time_limit:
                break
                
            # Pick random index to change
            i = random.randint(0, 11)
            old_val = S[i]
            
            # Pick new value
            # To avoid many duplicates, we can sample from the available range
            # or just sample and check.
            new_val = random.randint(0, 128)
            
            if new_val == old_val:
                continue
            if new_val in S:
                continue
                
            # Apply move
            S[i] = new_val
            new_collisions = count_collisions(S)
            
            if new_collisions > current_collisions:
                current_collisions = new_collisions
                no_improve_count = 0
                # Check sidon size occasionally? 
                # It's expensive. Let's check every 1000 successful moves or at the end.
                # Actually, let's just check at the end of the local search phase.
            else:
                # Revert
                S[i] = old_val
                no_improve_count += 1
                
        # End of local search phase
        final_sidon = max_sidon_size(S)
        if final_sidon < best_sidon_size:
            best_sidon_size = final_sidon
            best_set = S[:]
            
    # Fallback if something went wrong
    if best_set is None:
        best_set = list(range(12))
        
    return best_set
