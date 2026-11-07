import time
import random

def construct():
    deadline = time.time() + 55.0
    best_set = []
    best_size = 0
    
    # Helper to check if a set of sums per cardinality has collisions
    def has_collision(sums_by_k):
        # sums_by_k is a list of sets
        # Check if any sum is in sums_by_k[i] and sums_by_k[j] for i != j
        # We can flatten and check duplicates, but we need to know the index
        # Or simply: for each i, for each j > i, check intersection
        n = len(sums_by_k)
        for i in range(n):
            for j in range(i+1, n):
                if sums_by_k[i] & sums_by_k[j]:
                    return True
        return False

    # Greedy construction with incremental update
    def try_build(candidates):
        current_set = []
        # sums_by_k[k] is the set of all subset sums of current_set with cardinality k
        sums_by_k = [set([0])]  # Start with empty set sum 0, cardinality 0
        
        for x in candidates:
            # Try to add x
            # New sums_by_k_new
            # sums_by_k_new[k] = sums_by_k[k] U {s + x for s in sums_by_k[k-1]}
            
            # We need to check if adding x causes a collision.
            # We can compute the new sets and check.
            # To avoid copying large sets, we can just compute the new sums for the new cardinalities?
            # No, the existing sums remain, but they might collide with the new ones.
            
            # Let's just compute the full new state if the set is small.
            # Since we expect size < 20, copying is fine.
            
            new_sums_by_k = [set(s) for s in sums_by_k]
            # Extend the list if needed
            if len(new_sums_by_k) == len(current_set) + 1:
                 new_sums_by_k.append(set())
            
            # Update from high k down to low k to avoid using updated values?
            # Actually, the formula is:
            # New Sums[k] = Old Sums[k] U {s+x for s in Old Sums[k-1]}
            # We can iterate k from len(current_set) down to 1?
            # Or just create a new list.
            
            old_sums = sums_by_k
            new_sums = [set(s) for s in old_sums]
            if len(new_sums) < len(current_set) + 2:
                 new_sums.append(set())
            
            for k in range(1, len(current_set) + 2):
                if k > 0 and k-1 < len(old_sums):
                    new_sums[k].update(s + x for s in old_sums[k-1])
            
            if not has_collision(new_sums):
                current_set.append(x)
                sums_by_k = new_sums
            else:
                # Cannot add x
                pass
                
        return current_set

    # Randomized search
    while time.time() < deadline:
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        current_set = try_build(candidates)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
            # If we reach a very large size, we might stop early? 
            # Or just keep trying to see if we can go higher.
            # 20 is likely a good upper bound for this range.
            if best_size >= 20:
                # We can break or continue to try for 21, 22...
                # Let's continue for a bit longer to be sure.
                pass

    return list(best_set)
