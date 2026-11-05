import random
import time
from collections import Counter

def construct():
    start_time = time.time()
    best_set = []
    
    # Try to find a set of size K, starting from high K
    # Upper bound estimate: ~42
    for K in range(45, 0, -1):
        if time.time() - start_time > 55:
            break
        
        deadline = start_time + 5.0
        
        # Initialize
        A = random.sample(range(1, 4097), K)
        
        while time.time() < deadline:
            # Generate sums and track collisions
            sums = []
            # We need to be able to trace back to triples if we want targeted repair
            # For now, let's just calculate cost and do random repair to keep it simple and robust.
            # If random repair is too slow, we'll see.
            
            s_list = []
            for i in range(K):
                for j in range(i, K):
                    for l in range(j, K):
                        s_list.append(A[i] + A[j] + A[l])
            
            c = Counter(s_list)
            cost = len(s_list) - len(c)
            
            if cost == 0:
                return A
            
            # If we have time, try targeted repair
            # Find a sum with count > 1
            # Get one such sum
            target_sum = None
            for s, cnt in c.items():
                if cnt > 1:
                    target_sum = s
                    break
            
            if target_sum is not None:
                # Find two triples that sum to target_sum
                # We can re-iterate to find them
                found_triples = []
                for i in range(K):
                    for j in range(i, K):
                        for l in range(j, K):
                            if A[i] + A[j] + A[l] == target_sum:
                                found_triples.append((i, j, l))
                                if len(found_triples) == 2:
                                    break
                        if len(found_triples) == 2:
                            break
                    if len(found_triples) == 2:
                        break
                
                if len(found_triples) == 2:
                    # Pick a random index from the union of these two triples
                    idxs = set()
                    for t in found_triples:
                        idxs.update(t)
                    idx_to_change = random.choice(list(idxs))
                else:
                    idx_to_change = random.randint(0, K-1)
            else:
                idx_to_change = random.randint(0, K-1)
            
            # Replace A[idx_to_change] with a random new value
            old_val = A[idx_to_change]
            new_val = random.randint(1, 4096)
            # Ensure distinctness
            # Simple check
            while new_val in A and new_val != old_val:
                new_val = random.randint(1, 4096)
            # If new_val == old_val, it's a no-op, which is bad. 
            # Ensure new_val != old_val
            if new_val == old_val:
                new_val = (old_val % 4096) + 1
                
            A[idx_to_change] = new_val

        # If we exited the while loop without returning, we didn't find a valid set for this K
        # We can keep the best set found during this K's search if it's better than global best?
        # But we don't track "best cost" separately from the global best set.
        # Since we are iterating K downwards, the first valid set found is the largest.
        # If we fail to find a valid set for K, we proceed to K-1.
        # We should NOT return an invalid set.
        # The problem requires a valid set.
        # So if we don't find a valid set for K, we just continue to K-1.
        # If we never find a valid set, we return a trivial valid set.
        
    return [1]
