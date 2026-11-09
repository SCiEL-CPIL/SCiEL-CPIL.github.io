import time
import random

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Safe margin for 90s limit

    # Precompute all 4-APs
    # For each element x (0-indexed), store the list of other 3 elements (0-indexed)
    # such that {x, o1, o2, o3} forms a 4-AP.
    # This is equivalent to checking if o1, o2, o3 are all in the set.
    # conflicts[x] = list of tuples (o1, o2, o3)
    
    # We can represent the set as a Python integer bitmask.
    # bit i is set if element i (0-indexed) is in the set.
    # Checking if o1, o2, o3 are in set: (mask >> o1) & 1, etc.
    # This is much faster than numpy indexing for small N.
    
    conflicts = [[] for _ in range(N)]
    
    for d in range(1, (N // 3) + 1):
        for a in range(1, N - 3 * d + 1):
            # 1-indexed: a, a+d, a+2d, a+3d
            # 0-indexed: a-1, a+d-1, a+2d-1, a+3d-1
            p0 = a - 1
            p1 = a + d - 1
            p2 = a + 2 * d - 1
            p3 = a + 3 * d - 1
            
            conflicts[p0].append((p1, p2, p3))
            conflicts[p1].append((p0, p2, p3))
            conflicts[p2].append((p0, p1, p3))
            conflicts[p3].append((p0, p1, p2))

    def can_add(mask, x):
        """
        Check if x (0-indexed) can be added to set represented by mask.
        Returns True if valid.
        """
        for (o1, o2, o3) in conflicts[x]:
            if ((mask >> o1) & 1) and ((mask >> o2) & 1) and ((mask >> o3) & 1):
                return False
        return True

    def get_blocking_triples(mask, x):
        """
        Return list of blocking triples for x given mask.
        A triple (o1, o2, o3) is blocking if all are in mask.
        """
        blocked = []
        for (o1, o2, o3) in conflicts[x]:
            if ((mask >> o1) & 1) and ((mask >> o2) & 1) and ((mask >> o3) & 1):
                blocked.append((o1, o2, o3))
        return blocked

    def popcount(mask):
        return bin(mask).count('1')

    def greedy_construct(seed):
        """
        Greedy construction with random order.
        Returns (mask, size).
        """
        rng = random.Random(seed)
        elements = list(range(N))
        rng.shuffle(elements)
        
        mask = 0
        size = 0
        for x in elements:
            if can_add(mask, x):
                mask |= (1 << x)
                size += 1
        return mask, size

    def local_search(mask, size, rng, time_budget):
        """
        Local search using kick strategy.
        Returns (best_mask, best_size).
        """
        best_mask = mask
        best_size = size
        
        # Precompute list of indices for faster random access?
        # Not really needed, randint is fast.
        
        while time.time() < time_budget:
            # 1. Try to add a random element
            x = rng.randint(0, N - 1)
            
            if (mask >> x) & 1:
                continue # Already in set
                
            if can_add(mask, x):
                mask |= (1 << x)
                size += 1
                if size > best_size:
                    best_size = size
                    best_mask = mask
            else:
                # Cannot add x. Find blocking triples.
                blocked = get_blocking_triples(mask, x)
                
                if blocked:
                    # Pick a random blocking triple
                    triple = blocked[rng.randint(0, len(blocked) - 1)]
                    
                    # Pick a random element from the triple to remove
                    y = triple[rng.randint(0, 2)]
                    
                    if (mask >> y) & 1:
                        mask &= ~(1 << y)
                        size -= 1
                        
                        # Try to add x again
                        if can_add(mask, x):
                            mask |= (1 << x)
                            size += 1
                            if size > best_size:
                                best_size = size
                                best_mask = mask
                        else:
                            # Still blocked.
                            # Revert if we've dropped too far below best
                            if size < best_size - 5:
                                mask = best_mask
                                size = best_size
                else:
                    # This case should not happen if can_add returned False
                    pass
                    
            # 2. Occasionally try a sweep to catch easy adds
            if rng.random() < 0.05:
                for _ in range(10):
                    y = rng.randint(0, N - 1)
                    if not ((mask >> y) & 1) and can_add(mask, y):
                        mask |= (1 << y)
                        size += 1
                        if size > best_size:
                            best_size = size
                            best_mask = mask

        return best_mask, best_size

    # Main
    best_mask = 0
    best_size = 0
    rng = random.Random(42)
    
    # Multiple restarts with different greedy seeds
    # Since bitmask is fast, we can do many more restarts.
    
    # Let's do a batch of initial greedy constructions
    # Then local search on the best ones.
    
    # Strategy:
    # 1. Run many greedy restarts to find a good base.
    # 2. Run local search on the best base.
    # 3. If time permits, run more restarts or refine further.
    
    # How many greedy restarts can we do?
    # Greedy is O(N * avg_conflicts). N=512, avg_conflicts ~ 50-100.
    # Very fast. Let's do 50-100 restarts.
    
    greedy_results = []
    
    # Time budget for greedy phase: 5 seconds
    greedy_deadline = start_time + 5.0
    
    num_restarts = 0
    while time.time() < greedy_deadline:
        seed = num_restarts
        mask, size = greedy_construct(seed)
        greedy_results.append((size, mask))
        num_restarts += 1
        
    # Sort by size descending
    greedy_results.sort(key=lambda x: x[0], reverse=True)
    
    # Take top few for local search
    # Let's take top 5 distinct masks (or just top 5 by size)
    # Actually, just take the best one and maybe a couple of others if time allows.
    
    # Time budget for local search: remaining time - 2s margin
    ls_deadline = start_time + time_limit
    
    # Process top candidates
    # We'll run local search on the best candidate first.
    # If it finishes early, try the next best.
    
    candidates = greedy_results[:5]
    
    for size, mask in candidates:
        if time.time() >= ls_deadline - 1:
            break
            
        if size < best_size:
            # No point in local searching a worse base than current best
            # unless we expect huge gains, but usually best base is best start.
            # However, sometimes a slightly smaller base allows larger growth.
            # Let's only search if it's close to best or if we have lots of time.
            if size < best_size - 2:
                continue
            # If we have very little time left, don't risk it
            if time.time() > ls_deadline - 5:
                continue

        new_mask, new_size = local_search(mask, size, rng, ls_deadline - 1)
        
        if new_size > best_size:
            best_size = new_size
            best_mask = new_mask

    # Final check: Try to add any remaining elements greedily in natural order
    # This catches any easy wins missed by the stochastic search
    for x in range(N):
        if not ((best_mask >> x) & 1) and can_add(best_mask, x):
            best_mask |= (1 << x)
            best_size += 1

    # Convert to list of 1-indexed integers
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result

# EVOLVE-BLOCK-END
