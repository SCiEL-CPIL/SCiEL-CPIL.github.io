import itertools
import random
import time

def construct(q, n, radius):
    # For this specific task, q=3, n=7, radius=1.
    # We assume the evaluator passes these specific parameters as per the task description.
    # If other parameters are passed, we fall back to a simple construction.
    if q != 3 or n != 7 or radius != 1:
        # Fallback: Direct sum construction (same as seed)
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Trivial case for small n
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific solver for K_3(7,1)
    total_words = 3 ** n
    
    def to_index(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    def to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return t

    # Precompute cover masks for each word
    # cover_masks[i] is an integer where bit j is 1 if word j is within radius of word i
    cover_masks = [0] * total_words
    for idx in range(total_words):
        w = to_tuple(idx)
        mask = 0
        # Self
        mask |= (1 << idx)
        # Distance 1
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = val
                    new_idx = to_index(tuple(new_w))
                    mask |= (1 << new_idx)
        cover_masks[idx] = mask

    best_code_size = float('inf')
    best_centers = []

    # Run multiple randomized greedy + local search iterations
    start_time = time.time()
    max_time = 115  # Leave 5 seconds buffer
    
    for iteration in range(10000):
        if time.time() - start_time > max_time:
            break
            
        random.seed(iteration)
        uncovered = (1 << total_words) - 1
        centers = []
        
        # Greedy Phase with randomization to escape local minima
        while uncovered:
            best_score = -1
            best_idx = -1
            # To speed up, we can just iterate all. 2187 is small.
            for i in range(total_words):
                # Count how many uncovered words this center covers
                score = (cover_masks[i] & uncovered).bit_count()
                # Add random noise to break ties and diversify
                if score > 0:
                    score += random.random() * 0.5
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_score < 0.1: # Only noise, no actual coverage
                break
                
            centers.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
        # Local Search: Remove redundant centers using bit-parallel counting
        # We compute for each center how many words it uniquely covers.
        # A center is redundant if it uniquely covers 0 words.
        # We can do this efficiently by maintaining counts.
        
        def remove_redundant(centers_list):
            """Remove redundant centers and return the reduced list."""
            changed = True
            while changed:
                changed = False
                # Count coverage from each center
                counts = [0] * total_words
                for c in centers_list:
                    mask = cover_masks[c]
                    # Iterate through set bits in mask
                    m = mask
                    while m:
                        # Get lowest set bit
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        m ^= lsb
                
                # Now check each center for redundancy
                new_centers = []
                for c in centers_list:
                    mask = cover_masks[c]
                    # Check if any word in mask has count == 1
                    redundant = True
                    m = mask
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        if counts[idx] == 1:
                            redundant = False
                            break
                        m ^= lsb
                    
                    if redundant:
                        changed = True
                    else:
                        new_centers.append(c)
                
                centers_list = new_centers
            return centers_list
        
        centers = remove_redundant(centers)
        
        # Additional local search: try swapping one center with another
        # This can help escape local minima where no single removal is possible
        # but a swap reduces size or enables further removals
        swapped = True
        while swapped:
            swapped = False
            centers_list = centers[:]
            n_centers = len(centers_list)
            for i in range(n_centers):
                if time.time() - start_time > max_time:
                    break
                # Try removing center i and replacing with any other word
                original_mask = cover_masks[centers_list[i]]
                centers_list.remove(centers_list[i])
                
                # Compute what's still covered
                covered = 0
                for c in centers_list:
                    covered |= cover_masks[c]
                
                # What's uncovered?
                uncovered_after = ((1 << total_words) - 1) & ~covered
                
                if uncovered_after == 0:
                    # We can just remove center i!
                    centers = centers_list
                    swapped = True
                    break
                else:
                    # Try to find a single center that covers all uncovered
                    found = False
                    for j in range(total_words):
                        if (cover_masks[j] & uncovered_after) == uncovered_after:
                            centers_list.append(j)
                            centers = remove_redundant(centers_list)
                            found = True
                            swapped = True
                            break
                    if found:
                        break
                    # If no single center works, try pairs
                    # This is expensive, so limit to a few attempts
                    if random.random() < 0.1:
                        # Try a random pair
                        for attempt in range(10):
                            j1 = random.randint(0, total_words - 1)
                            j2 = random.randint(0, total_words - 1)
                            if j1 == j2:
                                continue
                            test_covered = covered | cover_masks[j1] | cover_masks[j2]
                            if (((1 << total_words) - 1) & ~test_covered) == 0:
                                centers_list.append(j1)
                                centers_list.append(j2)
                                centers = remove_redundant(centers_list)
                                swapped = True
                                break
                            if time.time() - start_time > max_time:
                                break
                        if swapped:
                            break
            
            if time.time() - start_time > max_time:
                break
        
        if len(centers) < best_code_size:
            best_code_size = len(centers)
            best_centers = centers.copy()

    # Convert indices back to tuples/lists
    result = [to_tuple(idx) for idx in best_centers]
    return result
