import time
import random
import numpy as np

N = 512
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

def construct():
    start_time = time.time()
    
    # 1. Precomputation: Conflict Masks
    # For each x, we store a list of bitmasks. 
    # Each bitmask represents the other 3 elements in a 4-AP containing x.
    # If (CurrentSet & mask) == mask, then adding x creates a 4-AP.
    
    # We use 0-indexed bits for 0..511 corresponding to values 1..512
    # Value v corresponds to bit (v-1)
    
    conflict_masks = [[] for _ in range(N)]
    
    # Generate all 4-APs
    # a, a+d, a+2d, a+3d
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            # Elements: a, a+d, a+2d, a+3d
            # Indices: a-1, a+d-1, a+2d-1, a+3d-1
            idxs = [a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1]
            
            # For each element in the AP, the "other 3" form a conflict mask
            # Element idxs[0]: other 3 are idxs[1], idxs[2], idxs[3]
            mask0 = (1 << idxs[1]) | (1 << idxs[2]) | (1 << idxs[3])
            conflict_masks[idxs[0]].append(mask0)
            
            # Element idxs[1]: other 3 are idxs[0], idxs[2], idxs[3]
            mask1 = (1 << idxs[0]) | (1 << idxs[2]) | (1 << idxs[3])
            conflict_masks[idxs[1]].append(mask1)
            
            # Element idxs[2]: other 3 are idxs[0], idxs[1], idxs[3]
            mask2 = (1 << idxs[0]) | (1 << idxs[1]) | (1 << idxs[3])
            conflict_masks[idxs[2]].append(mask2)
            
            # Element idxs[3]: other 3 are idxs[0], idxs[1], idxs[2]
            mask3 = (1 << idxs[0]) | (1 << idxs[1]) | (1 << idxs[2])
            conflict_masks[idxs[3]].append(mask3)
    
    def is_addable(x_idx, current_mask):
        """
        Check if element x_idx (0-indexed) can be added to current_mask.
        Returns True if no 4-AP is formed.
        """
        for mask in conflict_masks[x_idx]:
            if (current_mask & mask) == mask:
                return False
        return True
    
    def bit_count(mask):
        return bin(mask).count("1")
    
    def greedy_fill(order, current_mask):
        """
        Greedily add elements from 'order' to current_mask if possible.
        Returns updated mask and count.
        """
        count = bit_count(current_mask)
        for x_idx in order:
            if not (current_mask >> x_idx) & 1:  # Not already in set
                if is_addable(x_idx, current_mask):
                    current_mask |= (1 << x_idx)
                    count += 1
        return current_mask, count
    
    def local_search(current_mask):
        """
        Perform local search: Add-only then Swap.
        """
        count = bit_count(current_mask)
        improved = True
        while improved:
            improved = False
            
            # Phase 1: Add-only
            # Try to add any missing element
            for x_idx in range(N):
                if not (current_mask >> x_idx) & 1:
                    if is_addable(x_idx, current_mask):
                        current_mask |= (1 << x_idx)
                        count += 1
                        improved = True
            if improved:
                continue # Restart loop to catch new opportunities
            
            # Phase 2: Swap
            # Try to remove one element and add more
            # To speed up, we only try swaps if we can gain at least 1
            # We iterate over all y in S
            present_indices = [i for i in range(N) if (current_mask >> i) & 1]
            
            for y_idx in present_indices:
                if time.time() - start_time > TIME_LIMIT - 2:
                    return current_mask, count
                
                # Tentatively remove y
                trial_mask = current_mask & ~(1 << y_idx)
                trial_count = count - 1
                
                # Try to add elements to trial_mask
                # We want to see if we can add at least 2 elements (net gain >= 1)
                # Or even 1 element if it allows further additions? 
                # Simple heuristic: Try to greedily add to trial_mask and see final size.
                # But that's expensive. 
                # Better: Check how many x are addable to trial_mask directly.
                
                addable_count = 0
                best_adds = []
                
                # Only check elements not in trial_mask
                # Optimization: We can skip this heavy check if we just want to find *any* improvement.
                # Let's try a limited greedy fill on trial_mask
                # Actually, just checking if *any* new element can be added is O(N*K).
                # Let's try to add elements in a specific order to maximize gain.
                
                # We'll try to add elements that were previously blocked by y.
                # But identifying them is hard.
                # Let's just do a quick greedy fill on trial_mask with a limited scope or random sample.
                # Given N=512, a full scan is ~512 * K operations. K is small (~100).
                # 512 * 100 = 51,200 ops. This is fast in Python if optimized.
                
                # Let's do a full scan for addable elements on trial_mask
                # We collect them and then add the best ones.
                candidates = []
                for x_idx in range(N):
                    if not (trial_mask >> x_idx) & 1:
                        if is_addable(x_idx, trial_mask):
                            candidates.append(x_idx)
                
                if len(candidates) > 1: # Net gain at least 1
                    # Add all possible candidates?
                    # Adding one might block another.
                    # Greedily add from candidates
                    new_mask = trial_mask
                    new_count = trial_count
                    for x_idx in candidates:
                        if is_addable(x_idx, new_mask):
                            new_mask |= (1 << x_idx)
                            new_count += 1
                    
                    if new_count > count:
                        current_mask = new_mask
                        count = new_count
                        improved = True
                        break # Restart local search with new mask
        
        return current_mask, count

    best_mask = 0
    best_count = 0
    
    def update_best(mask, count):
        nonlocal best_mask, best_count
        if count > best_count:
            best_mask = mask
            best_count = count

    # 2. Initialization Strategies
    
    # Strategy 1: Bit-Reversal Order
    # This often yields good 3-AP-free sets, might help for 4-AP
    bit_rev_order = []
    for x in range(N):
        # Reverse bits of x (9 bits for 512)
        rev = 0
        for i in range(9):
            if x & (1 << i):
                rev |= (1 << (8 - i))
        bit_rev_order.append(rev)
    
    mask, cnt = greedy_fill(bit_rev_order, 0)
    mask, cnt = local_search(mask)
    update_best(mask, cnt)
    
    # Strategy 2: Modular Orders
    for mod in [7, 11, 13, 17, 19]:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        mod_order = []
        for r in range(mod):
            for x in range(N):
                if x % mod == r:
                    mod_order.append(x)
        mask, cnt = greedy_fill(mod_order, 0)
        mask, cnt = local_search(mask)
        update_best(mask, cnt)
    
    # Strategy 3: Random Orders
    rng = random.Random(42)
    for seed in range(20):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        order = list(range(N))
        rng.shuffle(order)
        mask, cnt = greedy_fill(order, 0)
        mask, cnt = local_search(mask)
        update_best(mask, cnt)
        
    # Strategy 4: Iterated Local Search (Perturbation)
    # Start from best_mask
    if best_mask != 0:
        base_mask = best_mask
        base_count = best_count
        rng2 = random.Random(123)
        
        for iter in range(50):
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            
            # Perturb: Remove k random elements
            k = rng2.randint(5, 15)
            present = [i for i in range(N) if (base_mask >> i) & 1]
            if len(present) < k:
                continue
            to_remove = rng2.sample(present, k)
            
            perturbed_mask = base_mask
            for i in to_remove:
                perturbed_mask &= ~(1 << i)
            
            # Local Search on perturbed mask
            mask, cnt = local_search(perturbed_mask)
            update_best(mask, cnt)
            
            # Optionally, update base if improved
            if cnt > base_count:
                base_mask = mask
                base_count = cnt

    # Convert best_mask to sorted list of 1-based integers
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    
    return sorted(result)

if __name__ == "__main__":
    # Quick test
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s, Size: {len(res)}")
    # Verify no 4-AP
    # (Omitted for brevity in thought process, but would be done in real eval)
