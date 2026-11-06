import random
import time
from itertools import combinations

# Precompute all 4-subsets of 16
N = 16
K = 4
ALL_COMBOS = list(combinations(range(1, N + 1), K))
# Convert to bitmasks for faster intersection
ALL_SETS = [0] * len(ALL_COMBOS)
for i, c in enumerate(ALL_COMBOS):
    mask = 0
    for x in c:
        mask |= (1 << (x - 1))
    ALL_SETS[i] = mask

# Map from mask to index for quick lookup? 
# Or just keep list of masks.
# We need to track which sets are in the family.
# family_indices: list of indices into ALL_SETS

def can_add_mask(new_mask, family_masks):
    """
    Check if new_mask can be added to family_masks without forming a sunflower.
    Sunflower: A, B, C such that A&B == A&C == B&C.
    """
    m = len(family_masks)
    for i in range(m):
        a = family_masks[i]
        inter_as = a & new_mask
        for j in range(i + 1, m):
            b = family_masks[j]
            inter_bs = b & new_mask
            if inter_as != inter_bs:
                continue
            # If intersections with new_mask are equal, check if A&B is also equal to that
            if (a & b) == inter_as:
                return False
    return True

def greedy_randomized(seed=0):
    random.seed(seed)
    indices = list(range(len(ALL_SETS)))
    random.shuffle(indices)
    
    family_masks = []
    for idx in indices:
        s = ALL_SETS[idx]
        if can_add_mask(s, family_masks):
            family_masks.append(s)
    return family_masks

def local_search(family_masks, time_limit=5.0):
    """
    Attempt to increase the size of the family by performing swaps.
    Move: Remove A, add B. If B is valid, size is same.
    To increase size, we need to find a state where we can add a new set C.
    This is complex. 
    Alternative: Random restarts of Greedy are often better than simple local search for this problem.
    However, we can try to "complete" the family by trying to add sets that were previously rejected.
    """
    # This simple local search might not increase size easily.
    # Let's rely on multiple greedy restarts with different heuristics.
    pass

def construct():
    start_time = time.time()
    timeout = 50.0 # Safety margin for 60s limit
    
    best_family = []
    best_size = 0
    
    # Run multiple greedy restarts
    # Estimate time per run: 
    # M ~ 160. M^2 ~ 25,600. 
    # 1820 candidates * 25,600 ops ~ 46M ops.
    # Python is slow. Maybe 2-5 seconds per run.
    # Let's do as many as possible.
    
    seed_counter = 0
    while time.time() - start_time < timeout:
        fam = greedy_randomized(seed=seed_counter)
        if len(fam) > best_size:
            best_size = len(fam)
            best_family = fam
            # print(f"New best: {best_size}") # Disabled for output cleanliness
            
        seed_counter += 1
        
        # If we found a very good family, we might want to spend time on local search?
        # For now, just keep restarting.
        
    # Convert best_family (list of masks) to list of sorted lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(N):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
        
    return result

if __name__ == "__main__":
    # Quick self-test
    # res = construct()
    # print(f"Size: {len(res)}")
    pass
