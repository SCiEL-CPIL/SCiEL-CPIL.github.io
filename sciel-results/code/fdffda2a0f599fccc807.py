import time
import random
from itertools import combinations

N = 24
ALL = list(range(1, 25))
L = 5354228880  # lcm(1..24)
RECIP = [0] + [L // i for i in range(1, 25)]  # RECIP[i] = L//i
MAX_SUM = sum(RECIP[1:25])

def _gen_sums(arr, max_val):
    sums = {0}
    for w in arr:
        if w > max_val:
            continue
        new = set()
        for s in sums:
            ns = s + w
            if ns <= max_val:
                new.add(ns)
        sums |= new
    return sums

# Precompute for each i: the list of subsets (as frozensets of element indices)
# of [1..24]\{i} whose reciprocals sum to 1/i.
# Only store subsets up to a reasonable size to keep memory manageable.
DANGEROUS = [[] for _ in range(25)]  # DANGEROUS[i] = list of frozensets

def _precompute_dangerous():
    for i in range(1, 25):
        target = RECIP[i]
        others = [k for k in range(1, 25) if k != i]
        n = len(others)
        mid = n // 2
        left = others[:mid]
        right = others[mid:]
        
        left_sums = _gen_sums([RECIP[k] for k in left], target)
        right_sums = _gen_sums([RECIP[k] for k in right], target)
        
        found = set()
        for ls in left_sums:
            rs_needed = target - ls
            if rs_needed in right_sums:
                # Need to find actual subsets, not just that a sum exists.
                # Reconstruct: find subset of left summing to ls and subset of right summing to rs_needed.
                pass
        
        # Alternative: brute-force over all subsets of others (up to size 6 to be safe)
        # Since 1/i = sum of 1/k for distinct k > i typically needs small subsets.
        # Actually for small i, subsets can be larger. Let's do meet-in-the-middle with reconstruction.
        
        # Build mapping from sum -> list of subsets for left and right
        def gen_sums_with_sets(arr, max_val):
            # Returns dict: sum -> list of frozensets of indices into arr
            result = {0: [frozenset()]}
            for idx, w in enumerate(arr):
                if w > max_val:
                    continue
                new = {}
                for s, subsets in result.items():
                    ns = s + w
                    if ns <= max_val:
                        if ns not in new:
                            new[ns] = []
                        for sub in subsets:
                            new[ns].append(sub | frozenset([idx]))
                for s, subsets in new.items():
                    if s in result:
                        result[s].extend(subsets)
                    else:
                        result[s] = subsets
            return result
        
        left_map = gen_sums_with_sets([RECIP[k] for k in left], target)
        right_map = gen_sums_with_sets([RECIP[k] for k in right], target)
        
        for ls in left_sums:
            rs_needed = target - ls
            if rs_needed in right_sums:
                for lsub in left_map[ls]:
                    for rsub in right_map[rs_needed]:
                        # Convert indices to actual elements
                        elem_set = set()
                        for idx in lsub:
                            elem_set.add(left[idx])
                        for idx in rsub:
                            elem_set.add(right[idx])
                        if elem_set:  # non-empty
                            found.add(frozenset(elem_set))
        
        DANGEROUS[i] = list(found)

_precompute_dangerous()

# Convert dangerous subsets to bitmasks for fast containment check
DANGEROUS_MASKS = [[] for _ in range(25)]
for i in range(1, 25):
    masks = []
    for fs in DANGEROUS[i]:
        mask = 0
        for k in fs:
            mask |= (1 << (k - 1))
        masks.append(mask)
    DANGEROUS_MASKS[i] = masks

def is_valid_mask(mask):
    """Check if the set represented by mask is valid."""
    for i in range(1, 25):
        if mask & (1 << (i - 1)):
            for dm in DANGEROUS_MASKS[i]:
                if dm & mask == dm:
                    return False
    return True

def is_valid_list(S):
    """Check if list S is valid."""
    mask = 0
    for x in S:
        mask |= (1 << (x - 1))
    return is_valid_mask(mask)

def mask_to_list(mask):
    return [i for i in range(1, 25) if mask & (1 << (i - 1))]

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin
    
    def now():
        return time.time()
    
    # Start with upper half as seed
    seed_mask = 0
    for i in range(13, 25):
        seed_mask |= (1 << (i - 1))
    
    best_mask = seed_mask if is_valid_mask(seed_mask) else (1 << 23)
    best_size = bin(best_mask).count('1')
    
    def greedy_add(order, start_mask=None):
        mask = start_mask if start_mask is not None else 0
        for x in order:
            if now() > deadline:
                break
            bit = 1 << (x - 1)
            if mask & bit:
                continue
            trial = mask | bit
            if is_valid_mask(trial):
                mask = trial
        return mask
    
    # Multiple greedy orderings
    rng = random.Random(42)
    orderings = [
        ALL[:],
        ALL[::-1],
        sorted(ALL, key=lambda x: RECIP[x]),
        sorted(ALL, key=lambda x: RECIP[x], reverse=True),
        sorted(ALL, key=lambda x: x % 7),
    ]
    for _ in range(15):
        o = ALL[:]
        rng.shuffle(o)
        orderings.append(o)
    
    for o in orderings:
        if now() > deadline:
            break
        res = greedy_add(o)
        sz = bin(res).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = res
    
    # Local search: try adding missing elements, and swaps
    def local_search(mask, max_iters=200):
        iters = 0
        improved = True
        while improved and iters < max_iters and now() < deadline:
            iters += 1
            improved = False
            
            # Try additions
            missing = [x for x in ALL if not (mask & (1 << (x - 1)))]
            rng.shuffle(missing)
            for x in missing:
                if now() > deadline:
                    break
                trial = mask | (1 << (x - 1))
                if is_valid_mask(trial):
                    mask = trial
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove a, add b
            in_set = [x for x in ALL if mask & (1 << (x - 1))]
            if not in_set or not missing:
                break
            rng.shuffle(in_set)
            rng.shuffle(missing)
            for a in in_set:
                if now() > deadline:
                    break
                for b in missing:
                    trial = (mask & ~(1 << (a - 1))) | (1 << (b - 1))
                    if is_valid_mask(trial):
                        mask = trial
                        improved = True
                        break
                if improved:
                    break
        
        return mask
    
    # Apply local search to best
    if best_mask:
        ls = local_search(best_mask, max_iters=150)
        sz = bin(ls).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = ls
    
    # Multi-start: random subsets, repair, add back, local search
    for _ in range(30):
        if now() > deadline:
            break
        target_size = random.randint(max(2, best_size - 2), 24)
        rand_set = random.sample(ALL, min(target_size, 24))
        rand_mask = 0
        for x in rand_set:
            rand_mask |= (1 << (x - 1))
        
        # Repair: remove violators
        while now() < deadline:
            # Find a violator
            violator = None
            for i in range(1, 25):
                if rand_mask & (1 << (i - 1)):
                    for dm in DANGEROUS_MASKS[i]:
                        if dm & rand_mask == dm:
                            violator = i
                            break
                if violator:
                    break
            if violator is None:
                break
            rand_mask &= ~(1 << (violator - 1))
        
        # Add back
        rand_mask = greedy_add(ALL, start_mask=rand_mask)
        
        # Local search
        rand_mask = local_search(rand_mask, max_iters=80)
        
        sz = bin(rand_mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = rand_mask
    
    # Final validation
    if not is_valid_mask(best_mask):
        best_mask = 1 << 23  # {24}
    
    return mask_to_list(best_mask)

if __name__ == "__main__":
    print(construct())
