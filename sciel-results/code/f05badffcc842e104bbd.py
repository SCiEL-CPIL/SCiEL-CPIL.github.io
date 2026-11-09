import time
import random
import numpy as np

N = 512
aps = []
aps_by_element = [[] for _ in range(N+1)]

for d in range(1, (N-2)//3 + 1):
    for a in range(1, N - 3*d + 1):
        b = a + d
        c = a + 2*d
        d_val = a + 3*d
        ap_idx = len(aps)
        aps.append((a, b, c, d_val))
        for x in (a, b, c, d_val):
            aps_by_element[x].append(ap_idx)

# Precompute others arrays for fast recompute
others_arrays = [None] * (N+1)
for z in range(1, N+1):
    if aps_by_element[z]:
        arr = np.array([aps[idx][i for i in range(4) if aps[idx][i] != z] for idx in aps_by_element[z]])
        others_arrays[z] = arr
    else:
        others_arrays[z] = np.array([], dtype=int).reshape(0, 3)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    S_mask = np.zeros(N+1, dtype=bool)
    
    # Init
    elements = list(range(1, N+1))
    random.shuffle(elements)
    for x in elements:
        # Check if valid
        # Using fast check
        if others_arrays[x] is not None and len(others_arrays[x]) > 0:
            valid = not (S_mask[others_arrays[x]].all(axis=1).any())
        else:
            valid = True
        if valid:
            S_mask[x] = True
            
    best_set = S_mask.copy()
    best_size = S_mask.sum()
    
    def recompute_conflicts():
        # Returns boolean mask of "can add"
        can_add = np.zeros(N+1, dtype=bool)
        for z in range(1, N+1):
            if S_mask[z]:
                continue
            if others_arrays[z] is None or len(others_arrays[z]) == 0:
                can_add[z] = True
            else:
                # If any AP is complete, z is invalid
                # S_mask[others_arrays[z]] is shape (K, 3)
                # .all(axis=1) is shape (K,)
                # .any() is bool
                is_invalid = S_mask[others_arrays[z]].all(axis=1).any()
                can_add[z] = not is_invalid
        return can_add

    current_can_add = recompute_conflicts()
    
    while time.time() < deadline:
        # Try to add
        candidates = np.where(current_can_add & ~S_mask)[0]
        if len(candidates) > 0:
            x = candidates[random.randint(0, len(candidates)-1)]
            S_mask[x] = True
            current_can_add = recompute_conflicts()
            size = S_mask.sum()
            if size > best_size:
                best_size = size
                best_set = S_mask.copy()
        else:
            # Stuck. Swap.
            in_s = np.where(S_mask)[0]
            not_in_s = np.where(~S_mask)[0]
            if len(in_s) > 0 and len(not_in_s) > 0:
                y = in_s[random.randint(0, len(in_s)-1)]
                x = not_in_s[random.randint(0, len(not_in_s)-1)]
                S_mask[y] = False
                S_mask[x] = True
                current_can_add = recompute_conflicts()
                
                # If we can add now, great.
                # If not, we continue.
                
        # Perturb
        if random.random() < 0.05:
            in_s = np.where(S_mask)[0]
            k = max(1, int(0.1 * len(in_s)))
            to_remove = random.sample(in_s, k)
            for z in to_remove:
                S_mask[z] = False
            current_can_add = recompute_conflicts()

    return [z for z in range(1, N+1) if best_set[z]]
