import random
import time
import itertools

N_POINTS = 16
TRIPLES = []
TRIPLE_TO_INDEX = {}

for a in range(N_POINTS):
    for b in range(a + 1, N_POINTS):
        for c in range(b + 1, N_POINTS):
            TRIPLES.append((a, b, c))
            TRIPLE_TO_INDEX[(a, b, c)] = len(TRIPLES) - 1

NUM_TRIPLES = len(TRIPLES)

masks = [[] for _ in range(NUM_TRIPLES)]

def generate_pairings(points):
    res = []
    for i in range(1, 6):
        pair1 = (points[0], points[i])
        rem_idx = [j for j in range(6) if j != 0 and j != i]
        for j in range(1, 4):
            pair2 = (points[rem_idx[0]], points[rem_idx[j]])
            rem2 = [points[k] for k in rem_idx if k != rem_idx[0] and k != rem_idx[j]]
            pair3 = (rem2[0], rem2[1])
            res.append((pair1, pair2, pair3))
    return res

# Precompute masks
for comb in itertools.combinations(range(N_POINTS), 6):
    pairings = generate_pairings(comb)
    for p1, p2, p3 in pairings:
        transversals = []
        for x in p1:
            for y in p2:
                for z in p3:
                    t = tuple(sorted((x, y, z)))
                    idx = TRIPLE_TO_INDEX[t]
                    transversals.append(idx)
        
        for i in range(8):
            mask = 0
            for j in range(8):
                if i != j:
                    mask |= (1 << transversals[j])
            masks[transversals[i]].append(mask)

# Remove duplicate masks? 
# It's possible that the same mask is added multiple times if the same K(2,2,2) is generated?
# No, each K(2,2,2) is unique.
# But different K(2,2,2)s can have the same set of other 7 triples?
# Unlikely, but possible.
# We can use a set for each index to deduplicate.
for i in range(NUM_TRIPLES):
    masks[i] = list(set(masks[i]))

def construct():
    deadline = time.time() + 140.0
    
    best_count = 0
    best_set = 0
    best_present = [False] * NUM_TRIPLES
    
    # Multiple restarts
    while time.time() < deadline:
        # Greedy Init
        indices = list(range(NUM_TRIPLES))
        random.shuffle(indices)
        
        current_set = 0
        current_count = 0
        present = [False] * NUM_TRIPLES
        absent = set(range(NUM_TRIPLES))
        
        for idx in indices:
            if time.time() > deadline:
                break
            valid = True
            for mask in masks[idx]:
                if (current_set & mask) == mask:
                    valid = False
                    break
            if valid:
                current_set |= (1 << idx)
                current_count += 1
                present[idx] = True
                absent.remove(idx)
        
        if current_count > best_count:
            best_count = current_count
            best_set = current_set
            best_present = present[:]
        
        # Local Search
        # We maintain a list of present indices for quick sampling
        present_list = [i for i in range(NUM_TRIPLES) if present[i]]
        
        no_improve_counter = 0
        max_no_improve = 5000 # Adjust based on speed
        
        while time.time() < deadline and no_improve_counter < max_no_improve:
            improved = False
            
            # 1. Try Add
            if absent:
                idx = random.choice(list(absent))
                valid = True
                for mask in masks[idx]:
                    if (current_set & mask) == mask:
                        valid = False
                        break
                if valid:
                    current_set |= (1 << idx)
                    current_count += 1
                    present[idx] = True
                    absent.remove(idx)
                    present_list.append(idx)
                    improved = True
                    if current_count > best_count:
                        best_count = current_count
                        best_set = current_set
                        best_present = present[:]
                    no_improve_counter = 0
                    continue
            
            # 2. Try Swap (1-out 1-in)
            if present_list and absent:
                # Try a few random swaps
                for _ in range(20):
                    if not absent: break
                    out_idx = random.choice(present_list)
                    in_idx = random.choice(list(absent))
                    
                    new_set = current_set & ~(1 << out_idx)
                    valid = True
                    for mask in masks[in_idx]:
                        if (new_set & mask) == mask:
                            valid = False
                            break
                    if valid:
                        # Apply
                        current_set = new_set | (1 << in_idx)
                        # current_count unchanged
                        present[out_idx] = False
                        present[in_idx] = True
                        absent.remove(in_idx)
                        
                        # Update present_list
                        # Remove out_idx, add in_idx
                        # List ops are O(N)
                        present_list.remove(out_idx)
                        present_list.append(in_idx)
                        
                        improved = True
                        no_improve_counter = 0
                        break
            
            if not improved:
                no_improve_counter += 1
            
            # 3. Try 2-out 2-in if stuck for a while
            if no_improve_counter > 1000 and len(present_list) >= 2 and len(absent) >= 2:
                for _ in range(10):
                    outs = random.sample(present_list, 2)
                    ins = random.sample(list(absent), 2)
                    new_set = current_set
                    for o in outs:
                        new_set &= ~(1 << o)
                    
                    valid = True
                    for i in ins:
                        for mask in masks[i]:
                            if (new_set & mask) == mask:
                                valid = False
                                break
                        if not valid: break
                    if valid:
                        for i in ins:
                            new_set |= (1 << i)
                        current_set = new_set
                        for o in outs:
                            present[o] = False
                            present_list.remove(o)
                        for i in ins:
                            present[i] = True
                            present_list.append(i)
                            absent.remove(i)
                        improved = True
                        no_improve_counter = 0
                        break
                if not improved:
                    no_improve_counter += 1000 # Force restart sooner

    result = []
    for i in range(NUM_TRIPLES):
        if best_set & (1 << i):
            result.append(list(TRIPLES[i]))
    return result
