import random
import time

N = 256
three_aps = []
four_aps = []
three_aps_by_elem = {i: [] for i in range(1, N + 1)}
four_aps_by_elem = {i: [] for i in range(1, N + 1)}

for d in range(1, N + 1):
    for a in range(1, N - 2 * d + 1):
        b = a + d
        c = a + 2 * d
        m = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
        three_aps.append((m, a, b, c))
        three_aps_by_elem[a].append(m)
        three_aps_by_elem[b].append(m)
        three_aps_by_elem[c].append(m)

    for a in range(1, N - 3 * d + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        m = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
        four_aps.append((m, a, b, c, e))
        four_aps_by_elem[a].append((m, b, c, e))
        four_aps_by_elem[b].append((m, a, c, e))
        four_aps_by_elem[c].append((m, a, b, e))
        four_aps_by_elem[e].append((m, a, b, c))

def count_3aps_mask(mask):
    count = 0
    for m, _, _, _ in three_aps:
        if mask & m == m:
            count += 1
    return count

def find_4ap_mask(mask):
    for m, a, b, c, e in four_aps:
        if mask & m == m:
            return (a, b, c, e)
    return None

def construct():
    N = 256
    full_mask = (1 << N) - 1
    best_score = -1
    best_mask = 0
    
    start_time = time.time()
    time_limit = 55.0
    
    # Run multiple restarts
    while time.time() - start_time < time_limit:
        mask = full_mask
        # Randomize tie-breaking by shuffling order of elements in APs?
        # Or just use random selection among min-loss elements.
        
        # Phase 1: Remove to eliminate 4-APs
        while True:
            ap = find_4ap_mask(mask)
            if not ap:
                break
            
            min_loss = float('inf')
            candidates = []
            
            for x in ap:
                # Count 3-APs containing x in current mask
                c = 0
                for m in three_aps_by_elem[x]:
                    if mask & m == m:
                        c += 1
                if c < min_loss:
                    min_loss = c
                    candidates = [x]
                elif c == min_loss:
                    candidates.append(x)
            
            # Pick random candidate to break ties
            x_to_remove = random.choice(candidates)
            mask &= ~(1 << (x_to_remove - 1))
        
        # Phase 2: Add back elements
        changed = True
        while changed:
            changed = False
            # Try to add any element not in mask
            # We can iterate through 1..N
            # To be smarter, we can try elements that were removed?
            # For now, scan all.
            for x in range(1, N + 1):
                if not (mask & (1 << (x - 1))):
                    # Check if adding x creates a 4-AP
                    creates = False
                    for m, others in four_aps_by_elem[x]:
                        # Check if all 'others' are in mask
                        # m has bits for x and others.
                        # We need to check if (mask | (1 << (x-1))) & m == m
                        # Which is equivalent to checking if (mask & m) == (m & ~(1 << (x-1)))
                        # i.e. all other bits are set.
                        other_mask = m & ~(1 << (x - 1))
                        if mask & other_mask == other_mask:
                            creates = True
                            break
                    if not creates:
                        mask |= (1 << (x - 1))
                        changed = True
        
        # Evaluate
        score = count_3aps_mask(mask)
        if score > best_score:
            best_score = score
            best_mask = mask
            
    # Convert best_mask to sorted list
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
    return result
