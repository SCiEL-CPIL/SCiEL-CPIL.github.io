import time
import random

N = 256

def construct():
    # Precomputation for 4-AP conflicts
    # For each x, list of masks of other 3 elements in 4-APs containing x
    four_ap_masks = [[] for _ in range(N + 1)]
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            d_val = a + 3 * d
            
            # Convert to 0-indexed bits
            # Elements: a, b, c, d_val
            
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (d_val - 1))
            four_ap_masks[a].append(mask_a)
            
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (d_val - 1))
            four_ap_masks[b].append(mask_b)
            
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (d_val - 1))
            four_ap_masks[c].append(mask_c)
            
            mask_d = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            four_ap_masks[d_val].append(mask_d)

    # Precomputation for 3-AP counts
    # For each x, list of (y, z) pairs such that (x,y,z), (y,x,z), (y,z,x) form a 3-AP
    # We need to count how many 3-APs are formed when x is added.
    # added_3aps(x, mask) = count of (y,z) in mask such that {x,y,z} is a 3-AP
    
    # To make this fast, we can precompute for each x a list of "partners".
    # But checking (y in mask) and (z in mask) is O(1) with bit ops.
    # We just need to iterate over possible d.
    
    def count_3aps_involving(x, A_mask):
        count = 0
        # x is first: (x, x+d, x+2d)
        for d in range(1, (N - x) // 2 + 1):
            b = x + d
            c = x + 2 * d
            if b > N or c > N: break
            if ((A_mask >> (b - 1)) & 1) and ((A_mask >> (c - 1)) & 1):
                count += 1
        # x is second: (x-d, x, x+d)
        for d in range(1, min(x - 1, N - x) + 1):
            a = x - d
            c = x + d
            if a < 1 or c > N: break
            if ((A_mask >> (a - 1)) & 1) and ((A_mask >> (c - 1)) & 1):
                count += 1
        # x is third: (x-2d, x-d, x)
        for d in range(1, (x - 1) // 2 + 1):
            a = x - 2 * d
            b = x - d
            if a < 1: break
            if ((A_mask >> (a - 1)) & 1) and ((A_mask >> (b - 1)) & 1):
                count += 1
        return count

    def is_valid_add(A_mask, x):
        idx = x - 1
        if (A_mask >> idx) & 1:
            return False
        for mask in four_ap_masks[x]:
            if (A_mask & mask) == mask:
                return False
        return True

    # Initialize with a random valid set using greedy
    A_mask = 0
    current_score = 0
    # Randomize the order of addition to get different starting points
    elements = list(range(1, N + 1))
    random.shuffle(elements)
    
    for x in elements:
        if is_valid_add(A_mask, x):
            A_mask |= (1 << (x - 1))
            current_score += count_3aps_involving(x, A_mask)

    best_mask = A_mask
    best_score = current_score
    
    # Simulated Annealing
    start_time = time.time()
    timeout = 55.0
    
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # To speed up, we can precompute the list of valid moves? No, validity depends on mask.
    # We will just pick random x and check.
    
    while (time.time() - start_time) < timeout:
        if T < min_T:
            break
            
        x = random.randint(1, N)
        idx = x - 1
        in_set = (A_mask >> idx) & 1
        
        if in_set:
            # Try removing x
            # Removal is always valid
            delta_score = -count_3aps_involving(x, A_mask)
            new_mask = A_mask & ~(1 << idx)
            new_score = current_score + delta_score
            
            # Accept if better or with prob exp(delta/T)
            if delta_score > 0:
                A_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = A_mask
            else:
                # delta_score <= 0
                # Use random to accept
                # To avoid math.exp overhead, we can use a simple comparison if we precompute? 
                # Or just use math.exp. It's fast enough in Python for 60s.
                import math
                if random.random() < math.exp(delta_score / T):
                    A_mask = new_mask
                    current_score = new_score
        else:
            # Try adding x
            if is_valid_add(A_mask, x):
                delta_score = count_3aps_involving(x, A_mask)
                new_mask = A_mask | (1 << idx)
                new_score = current_score + delta_score
                
                if delta_score > 0:
                    A_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = A_mask
                else:
                    import math
                    if random.random() < math.exp(delta_score / T):
                        A_mask = new_mask
                        current_score = new_score
                        
        T *= cooling

    return sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
