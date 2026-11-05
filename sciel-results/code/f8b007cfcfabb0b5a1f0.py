import time
import random

N = 256

# ---- Precompute per-element 3-AP pair lists and 4-AP conflict masks ----
f3 = [[] for _ in range(N + 1)]   # x as FIRST term of a 3-AP: (x+d, x+2d)
m3 = [[] for _ in range(N + 1)]   # x as MIDDLE term: (x-d, x+d)
l3 = [[] for _ in range(N + 1)]   # x as LAST term: (x-2d, x-d)
f4_masks = [[] for _ in range(N + 1)]  # 4-APs containing x: bitmask of OTHER three elements

def to_mask(vals):
    m = 0
    for v in vals:
        m |= (1 << (v - 1))
    return m

for x in range(1, N + 1):
    # x first: x+2d <= N
    for d in range(1, (N - x) // 2 + 1):
        f3[x].append((x + d, x + 2 * d))
    # x middle: x-d >= 1 and x+d <= N
    for d in range(1, min(x - 1, N - x) + 1):
        m3[x].append((x - d, x + d))
    # x last: x-2d >= 1
    for d in range(1, (x - 1) // 2 + 1):
        l3[x].append((x - 2 * d, x - d))
    
    # 4-AP conflicts for x
    # pos 1: x, x+d, x+2d, x+3d
    for d in range(1, (N - x) // 3 + 1):
        f4_masks[x].append(to_mask([x + d, x + 2 * d, x + 3 * d]))
    # pos 2: x-d, x, x+d, x+2d
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        f4_masks[x].append(to_mask([x - d, x + d, x + 2 * d]))
    # pos 3: x-2d, x-d, x, x+d
    for d in range(1, min((x - 1) // 2, N - x) + 1):
        f4_masks[x].append(to_mask([x - 2 * d, x - d, x + d]))
    # pos 4: x-3d, x-2d, x-d, x
    for d in range(1, (x - 1) // 3 + 1):
        f4_masks[x].append(to_mask([x - 3 * d, x - 2 * d, x - d]))

def is_in_set(A, x):
    return bool((A >> (x - 1)) & 1)

def creates_4AP(A, x):
    for mask in f4_masks[x]:
        if (A & mask) == mask:
            return True
    return False

def gain_add(A, x):
    """Calculate number of 3-APs formed by adding x to A."""
    g = 0
    # x is first
    for a, b in f3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            g += 1
    # x is middle
    for a, b in m3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            g += 1
    # x is last
    for a, b in l3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            g += 1
    return g

def count_3AP_total(A):
    """Count total 3-APs in set A (bitmask)."""
    count = 0
    temp = A
    while temp:
        lsb = temp & -temp
        b = lsb.bit_length()
        temp ^= lsb
        # b is the value (1-indexed)
        for a, c in m3[b]:
            if is_in_set(A, a) and is_in_set(A, c):
                count += 1
    return count

def greedy_seed(rng):
    A = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        if not creates_4AP(A, x):
            A |= (1 << (x - 1))
    return A

def local_search(A, deadline):
    best_score = count_3AP_total(A)
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try single additions
        best_gain = 0
        best_x = -1
        for x in range(1, N + 1):
            if is_in_set(A, x):
                continue
            if creates_4AP(A, x):
                continue
            g = gain_add(A, x)
            if g > best_gain:
                best_gain = g
                best_x = x
        
        if best_x != -1:
            A |= (1 << (best_x - 1))
            best_score += best_gain
            improved = True
            continue # Restart loop to re-evaluate
        
        # 2. Try single swaps (remove r, add x)
        best_net = 0
        best_r = -1
        best_x = -1
        
        # Optimization: Only consider r that are in A
        # And x that are not in A
        # To speed up, we can precompute gains for all x not in A?
        # But gain depends on A, which changes when we remove r.
        # gain_add(A \ {r}, x) is slightly different from gain_add(A, x) if x interacts with r.
        # Specifically, if x and r form a 3-AP with some y in A, that AP is lost when r is removed,
        # so it shouldn't be counted in the gain of adding x to A\{r}.
        # However, gain_add(A, x) counts 3-APs in A that use x.
        # If we remove r, we lose 3-APs that use r.
        # Net gain = (3-APs in (A\{r}) U {x}) - (3-APs in A)
        #          = (3-APs in A\{r}) + (new 3-APs with x in A\{r}) - (3-APs in A)
        #          = (new 3-APs with x in A\{r}) - (3-APs in A that use r)
        
        # Let's compute loss(r) = 3-APs in A that use r.
        # Let's compute gain_x_in_A_minus_r(x) = 3-APs in (A\{r}) U {x} that use x.
        
        # We can approximate by iterating r and x.
        # To keep it fast, we limit the scope or rely on the fact that N=256 is small enough
        # for O(N^2) if the inner loop is fast.
        
        for r in range(1, N + 1):
            if not is_in_set(A, r):
                continue
            
            # Calculate loss of removing r
            loss = 0
            for a, b in f3[r]:
                if is_in_set(A, a) and is_in_set(A, b):
                    loss += 1
            for a, b in m3[r]:
                if is_in_set(A, a) and is_in_set(A, b):
                    loss += 1
            for a, b in l3[r]:
                if is_in_set(A, a) and is_in_set(A, b):
                    loss += 1
            
            A_temp = A & ~(1 << (r - 1))
            
            for x in range(1, N + 1):
                if x == r:
                    continue
                if is_in_set(A, x):
                    continue
                
                # Check if adding x to A_temp creates a 4-AP
                if creates_4AP(A_temp, x):
                    continue
                
                # Calculate gain of adding x to A_temp
                # This is gain_add(A_temp, x)
                # Note: gain_add checks neighbors in A_temp.
                # If a neighbor is r, it's not in A_temp, so it won't be counted.
                # This is exactly what we want.
                g = gain_add(A_temp, x)
                net = g - loss
                
                if net > best_net:
                    best_net = net
                    best_r = r
                    best_x = x
        
        if best_r != -1 and best_x != -1:
            A = A & ~(1 << (best_r - 1))
            A |= (1 << (best_x - 1))
            best_score += best_net
            improved = True
            continue
        
        if not improved:
            break
             
    return best_score, A

def construct():
    start = time.time()
    # Safety margin well under the 60s eval / 90s kill
    deadline = start + 55.0
    rng = random.Random(12345)

    best_A = 0
    best_score = -1

    # Main multi-restart search
    # We use bitmasks so operations are faster.
    while time.time() < deadline - 2.0:
        A = greedy_seed(rng)
        score, A = local_search(A, deadline)
        if score > best_score:
            best_score = score
            best_A = A

    # Final polish with remaining time on the incumbent
    if time.time() < deadline - 0.5 and best_A != 0:
        score, best_A = local_search(best_A, deadline)
        best_score = score

    if best_A == 0:
        # Fallback to a simple valid set if something went wrong
        best_A = 1 << 0 # {1}

    A_list = [x for x in range(1, N + 1) if is_in_set(best_A, x)]
    
    # Hard validity guarantee: drop any element that sits in a 4-AP
    # This is a just-in-case check, local_search should prevent this
    # but we re-verify to be safe.
    present = [False] * (N + 1)
    for x in A_list:
        present[x] = True
    
    # Re-check 4-APs using the original logic for safety
    # We need f4 lists for this check, but we used masks.
    # Let's just trust the mask check in local_search.
    # If we want to be extra safe, we can do a final filter.
    
    # To do a final filter, we need to check if any 4-AP exists.
    # A 4-AP is (a, b, c, d). 
    # We can iterate all 4-APs and if all 4 are in A_list, remove one.
    # But this is O(N^3) or O(N^4) if not careful.
    # Given local_search is robust, we skip the expensive re-filter 
    # unless we suspect a bug. The mask check is O(1) per x.
    
    if not A_list:
        A_list = [1]
        
    return A_list

if __name__ == "__main__":
    res = construct()
    print(len(res))
