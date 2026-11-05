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
    for a, b in f3[x]:
        if ((A >> (a-1)) & 1) and ((A >> (b-1)) & 1):
            g += 1
    for a, b in m3[x]:
        if ((A >> (a-1)) & 1) and ((A >> (b-1)) & 1):
            g += 1
    for a, b in l3[x]:
        if ((A >> (a-1)) & 1) and ((A >> (b-1)) & 1):
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
        for a, c in m3[b]:
            if ((A >> (a-1)) & 1) and ((A >> (c-1)) & 1):
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

def reverse_greedy_seed(rng):
    """Start with all elements, remove violators in random order."""
    A = (1 << N) - 1
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        # Check if x is part of any 4-AP in current A
        for mask in f4_masks[x]:
            if (A & mask) == mask:
                # x is in a 4-AP; remove x
                A &= ~(1 << (x - 1))
                break
    return A

def local_search(A, deadline):
    best_score = count_3AP_total(A)
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try single additions
        best_gain = 0
        best_x = -1
        for x in range(1, N + 1):
            if (A >> (x - 1)) & 1:
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
            continue
        
        # 2. Try single swaps (remove r, add x)
        # Precompute loss for each r in A
        loss_arr = [0] * (N + 1)
        in_A = [False] * (N + 1)
        for r in range(1, N + 1):
            if (A >> (r - 1)) & 1:
                in_A[r] = True
                loss = 0
                for a, b in f3[r]:
                    if ((A >> (a-1)) & 1) and ((A >> (b-1)) & 1):
                        loss += 1
                for a, b in m3[r]:
                    if ((A >> (a-1)) & 1) and ((A >> (b-1)) & 1):
                        loss += 1
                for a, b in l3[r]:
                    if ((A >> (a-1)) & 1) and ((A >> (b-1)) & 1):
                        loss += 1
                loss_arr[r] = loss
        
        # Precompute gain_add(A, x) for all x not in A
        gain_arr = [0] * (N + 1)
        for x in range(1, N + 1):
            if not in_A[x]:
                gain_arr[x] = gain_add(A, x)
        
        best_net = 0
        best_r = -1
        best_x = -1
        
        for r in range(1, N + 1):
            if not in_A[r]:
                continue
            loss = loss_arr[r]
            A_temp = A & ~(1 << (r - 1))
            
            for x in range(1, N + 1):
                if in_A[x]:
                    continue
                
                # Check if adding x to A_temp creates a 4-AP
                if creates_4AP(A_temp, x):
                    continue
                
                # Adjust gain: subtract 3-APs that used r as a partner
                # gain_add(A, x) counts pairs (a,b) in A such that x,a,b form 3-AP
                # In A_temp, r is removed, so any pair involving r is lost
                # We need to subtract the count of 3-APs in A that use both x and r
                # i.e., pairs where one element is r and the other is in A, and x,r,other form 3-AP
                g = gain_arr[x]
                # Subtract contributions where r was one of the two elements
                # x first: r is at position 2 or 3
                for a, b in f3[x]:
                    if a == r and ((A >> (b-1)) & 1):
                        g -= 1
                    elif b == r and ((A >> (a-1)) & 1):
                        g -= 1
                # x middle: r is at position 1 or 3
                for a, b in m3[x]:
                    if a == r and ((A >> (b-1)) & 1):
                        g -= 1
                    elif b == r and ((A >> (a-1)) & 1):
                        g -= 1
                # x last: r is at position 1 or 2
                for a, b in l3[x]:
                    if a == r and ((A >> (b-1)) & 1):
                        g -= 1
                    elif b == r and ((A >> (a-1)) & 1):
                        g -= 1
                
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
    deadline = start + 55.0
    rng = random.Random(12345)

    best_A = 0
    best_score = -1

    # Main multi-restart search with both seed strategies
    while time.time() < deadline - 2.0:
        A = greedy_seed(rng)
        score, A = local_search(A, deadline)
        if score > best_score:
            best_score = score
            best_A = A
        
        if time.time() < deadline - 2.0:
            A = reverse_greedy_seed(rng)
            score, A = local_search(A, deadline)
            if score > best_score:
                best_score = score
                best_A = A

    # Final polish with remaining time on the incumbent
    if time.time() < deadline - 0.5 and best_A != 0:
        score, best_A = local_search(best_A, deadline)
        best_score = score

    if best_A == 0:
        best_A = 1 << 0

    A_list = [x for x in range(1, N + 1) if (best_A >> (x - 1)) & 1]
    
    if not A_list:
        A_list = [1]
        
    return A_list

if __name__ == "__main__":
    res = construct()
    print(len(res))
