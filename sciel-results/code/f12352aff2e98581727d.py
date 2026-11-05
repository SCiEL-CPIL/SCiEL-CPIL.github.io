import time
import random

N = 256

# Precomputation
# 1. All 3-APs
all_3aps = []
ap3_containing = [[] for _ in range(N + 1)]
for d in range(1, N):
    for a in range(1, N - 2 * d + 1):
        b = a + d
        c = a + 2 * d
        all_3aps.append((a, b, c))
        ap3_containing[a].append((b, c))
        ap3_containing[b].append((a, c))
        ap3_containing[c].append((a, b))

# 2. All 4-APs and conflict masks
# conflict_masks[x] will contain bitmasks of the other 3 elements in any 4-AP containing x
conflict_masks = [[] for _ in range(N + 1)]
for d in range(1, N):
    for a in range(1, N - 3 * d + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        # The 4-AP is (a, b, c, e)
        # For each element, the other three form a conflict mask
        mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
        mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1))
        mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1))
        mask_e = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
        
        conflict_masks[a].append(mask_a)
        conflict_masks[b].append(mask_b)
        conflict_masks[c].append(mask_c)
        conflict_masks[e].append(mask_e)

def bit(A, x):
    return (A >> (x - 1)) & 1

def creates_4AP(A, x):
    for mask in conflict_masks[x]:
        if (A & mask) == mask:
            return True
    return False

def count_3APs(A):
    count = 0
    for a, b, c in all_3aps:
        if bit(A, a) and bit(A, b) and bit(A, c):
            count += 1
    return count

def gain_add(A, x):
    """Calculate increase in 3-AP count if x is added to A."""
    gain = 0
    for y, z in ap3_containing[x]:
        if bit(A, y) and bit(A, z):
            gain += 1
    return gain

def loss_remove(A, x):
    """Calculate decrease in 3-AP count if x is removed from A."""
    loss = 0
    for y, z in ap3_containing[x]:
        if bit(A, y) and bit(A, z):
            loss += 1
    return loss

def greedy_seed(rng):
    A = 0
    # Random order for greedy
    elements = list(range(1, N + 1))
    rng.shuffle(elements)
    
    for x in elements:
        if creates_4AP(A, x):
            continue
        # Heuristic: prefer elements with high gain
        # To speed up, just add if valid. 
        # A better heuristic: sort by gain dynamically? Too slow.
        # Let's just add valid elements in random order.
        A |= (1 << (x - 1))
    return A

def local_search(A, deadline):
    """Hill climbing with swaps."""
    current_score = count_3APs(A)
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try adds
        for x in range(1, N + 1):
            if bit(A, x):
                continue
            if creates_4AP(A, x):
                continue
            g = gain_add(A, x)
            if g > 0:
                A |= (1 << (x - 1))
                current_score += g
                improved = True
                # Restart search after improvement
                break
        
        if improved:
            continue
            
        # 2. Try swaps
        # We need to find (x in A, y not in A) such that
        # gain_add(A \ {x}, y) - loss_remove(A, x) > 0
        # And (A \ {x}) U {y} is valid (no 4-APs)
        
        # Optimization: Precompute valid swaps? No, A changes.
        # Iterate x in A, y not in A.
        # To make this faster, we can limit the search or use heuristics.
        # Given N=256, |A| is likely ~100-150.
        # Pairs to check: ~100 * 100 = 10,000.
        # Each check involves:
        # 1. Validity of new A: Check 4-APs for y in A\x.
        #    We can check if y creates 4AP with A\x.
        #    Note: Removing x might *remove* a 4-AP that y was part of? No, y is not in A.
        #    So we just need to check if y creates a 4-AP with the elements in A\x.
        #    This is `creates_4AP(A ^ (1 << (x-1)), y)`.
        # 2. Score change: gain_add(A\x, y) - loss_remove(A, x).
        
        best_net_gain = 0
        best_x = None
        best_y = None
        
        # To speed up, we can iterate y first?
        # Let's stick to x, y.
        
        for x in range(1, N + 1):
            if not bit(A, x):
                continue
            loss_x = loss_remove(A, x)
            A_without_x = A & ~(1 << (x - 1))
            
            for y in range(1, N + 1):
                if bit(A, y):
                    continue
                # Check validity of adding y to A_without_x
                if creates_4AP(A_without_x, y):
                    continue
                gain_y = gain_add(A_without_x, y)
                net = gain_y - loss_x
                if net > best_net_gain:
                    best_net_gain = net
                    best_x = x
                    best_y = y
                    
        if best_net_gain > 0:
            # Perform swap
            A &= ~(1 << (best_x - 1))
            A |= (1 << (best_y - 1))
            current_score += best_net_gain
            # Loop continues to try more adds/swaps

    return A

def simulated_annealing(A, deadline):
    """SA polish."""
    rng = random.Random()
    temp = 5.0
    cooling = 0.99
    current_score = count_3APs(A)
    
    while time.time() < deadline and temp > 0.1:
        # Propose move: Add, Remove, or Swap
        move_type = rng.random()
        
        if move_type < 0.3:
            # Add
            x = rng.randint(1, N)
            if bit(A, x) or creates_4AP(A, x):
                temp *= cooling
                continue
            A_new = A | (1 << (x - 1))
            delta = gain_add(A, x)
        elif move_type < 0.6:
            # Remove
            present = [x for x in range(1, N + 1) if bit(A, x)]
            if not present:
                temp *= cooling
                continue
            x = rng.choice(present)
            A_new = A & ~(1 << (x - 1))
            delta = -loss_remove(A, x)
        else:
            # Swap
            present = [x for x in range(1, N + 1) if bit(A, x)]
            if not present:
                temp *= cooling
                continue
            x = rng.choice(present)
            y = rng.randint(1, N)
            if x == y:
                temp *= cooling
                continue
            if bit(A, y):
                temp *= cooling
                continue
            A_without_x = A & ~(1 << (x - 1))
            if creates_4AP(A_without_x, y):
                temp *= cooling
                continue
            delta = gain_add(A_without_x, y) - loss_remove(A, x)
            A_new = A_without_x | (1 << (y - 1))
            
        # Acceptance
        if delta >= 0 or rng.random() < (2.718281828 ** (delta / temp)):
            A = A_new
            current_score += delta
        temp *= cooling
        
    return A

def construct():
    start = time.time()
    deadline = start + 55.0  # Safety margin
    
    best_A = 0
    best_score = -1
    
    rng = random.Random(42)
    
    # Multi-restart
    while time.time() < deadline - 5.0:
        A = greedy_seed(rng)
        # Quick local search
        A = local_search(A, min(deadline - 5.0, time.time() + 2.0))
        score = count_3APs(A)
        
        if score > best_score:
            best_score = score
            best_A = A
            
    # SA polish on the best found
    if time.time() < deadline - 1.0 and best_A:
        A = simulated_annealing(best_A, deadline - 0.5)
        score = count_3APs(A)
        if score > best_score:
            best_score = score
            best_A = A
            
    # Final local search
    if time.time() < deadline - 0.5 and best_A:
        A = local_search(best_A, deadline - 0.1)
        score = count_3APs(A)
        if score > best_score:
            best_score = score
            best_A = A
            
    A_list = [x for x in range(1, N + 1) if bit(best_A, x)]
    if not A_list:
        A_list = [1]
        
    return A_list
