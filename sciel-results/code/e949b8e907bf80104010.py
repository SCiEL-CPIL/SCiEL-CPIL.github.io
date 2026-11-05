import time
import random
import sys

N = 256
DEADLINE = 85.0

# Precomputation
# 4-APs: list of (a, b, c, d)
four_aps = []
for a in range(1, N + 1):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = b + d
        e = c + d
        if e <= N:
            four_aps.append((a, b, c, e))

# For each element x, store the bitmasks of the other 3 elements for each 4-AP containing x
# This allows fast check: if (S_mask & mask) == mask, then adding x creates a 4-AP
# Note: x is 1-indexed in logic, but bitmask is 0-indexed (bit 0 is element 1)
elem_to_4ap_masks = {x: [] for x in range(1, N + 1)}
for a, b, c, e in four_aps:
    # Mask for a: b, c, e
    mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
    elem_to_4ap_masks[a].append(mask_a)
    # Mask for b: a, c, e
    mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1))
    elem_to_4ap_masks[b].append(mask_b)
    # Mask for c: a, b, e
    mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1))
    elem_to_4ap_masks[c].append(mask_c)
    # Mask for e: a, b, c
    mask_e = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
    elem_to_4ap_masks[e].append(mask_e)

# Precompute 3-AP partners for fast gain calculation
# first_partners[x]: list of d such that x, x+d, x+2d is a valid 3-AP
# mid_partners[x]: list of d such that x-d, x, x+d is a valid 3-AP
# last_partners[x]: list of d such that x-2d, x-d, x is a valid 3-AP
first_partners = {x: [] for x in range(1, N + 1)}
mid_partners = {x: [] for x in range(1, N + 1)}
last_partners = {x: [] for x in range(1, N + 1)}

for x in range(1, N + 1):
    # x is first term: x, x+d, x+2d
    max_d = (N - x) // 2
    for d in range(1, max_d + 1):
        first_partners[x].append(d)
    
    # x is middle term: x-d, x, x+d
    max_d = min(x - 1, N - x)
    for d in range(1, max_d + 1):
        mid_partners[x].append(d)
        
    # x is last term: x-2d, x-d, x
    max_d = (x - 1) // 2
    for d in range(1, max_d + 1):
        last_partners[x].append(d)

def count_3aps_mask(S_mask):
    """Count 3-APs in the set represented by S_mask."""
    count = 0
    # Iterate over all possible middle terms b
    # If b is in S, count pairs (a, c) in S such that a+c=2b
    # We can just iterate over all 3-APs and check?
    # Precomputed list of all 3-APs is more efficient for counting if we do it once.
    # But for delta, we use partners.
    # For total count, let's use the partner lists to be consistent and fast.
    # Actually, iterating over all b in S and counting pairs is O(N^2) in worst case.
    # Let's precompute a list of all 3-APs (a, b, c) and check bits.
    # Number of 3-APs is ~ N^2/4 = 16k. Checking 16k bitmasks is fast.
    
    # Let's just generate all 3-APs and check.
    total = 0
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = b + d
            if c <= N:
                # Check if a, b, c are in S_mask
                if (S_mask >> (a - 1)) & 1 and (S_mask >> (b - 1)) & 1 and (S_mask >> (c - 1)) & 1:
                    total += 1
    return total

def get_gain(x, S_mask):
    """Calculate the increase in 3-AP count if x is added to the set represented by S_mask.
    Assumes x is NOT in S_mask."""
    gain = 0
    # 1. x is the first term
    for d in first_partners[x]:
        b = x + d
        c = b + d
        # Check if b and c are in S_mask
        if ((S_mask >> (b - 1)) & 1) and ((S_mask >> (c - 1)) & 1):
            gain += 1
            
    # 2. x is the middle term
    for d in mid_partners[x]:
        a = x - d
        c = x + d
        # Check if a and c are in S_mask
        if ((S_mask >> (a - 1)) & 1) and ((S_mask >> (c - 1)) & 1):
            gain += 1
            
    # 3. x is the last term
    for d in last_partners[x]:
        a = x - 2 * d
        b = x - d
        # Check if a and b are in S_mask
        if ((S_mask >> (a - 1)) & 1) and ((S_mask >> (b - 1)) & 1):
            gain += 1
            
    return gain

def get_loss(x, S_mask):
    """Calculate the decrease in 3-AP count if x is removed from the set represented by S_mask.
    Assumes x IS in S_mask."""
    # The loss is the number of 3-APs containing x.
    # This is exactly the same calculation as get_gain, but we check against S_mask (which contains x).
    # However, the pairs (a,c) must not include x.
    # In get_gain, we check if partners are in S_mask.
    # If x is in S_mask, does it interfere?
    # For x as first: partners are b, c > x. x is not b or c. So no interference.
    # For x as middle: partners are a < x, c > x. x is not a or c. So no interference.
    # For x as last: partners are a, b < x. x is not a or b. So no interference.
    # So get_gain(x, S_mask) gives the correct loss if x is in S_mask?
    # Wait. get_gain counts 3-APs formed by x and elements in S_mask.
    # If x is in S_mask, the 3-APs formed by x and other elements in S_mask are exactly the ones lost when x is removed.
    # So yes, get_gain(x, S_mask) works for loss too, provided we don't double count or something.
    # Let's verify.
    # Loss = # 3-APs containing x.
    # Each such 3-AP is formed by x and two other elements in S_mask.
    # get_gain(x, S_mask) counts exactly these.
    return get_gain(x, S_mask)

def check_4ap_violation(x, S_mask):
    """Check if adding x to S (represented by S_mask) creates a 4-AP.
    Assumes x is NOT in S_mask."""
    for mask in elem_to_4ap_masks[x]:
        if (S_mask & mask) == mask:
            return True
    return False

def construct():
    start_time = time.time()
    
    # Greedy Initialization
    S_mask = 0
    # Try to add elements in random order or sorted order?
    # Sorted order often works well for these problems.
    # Let's try a few random greedy runs.
    
    best_score = 0
    best_S_mask = 0
    
    # We will run a local search starting from a greedy solution.
    # To find a good greedy solution, we can try adding elements in increasing order,
    # but maybe skipping some allows better packing?
    # Standard greedy: iterate 1..N, add if valid.
    
    # Let's do a few random restarts of greedy + local search.
    # But 90s is not a lot. Let's do one good run.
    
    # Greedy 1: Increasing order
    S_mask = 0
    for x in range(1, N + 1):
        if not check_4ap_violation(x, S_mask):
            S_mask |= (1 << (x - 1))
            
    current_score = count_3aps_mask(S_mask)
    best_score = current_score
    best_S_mask = S_mask
    
    # Local Search: Simulated Annealing
    T = 5.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute list of elements for random selection
    all_elements = list(range(1, N + 1))
    
    while time.time() - start_time < DEADLINE:
        if T < min_T:
            break
            
        # Choose a move: Add, Remove, or Swap
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            # Find candidates not in S_mask
            # We can iterate all and pick random one not in mask
            # Or maintain a list? Mask is fast.
            # Just pick random x
            x = random.choice(all_elements)
            if (S_mask >> (x - 1)) & 1:
                continue # Already in set
            
            if check_4ap_violation(x, S_mask):
                continue
                
            gain = get_gain(x, S_mask)
            delta = gain
            
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                S_mask |= (1 << (x - 1))
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_S_mask = S_mask
                    
        elif move_type == 'remove':
            # Pick random x in S_mask
            # To pick random x in S_mask, we can find set bits.
            # Or just pick random x and check if in mask.
            # Since N=256, picking random x and checking is fast.
            x = random.choice(all_elements)
            if not ((S_mask >> (x - 1)) & 1):
                continue # Not in set
                
            loss = get_loss(x, S_mask)
            delta = -loss
            
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                S_mask &= ~(1 << (x - 1))
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_S_mask = S_mask
                    
        else: # swap
            # Pick x_add not in S_mask, x_rem in S_mask
            # Try a few times to find valid pair
            for _ in range(10):
                x_add = random.choice(all_elements)
                if (S_mask >> (x_add - 1)) & 1:
                    continue
                x_rem = random.choice(all_elements)
                if not ((S_mask >> (x_rem - 1)) & 1):
                    continue
                
                # Check if adding x_add is valid given x_rem is removed
                # Temporarily remove x_rem from mask for check
                temp_mask = S_mask & ~(1 << (x_rem - 1))
                
                if check_4ap_violation(x_add, temp_mask):
                    continue
                
                # Calculate delta
                # Loss from removing x_rem (in original S_mask)
                # Note: get_loss(x_rem, S_mask) counts 3-APs with x_rem and others in S_mask.
                # Some of these might involve x_add? No, x_add is not in S_mask.
                # So loss is correct.
                loss = get_loss(x_rem, S_mask)
                
                # Gain from adding x_add (to temp_mask)
                gain = get_gain(x_add, temp_mask)
                
                delta = gain - loss
                
                if delta > 0 or random.random() < pow(2.71828, delta / T):
                    # Apply swap
                    S_mask = temp_mask | (1 << (x_add - 1))
                    current_score += delta
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_S_mask = S_mask
                    break # Found a valid move, stop inner loop
            
        T *= cooling
        
    # Convert best_S_mask to list
    result = []
    for x in range(1, N + 1):
        if (best_S_mask >> (x - 1)) & 1:
            result.append(x)
            
    return result

def validate(result):
    sel = set(result)
    for (a, b, c, e) in four_aps:
        if a in sel and b in sel and c in sel and e in sel:
            return False
    return True

if __name__ == "__main__":
    result = construct()
    score = count_3aps_mask(0) # Dummy, just to ensure function exists if called
    # Actual score calculation for printing
    mask = 0
    for x in result:
        mask |= (1 << (x - 1))
    score = count_3aps_mask(mask)
    valid = validate(result)
    print(f"Result size: {len(result)}")
    print(f"3-AP count: {score}")
    print(f"Valid (no 4-AP): {valid}")
