import time
import random
import sys

N = 256
# Precompute 3-APs and 4-APs
# For efficiency, we store:
# 1. For each x, the list of 4-APs it participates in (as tuples of the other 3 elements)
# 2. For each x, the list of 3-APs it participates in (as tuples of the other 2 elements)
# But for faster checking, we use bitmasks.

# Let's use a bitmask for the set A.
# To check if adding x creates a 4-AP:
# We need to know if there exist a, b, c such that {a, b, c, x} is a 4-AP and a,b,c in A.
# Since x is fixed, the 4-APs containing x are limited.
# For each x, precompute the list of "conflict masks": a 4-AP containing x is (x1, x2, x3, x4).
# If x is in the AP, the other 3 elements must NOT all be in A.
# So for each x, we store a list of masks M_j such that (A & M_j) == M_j implies adding x creates a 4-AP.

# Similarly for 3-APs:
# For each x, precompute the list of "gain masks": a 3-AP containing x is (x1, x2, x3).
# If x is in the AP, the other 2 elements being in A implies adding x creates a new 3-AP.
# So for each x, we store a list of masks G_j such that (A & G_j) == G_j implies adding x creates a 3-AP.

conflict_masks = {i: [] for i in range(1, N+1)}
gain_masks = {i: [] for i in range(1, N+1)}

# Generate 4-APs
for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
        # For each element in this 4-AP, the other 3 form a conflict mask
        mask_a = (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
        mask_b = (1 << (a-1)) | (1 << (c-1)) | (1 << (e-1))
        mask_c = (1 << (a-1)) | (1 << (b-1)) | (1 << (e-1))
        mask_e = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
        
        conflict_masks[a].append(mask_a)
        conflict_masks[b].append(mask_b)
        conflict_masks[c].append(mask_c)
        conflict_masks[e].append(mask_e)

# Generate 3-APs
for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        mask_a = (1 << (b-1)) | (1 << (c-1))
        mask_b = (1 << (a-1)) | (1 << (c-1))
        mask_c = (1 << (a-1)) | (1 << (b-1))
        
        gain_masks[a].append(mask_a)
        gain_masks[b].append(mask_b)
        gain_masks[c].append(mask_c)

def is_4ap_free_addition(A_mask, x):
    """Check if adding x to A_mask creates a 4-AP."""
    for mask in conflict_masks[x]:
        if (A_mask & mask) == mask:
            return False
    return True

def delta_3aps_for_addition(A_mask, x):
    """Calculate how many new 3-APs are created by adding x to A_mask."""
    count = 0
    for mask in gain_masks[x]:
        if (A_mask & mask) == mask:
            count += 1
    return count

def count_3aps_mask(A_mask):
    """Count distinct 3-APs in the set represented by A_mask."""
    # We can sum the contributions, but be careful not to double count?
    # No, each 3-AP is counted exactly once per element in it? No.
    # If we sum delta_3aps for all x in A, we count each 3-AP 3 times (once for each element).
    # So count_3aps = sum(delta_3aps_for_addition(A_mask \\ {x}, x) for x in A) / 3?
    # Actually, simpler: just iterate through all 3-APs and check if all 3 are in A.
    # But that's slow.
    # Let's use the property: Total 3-APs = (1/3) * sum_{x in A} (number of 3-APs containing x with other 2 in A).
    # This is exactly (1/3) * sum_{x in A} delta_3aps_for_addition(A_mask \\ {x}, x).
    # But calculating delta requires A_mask \\ {x}.
    # Alternatively, just iterate through the list of all 3-APs.
    # How many 3-APs are there?
    # d=1: 254, d=2: 252, ... d=128: 0.
    # Sum_{d=1}^{128} (256 - 2d) = 256*128 - 2*(128*129/2) = 32768 - 16512 = 16256.
    # 16k checks is fast enough if done occasionally, but maybe not inside SA loop.
    # We will maintain the score incrementally in SA.
    
    # For initial calculation:
    count = 0
    # We can use the gain_masks logic in reverse?
    # For each 3-AP (a,b,c), check if (A_mask & (1<<(a-1)) & (1<<(b-1)) & (1<<(c-1))) != 0
    # Let's precompute a list of 3-AP masks for fast counting.
    pass

# Precompute list of 3-AP masks for fast counting
three_aps_masks = []
for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        three_aps_masks.append((1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)))

def count_3aps_fast(A_mask):
    count = 0
    for m in three_aps_masks:
        if (A_mask & m) == m:
            count += 1
    return count

def greedy_construct(seed_val):
    """
    Randomized greedy construction.
    Order elements randomly. Add element if it doesn't create a 4-AP.
    To bias towards 3-APs, we can prioritize elements that complete more 3-APs.
    """
    random.seed(seed_val)
    A_mask = 0
    candidates = list(range(1, N+1))
    random.shuffle(candidates)
    
    # Sort candidates by "potential" (number of 3-APs they could form with current A?)
    # This is dynamic. Let's just do random order first, then maybe a second pass.
    # Actually, a better greedy: at each step, pick the element x not in A that maximizes
    # delta_3aps_for_addition(A, x) subject to is_4ap_free_addition(A, x).
    # If tie, pick randomly.
    
    remaining = set(range(1, N+1))
    while remaining:
        best_x = None
        best_delta = -1
        best_candidates = []
        
        # Optimization: If we are slow, we might need to restrict candidates.
        # But N=256 is small. 256 * 256 = 65k ops per step? No, delta is O(1) with bitmasks.
        # Conflict check is O(1) with bitmasks.
        # So each step is O(N). Total O(N^2) = 65k ops. Very fast.
        
        for x in remaining:
            if is_4ap_free_addition(A_mask, x):
                delta = delta_3aps_for_addition(A_mask, x)
                if delta > best_delta:
                    best_delta = delta
                    best_candidates = [x]
                elif delta == best_delta:
                    best_candidates.append(x)
        
        if best_candidates:
            x = random.choice(best_candidates)
            A_mask |= (1 << (x-1))
            remaining.remove(x)
        else:
            # No element can be added without creating a 4-AP?
            # This means we are stuck.
            break
            
    return A_mask

def simulated_annealing(A_mask_init, start_temp, end_temp, iterations, deadline):
    """
    Simulated annealing to maximize 3-APs while maintaining 4-AP freedom.
    Moves:
    1. Add x (if valid)
    2. Remove x
    3. Swap x, y (remove x, add y if valid)
    """
    A_mask = A_mask_init
    current_score = count_3aps_fast(A_mask)
    best_mask = A_mask
    best_score = current_score
    
    T = start_temp
    
    for i in range(iterations):
        if time.time() > deadline:
            break
            
        # Update temperature
        T = start_temp * (end_temp / start_temp) ** (i / iterations)
        
        # Choose a move
        move_type = random.random()
        new_mask = A_mask
        delta_score = 0
        
        if move_type < 0.4: # Add
            # Pick random x not in A
            x = random.randint(1, N)
            if not (A_mask & (1 << (x-1))):
                if is_4ap_free_addition(A_mask, x):
                    new_mask = A_mask | (1 << (x-1))
                    delta_score = delta_3aps_for_addition(A_mask, x)
                else:
                    continue # Invalid move, skip
        elif move_type < 0.6: # Remove
            # Pick random x in A
            # Get list of set bits
            bits = [i for i in range(1, N+1) if A_mask & (1 << (i-1))]
            if bits:
                x = random.choice(bits)
                new_mask = A_mask & ~(1 << (x-1))
                # Score change: loss of 3-APs involving x
                # delta_3aps_for_addition(new_mask, x) gives how many 3-APs x forms with new_mask.
                # Since new_mask = A_mask \\ {x}, this is exactly the number of 3-APs lost.
                delta_score = -delta_3aps_for_addition(new_mask, x)
        else: # Swap
            # Pick random x in A, random y not in A
            bits = [i for i in range(1, N+1) if A_mask & (1 << (i-1))]
            if bits:
                x = random.choice(bits)
                y = random.randint(1, N)
                if y != x and not (A_mask & (1 << (y-1))):
                    # Remove x, add y
                    temp_mask = A_mask & ~(1 << (x-1))
                    if is_4ap_free_addition(temp_mask, y):
                        new_mask = temp_mask | (1 << (y-1))
                        # Score change:
                        # Loss from removing x: -delta_3aps_for_addition(temp_mask, x)
                        # Gain from adding y: +delta_3aps_for_addition(temp_mask, y)
                        loss = delta_3aps_for_addition(temp_mask, x)
                        gain = delta_3aps_for_addition(temp_mask, y)
                        delta_score = gain - loss
                    else:
                        continue # Invalid move, skip
                        # Note: if swap is invalid, we skip. 
                        # This might bias the search, but it's okay.
            
        if new_mask == A_mask:
            continue
            
        new_score = current_score + delta_score
        
        # Acceptance probability
        if delta_score > 0:
            accept = True
        else:
            accept = random.random() < pow(2.71828, delta_score / T)
            
        if accept:
            A_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = A_mask
                
    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    best_mask = 0
    best_score = -1
    
    # Multi-start
    num_starts = 20
    time_per_start = (deadline - start_time - 2.0) / num_starts # 2s safety
    
    for i in range(num_starts):
        if time.time() > deadline - 2.0:
            break
            
        # Generate initial solution
        init_mask = greedy_construct(i)
        init_score = count_3aps_fast(init_mask)
        
        if init_score > best_score:
            best_score = init_score
            best_mask = init_mask
            
        # Run SA on this initial solution
        # Allocate time for SA
        sa_start = time.time()
        sa_deadline = min(deadline - 2.0, sa_start + time_per_start)
        
        # Temperature schedule
        # Start high, end low
        start_temp = 10.0
        end_temp = 0.01
        # Iterations: estimate how many we can do
        # Each iteration is ~O(N) = 256 ops. 
        # 10^6 iterations ~ 0.1s? Let's just run until deadline.
        
        sa_mask, sa_score = simulated_annealing(init_mask, start_temp, end_temp, 1000000, sa_deadline)
        
        if sa_score > best_score:
            best_score = sa_score
            best_mask = sa_mask
            
    # Final polish: try to add any remaining elements that don't create 4-APs
    # and increase score
    final_mask = best_mask
    improved = True
    while improved:
        improved = False
        for x in range(1, N+1):
            if not (final_mask & (1 << (x-1))):
                if is_4ap_free_addition(final_mask, x):
                    delta = delta_3aps_for_addition(final_mask, x)
                    if delta > 0:
                        final_mask |= (1 << (x-1))
                        best_score += delta
                        improved = True
        # If no improvement, break
        
    # Convert to list
    result = []
    for i in range(1, N+1):
        if best_mask & (1 << (i-1)):
            result.append(i)
            
    return sorted(result)

# For testing/debugging
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(f"Score: {count_3aps_fast(sum(1 << (x-1) for x in res))}")
