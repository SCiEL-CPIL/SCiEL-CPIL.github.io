import time
import random
import math

N = 256

def count_3aps(mask):
    """
    Count the number of 3-term arithmetic progressions in the set represented by mask.
    Bit i represents element i+1.
    """
    count = 0
    # Iterate over all possible differences d
    # Max d is such that 2*d < N
    for d in range(1, N // 2 + 1):
        # For a fixed d, we look for patterns (i, i+d, i+2d)
        # We can use bit shifting to check this efficiently.
        # Let A be the bitset.
        # We want to count i such that bit i, i+d, i+2d are all 1.
        # This is equivalent to popcount(A & (A >> d) & (A >> 2*d))
        # But we must ensure indices are within [0, N-1].
        # The shifts naturally handle the upper bound if we mask appropriately, 
        # but Python integers have arbitrary precision, so we must be careful.
        # Actually, shifting right moves bits to lower indices.
        # If we have bits at i, i+d, i+2d.
        # A has 1 at i.
        # A >> d has 1 at i-d.
        # A >> 2d has 1 at i-2d.
        # This doesn't align them to the same position.
        
        # Alternative:
        # Let B1 = A & (A << d) ? No.
        # Let's stick to a simpler loop for clarity and correctness, 
        # optimized by only iterating over set bits if possible, 
        # or just the O(N^2) loop which is 256^2/2 ~ 30k ops, very fast.
        
        # Actually, the bit trick:
        # We want to count triples (a, a+d, a+2d).
        # Let X = A
        # Let Y = X & (X >> d)  -> Bit i is 1 if i and i+d are in A.
        # Let Z = Y & (X >> 2*d) -> Bit i is 1 if i, i+d, i+2d are in A.
        # Wait, if i is in Y, it means bit i and i+d in X are 1.
        # If we shift X by 2d, bit i in (X >> 2d) corresponds to bit i+2d in X.
        # So Z = (X & (X >> d)) & (X >> 2*d)
        # Let's verify:
        # Bit k in Z is 1 iff:
        # Bit k in X is 1 (k in A)
        # Bit k in (X >> d) is 1 -> Bit k+d in X is 1 (k+d in A)
        # Bit k in (X >> 2d) is 1 -> Bit k+2d in X is 1 (k+2d in A)
        # Yes, this works.
        
        # We need to mask out bits that are out of bounds?
        # Since we only care about bits 0 to N-1, and X only has bits 0 to N-1 set,
        # the shifts will move 1s to positions < 0 (which become 0 in Python int) 
        # or positions > N-1 (which are 0 in X, so 0 in result).
        # So no explicit mask is needed for the count, just the operation.
        
        z = (mask & (mask >> d)) & (mask >> (2 * d))
        count += z.bit_count()
        
    return count

def has_4ap_with_new(x, mask):
    """
    Check if adding element x (index x-1) to the set represented by mask
    creates a 4-term arithmetic progression.
    x is 1-indexed in the problem, so bit index is x-1.
    """
    idx = x - 1
    # x can be the 1st, 2nd, 3rd, or 4th term of a 4-AP.
    
    # 1. x is 1st term: x, x+d, x+2d, x+3d
    # Bits: idx, idx+d, idx+2d, idx+3d
    # We need idx+d, idx+2d, idx+3d to be in mask.
    # d can be 1 to (N-1 - idx) // 3
    max_d = (N - 1 - idx) // 3
    for d in range(1, max_d + 1):
        if (mask & (1 << (idx + d))) and (mask & (1 << (idx + 2 * d))) and (mask & (1 << (idx + 3 * d))):
            return True
            
    # 2. x is 2nd term: x-d, x, x+d, x+2d
    # Bits: idx-d, idx, idx+d, idx+2d
    # We need idx-d, idx+d, idx+2d to be in mask.
    # d can be 1 to min(idx, (N-1 - idx) // 2)
    max_d = min(idx, (N - 1 - idx) // 2)
    for d in range(1, max_d + 1):
        if (mask & (1 << (idx - d))) and (mask & (1 << (idx + d))) and (mask & (1 << (idx + 2 * d))):
            return True
            
    # 3. x is 3rd term: x-2d, x-d, x, x+d
    # Bits: idx-2d, idx-d, idx, idx+d
    # We need idx-2d, idx-d, idx+d to be in mask.
    # d can be 1 to min(idx // 2, (N-1 - idx) // 1) -> idx//2 is the limit for idx-2d >= 0
    max_d = min(idx // 2, N - 1 - idx)
    for d in range(1, max_d + 1):
        if (mask & (1 << (idx - 2 * d))) and (mask & (1 << (idx - d))) and (mask & (1 << (idx + d))):
            return True
            
    # 4. x is 4th term: x-3d, x-2d, x-d, x
    # Bits: idx-3d, idx-2d, idx-d, idx
    # We need idx-3d, idx-2d, idx-d to be in mask.
    # d can be 1 to idx // 3
    max_d = idx // 3
    for d in range(1, max_d + 1):
        if (mask & (1 << (idx - 3 * d))) and (mask & (1 << (idx - 2 * d))) and (mask & (1 << (idx - d))):
            return True
            
    return False

def has_4ap(mask):
    """
    Check if the set represented by mask contains any 4-AP.
    """
    # Iterate over all possible 4-APs
    # a, a+d, a+2d, a+3d
    # a from 0 to N-1, d from 1 to ...
    # This is O(N^2) roughly.
    for d in range(1, N // 3 + 1):
        # Check if there is any a such that a, a+d, a+2d, a+3d are all in mask
        # Pattern: 1111 in binary, shifted by d? No, spaced by d.
        # We can use the bit trick:
        # Y = mask & (mask >> d)
        # Z = Y & (mask >> 2*d)
        # W = Z & (mask >> 3*d)
        # If W != 0, there is a 4-AP.
        w = ((mask & (mask >> d)) & (mask >> (2 * d))) & (mask >> (3 * d))
        if w != 0:
            return True
    return False

def greedy_construct():
    """
    Construct an initial set using a greedy strategy.
    """
    mask = 0
    # Try to add elements in a random order or sorted order
    # Sorted order might be bad for randomness, let's use a specific heuristic
    # or just random.
    elements = list(range(1, N + 1))
    random.shuffle(elements)
    
    for x in elements:
        if not has_4ap_with_new(x, mask):
            mask |= (1 << (x - 1))
            
    return mask

def simulated_annealing(initial_mask, time_limit=80):
    """
    Run Simulated Annealing to optimize the number of 3-APs.
    """
    start_time = time.time()
    mask = initial_mask
    current_score = count_3aps(mask)
    
    best_mask = mask
    best_score = current_score
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.001
    
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            T = min_T
            # Restart or stop? Let's stop for now, or restart with higher T.
            # For simplicity, stop.
            break
            
        # Generate a random move
        move_type = rng.random()
        
        new_mask = mask
        valid_move = False
        
        if move_type < 0.4:
            # Add an element
            # Pick a random element not in mask
            # We can pick a random index and check if it's in mask
            # If in mask, retry or pick another.
            # Better: create a list of unset bits? Expensive.
            # Just pick random x.
            x = rng.randint(1, N)
            idx = x - 1
            if not (mask & (1 << idx)):
                if not has_4ap_with_new(x, mask):
                    new_mask = mask | (1 << idx)
                    valid_move = True
        elif move_type < 0.7:
            # Remove an element
            # Pick a random element in mask
            # We can pick a random index and check if it's in mask
            x = rng.randint(1, N)
            idx = x - 1
            if (mask & (1 << idx)):
                new_mask = mask & ~(1 << idx)
                valid_move = True
        else:
            # Swap: Remove one, add one
            x_out = rng.randint(1, N)
            x_in = rng.randint(1, N)
            idx_out = x_out - 1
            idx_in = x_in - 1
            
            # Ensure x_out is in mask and x_in is not
            if (mask & (1 << idx_out)) and not (mask & (1 << idx_in)):
                # Try removing x_out
                temp_mask = mask & ~(1 << idx_out)
                # Check if adding x_in to temp_mask is valid
                if not has_4ap_with_new(x_in, temp_mask):
                    new_mask = temp_mask | (1 << idx_in)
                    valid_move = True

        if not valid_move:
            T *= cooling_rate
            continue

        new_score = count_3aps(new_mask)
        delta = new_score - current_score
        
        if delta > 0:
            # Accept
            mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = mask
        else:
            # Accept with probability exp(delta / T)
            if T > 0:
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    mask = new_mask
                    current_score = new_score
                    
        T *= cooling_rate

    return best_mask, best_score

def construct():
    """
    Main entry point.
    """
    deadline = time.time() + 85  # Leave 5s margin
    
    # Phase 1: Greedy Construction
    initial_mask = greedy_construct()
    
    # Phase 2: Simulated Annealing
    # We can run multiple short SA runs or one long one.
    # Let's run one long SA run with a time limit.
    best_mask, best_score = simulated_annealing(initial_mask, time_limit=60)
    
    # Phase 3: Local Search Polish
    # Use the best mask from SA and do a quick local search (add/swap) to polish.
    # This is similar to the original local search but starts from a better point.
    start_time = time.time()
    mask = best_mask
    current_score = best_score
    improved = True
    while improved and time.time() - start_time < (deadline - time.time()):
        improved = False
        # Try adding
        for x in range(1, N + 1):
            if not (mask & (1 << (x - 1))):
                if not has_4ap_with_new(x, mask):
                    new_mask = mask | (1 << (x - 1))
                    new_score = count_3aps(new_mask)
                    if new_score > current_score:
                        mask = new_mask
                        current_score = new_score
                        improved = True
                        break
        if improved:
            continue
            
        # Try swapping
        for x_out in range(1, N + 1):
            if not (mask & (1 << (x_out - 1))):
                continue
            for x_in in range(1, N + 1):
                if mask & (1 << (x_in - 1)):
                    continue
                temp_mask = mask & ~(1 << (x_out - 1))
                if not has_4ap_with_new(x_in, temp_mask):
                    new_mask = temp_mask | (1 << (x_in - 1))
                    new_score = count_3aps(new_mask)
                    if new_score > current_score:
                        mask = new_mask
                        current_score = new_score
                        improved = True
                        break
            if improved:
                break
                
    # Convert mask to list of elements
    result = []
    for i in range(N):
        if mask & (1 << i):
            result.append(i + 1)
            
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Cardinality: {len(result)}")
    print(f"3-AP count: {count_3aps(sum(1 << (x-1) for x in result))}")
    print(f"4-AP check: {has_4ap(sum(1 << (x-1) for x in result))}")
