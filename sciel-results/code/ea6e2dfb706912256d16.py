import time
import random
import math

N = 256
DEADLINE = 55.0  # Safety margin under 90s kill

# Precomputation
# For each x, store the other two elements for every 3-AP containing x
# ap3_partners[x] is a list of tuples (y, z) such that {x, y, z} is a 3-AP
# We store y, z as 0-indexed or 1-indexed? Let's use 0-indexed for bitmask shifts.
# Elements are 1..N. Bit i corresponds to element i+1.

ap3_partners = [[] for _ in range(N)]
ap4_partners = [[] for _ in range(N)]

def precompute():
    # 3-APs: a, a+d, a+2d
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            # Indices: a-1, b-1, c-1
            idx_a, idx_b, idx_c = a - 1, b - 1, c - 1
            
            # For a: partners are b, c
            ap3_partners[idx_a].append((idx_b, idx_c))
            # For b: partners are a, c
            ap3_partners[idx_b].append((idx_a, idx_c))
            # For c: partners are a, b
            ap3_partners[idx_c].append((idx_a, idx_b))

    # 4-APs: a, a+d, a+2d, a+3d
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # Indices: a-1, b-1, c-1, e-1
            idx_a, idx_b, idx_c, idx_e = a - 1, b - 1, c - 1, e - 1
            
            # For a: partners are b, c, e
            ap4_partners[idx_a].append((idx_b, idx_c, idx_e))
            # For b: partners are a, c, e
            ap4_partners[idx_b].append((idx_a, idx_c, idx_e))
            # For c: partners are a, b, e
            ap4_partners[idx_c].append((idx_a, idx_b, idx_e))
            # For e: partners are a, b, c
            ap4_partners[idx_e].append((idx_a, idx_b, idx_c))

precompute()

def has_4ap_with_mask(mask, x_idx):
    """Check if adding x_idx to mask creates a 4-AP."""
    for y, z, w in ap4_partners[x_idx]:
        if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
            return True
    return False

def delta_3ap_add(mask, x_idx):
    """Calculate gain in 3-APs if x_idx is added to mask."""
    gain = 0
    for y, z in ap3_partners[x_idx]:
        if (mask >> y) & 1 and (mask >> z) & 1:
            gain += 1
    return gain

def delta_3ap_remove(mask, x_idx):
    """Calculate loss in 3-APs if x_idx is removed from mask."""
    # The APs lost are those involving x_idx and two other elements currently in mask.
    # This is exactly the same calculation as delta_3ap_add, because if {x, y, z} is in mask,
    # removing x destroys it.
    loss = 0
    for y, z in ap3_partners[x_idx]:
        if (mask >> y) & 1 and (mask >> z) & 1:
            loss += 1
    return loss

def count_3aps(mask):
    """Count total 3-APs in mask."""
    # Efficient count:
    # Each 3-AP is counted 3 times if we sum delta_add for each member.
    total = 0
    for x in range(N):
        if (mask >> x) & 1:
            total += delta_3ap_add(mask ^ (1 << x), x)
    return total // 3

def greedy_seed():
    """Build a maximal 4-AP-free set."""
    mask = 0
    for x in range(N):
        if not has_4ap_with_mask(mask, x):
            mask |= (1 << x)
    return mask

def local_search(mask, current_score, max_iter=10000, timeout=None, start_time=None):
    """
    Perform a local search to improve the 3-AP count.
    Moves: Add an element, Remove an element, Swap (Remove one, Add another).
    """
    if timeout is None:
        timeout = 5.0
    if start_time is None:
        start_time = time.time()
        
    improved = True
    while improved:
        improved = False
        for _ in range(max_iter):
            if timeout and time.time() - start_time > timeout:
                return mask, current_score
            
            # 1. Try single additions
            added = False
            for x in range(N):
                if (mask >> x) & 1:
                    continue
                if has_4ap_with_mask(mask, x):
                    continue
                delta = delta_3ap_add(mask, x)
                if delta > 0:
                    mask |= (1 << x)
                    current_score += delta
                    improved = True
                    added = True
                    break
            if added:
                continue
                
            # 2. Try single removals (rarely helps if we want to maximize, but might unlock swaps)
            # Actually, removals only help if they allow a better addition later. 
            # Pure removals reduce score, so we only do them as part of a swap or if stuck.
            # Let's stick to swaps for efficiency.
            
            # 3. Try swaps: Remove x, Add y
            # To be efficient, we can just try random swaps or systematic ones.
            # Systematic O(N^2) is 256^2 = 65k checks. Each check is O(1) for 4AP and O(degree) for 3AP delta.
            # Degree is small. Let's do a random sample of swaps to avoid O(N^2) overhead per iteration.
            
            # Pick random x to remove
            present_bits = [i for i in range(N) if (mask >> i) & 1]
            if not present_bits:
                break
            x = random.choice(present_bits)
            if not (mask >> x) & 1:
                continue
                
            # Calculate loss of removing x
            loss = delta_3ap_remove(mask, x)
            mask_without_x = mask ^ (1 << x)
            
            # Try to add a random y
            absent_bits = [i for i in range(N) if not (mask >> i) & 1]
            if not absent_bits:
                break
            y = random.choice(absent_bits)
            
            # Check if adding y to mask_without_x is valid
            if has_4ap_with_mask(mask_without_x, y):
                continue
                
            gain = delta_3ap_add(mask_without_x, y)
            net_delta = gain - loss
            
            if net_delta > 0:
                mask = mask_without_x | (1 << y)
                current_score += net_delta
                improved = True
                break # Restart scan after improvement
                
            # If net_delta <= 0, we might still accept it with some probability?
            # No, this is hill climbing. Let's keep it strict for now.
                
    return mask, current_score

def construct():
    start_time = time.time()
    random.seed(12345)
    
    # Initial solution: Greedy
    mask = greedy_seed()
    current_score = count_3aps(mask)
    
    best_mask = mask
    best_score = current_score
    
    # Local search phase 1
    mask, current_score = local_search(mask, current_score, max_iter=5000, timeout=10.0, start_time=start_time)
    if current_score > best_score:
        best_score = current_score
        best_mask = mask
        
    # Simulated Annealing Phase
    # We use SA to escape local optima.
    # Moves: Flip a bit (add or remove).
    # Constraint: 4-AP free.
    # Objective: Maximize 3-APs.
    
    # To handle constraints in SA, we can:
    # 1. Only allow valid moves (strict).
    # 2. Allow invalid moves with a penalty.
    
    # Let's try strict valid moves first, but with a "swap" move type to ensure cardinality changes are balanced or controlled.
    # Actually, the previous local search did swaps.
    # Let's do SA with single-bit flips (add/remove) but only if valid.
    # If stuck, do a swap.
    
    T_start = 10.0
    T_end = 0.1
    alpha = math.log(T_start / T_end) / 40.0 # Cool over 40 seconds
    
    last_improve_time = time.time()
    
    while time.time() - start_time < DEADLINE:
        elapsed = time.time() - start_time
        T = T_start * math.exp(-alpha * elapsed)
        
        # Choose move type
        # 70% single flip, 30% swap
        if random.random() < 0.7:
            # Single flip
            x_idx = random.randint(0, N - 1)
            bit = 1 << x_idx
            is_present = (mask >> x_idx) & 1
            
            if is_present:
                # Try to remove
                new_mask = mask ^ bit
                delta = -delta_3ap_remove(mask, x_idx)
                # Always valid to remove
                if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                    mask = new_mask
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = mask
                        last_improve_time = time.time()
            else:
                # Try to add
                if has_4ap_with_mask(mask, x_idx):
                    continue
                new_mask = mask | bit
                delta = delta_3ap_add(mask, x_idx)
                if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                    mask = new_mask
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = mask
                        last_improve_time = time.time()
        else:
            # Swap move
            present_bits = [i for i in range(N) if (mask >> i) & 1]
            absent_bits = [i for i in range(N) if not (mask >> i) & 1]
            
            if not present_bits or not absent_bits:
                continue
                
            x = random.choice(present_bits)
            y = random.choice(absent_bits)
            
            mask_without_x = mask ^ (1 << x)
            
            # Check validity of adding y to mask_without_x
            if has_4ap_with_mask(mask_without_x, y):
                continue
                
            new_mask = mask_without_x | (1 << y)
            
            # Calculate delta
            loss = delta_3ap_remove(mask, x)
            gain = delta_3ap_add(mask_without_x, y)
            delta = gain - loss
            
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                mask = new_mask
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
                    last_improve_time = time.time()
                    
        # Periodic local search polish if stuck?
        if time.time() - last_improve_time > 5.0:
             # Do a quick local search
             mask, current_score = local_search(mask, current_score, max_iter=100, timeout=1.0, start_time=time.time())
             if current_score > best_score:
                 best_score = current_score
                 best_mask = mask
                 last_improve_time = time.time()

    # Convert best_mask back to sorted list of integers (1-indexed)
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    return result

# Run once to verify it doesn't crash immediately
if __name__ == "__main__":
    # Just to ensure precomputation and basic logic works in isolation
    # construct() is the entry point
    pass
