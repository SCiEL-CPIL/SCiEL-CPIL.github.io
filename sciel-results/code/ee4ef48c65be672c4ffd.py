import time
import random
import numpy as np

N = 256
TIME_LIMIT = 55.0  # safety margin

def _precompute():
    # All 3-APs as bitmasks (bit i = number i+1)
    m3 = []
    for d in range(1, N):
        for a in range(1, N - 2*d + 1):
            mask = (1 << (a-1)) | (1 << (a+d-1)) | (1 << (a+2*d-1))
            m3.append(mask)
    # All 4-APs as bitmasks
    m4 = []
    for d in range(1, N):
        for a in range(1, N - 3*d + 1):
            mask = (1 << (a-1)) | (1 << (a+d-1)) | (1 << (a+2*d-1)) | (1 << (a+3*d-1))
            m4.append(mask)
    
    # For each element, which 4-APs contain it (for fast validity check)
    # And for each element, which 3-APs contain it (for fast counting)
    # Actually, let's use a different approach: 
    # Precompute for each 4-AP the tuple of its 4 elements (0-indexed)
    # And for each 3-AP the tuple of its 3 elements (0-indexed)
    
    ap4_elems = []
    for d in range(1, N):
        for a in range(1, N - 3*d + 1):
            ap4_elems.append((a-1, a+d-1, a+2*d-1, a+3*d-1))
    
    ap3_elems = []
    for d in range(1, N):
        for a in range(1, N - 2*d + 1):
            ap3_elems.append((a-1, a+d-1, a+2*d-1))
    
    return m3, m4, ap4_elems, ap3_elems

M3, M4, AP4_ELEMS, AP3_ELEMS = _precompute()

def _count_3aps(mask):
    """Count 3-APs in mask using bit operations on precomputed masks."""
    count = 0
    for m in M3:
        if (mask & m) == m:
            count += 1
    return count

def _is_valid_4ap_free(mask):
    """Check if mask is 4-AP-free."""
    for m in M4:
        if (mask & m) == m:
            return False
    return True

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT
    
    # --- Greedy construction to get a good starting point ---
    # Try multiple random orderings and keep the best
    best_greedy_mask = 0
    best_greedy_score = -1
    max_greedy_tries = 50
    
    for _ in range(max_greedy_tries):
        if time.time() > deadline - 10:
            break
        order = list(range(N))
        random.shuffle(order)
        mask = 0
        for idx in order:
            bit = 1 << idx
            new_mask = mask | bit
            # Check if adding this bit creates a 4-AP
            # We need to check if any 4-AP is now fully contained
            # Optimization: only check 4-APs that contain this bit
            # But for simplicity with 50 tries, let's just do the full check
            # Actually, let's precompute which 4-APs contain each bit
            valid = True
            for m in M4:
                if (new_mask & m) == m:
                    valid = False
                    break
            if valid:
                mask = new_mask
        score = _count_3aps(mask)
        if score > best_greedy_score:
            best_greedy_score = score
            best_greedy_mask = mask
    
    # --- Simulated Annealing ---
    current_mask = best_greedy_mask
    current_score = best_greedy_score
    best_mask = current_mask
    best_score = current_score
    
    # Precompute for each bit, the list of 4-AP masks that contain it
    # This allows us to only check relevant 4-APs when adding/removing a bit
    bit_to_4aps = [[] for _ in range(N)]
    for i, m in enumerate(M4):
        for bit in range(N):
            if m & (1 << bit):
                bit_to_4aps[bit].append(i)
    
    # Also precompute for each bit, the list of 3-AP masks that contain it
    bit_to_3aps = [[] for _ in range(N)]
    for i, m in enumerate(M3):
        for bit in range(N):
            if m & (1 << bit):
                bit_to_3aps[bit].append(i)
    
    # Precompute boolean arrays for faster checking
    # is_4ap_present[j] = whether M4[j] is fully in current_mask
    is_4ap_present = np.zeros(len(M4), dtype=bool)
    # is_3ap_present[j] = whether M3[j] is fully in current_mask
    is_3ap_present = np.zeros(len(M3), dtype=bool)
    
    # Initialize these arrays
    for j, m in enumerate(M4):
        if (current_mask & m) == m:
            is_4ap_present[j] = True
    for j, m in enumerate(M3):
        if (current_mask & m) == m:
            is_3ap_present[j] = True
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random bit to flip
        bit = random.randint(0, N - 1)
        bit_val = 1 << bit
        adding = (current_mask & bit_val) == 0
        
        # Compute delta in score and validity
        if adding:
            # Adding bit: 
            # New 3-APs formed: 3-APs containing bit where the other 2 bits are already in mask
            delta_score = 0
            for j in bit_to_3aps[bit]:
                m = M3[j]
                if not is_3ap_present[j]:
                    # Check if the other two bits are in mask
                    others = m & ~bit_val
                    if (current_mask & others) == others:
                        delta_score += 1
                        is_3ap_present[j] = True
            
            # Check if this creates a 4-AP
            creates_4ap = False
            for j in bit_to_4aps[bit]:
                m = M4[j]
                if not is_4ap_present[j]:
                    others = m & ~bit_val
                    if (current_mask & others) == others:
                        creates_4ap = True
                        is_4ap_present[j] = True
                        break
            
            if creates_4ap:
                # This move is invalid; revert the 3-AP updates and skip
                for j in bit_to_3aps[bit]:
                    m = M3[j]
                    if is_3ap_present[j]:
                        others = m & ~bit_val
                        if (current_mask & others) == others:
                            is_3ap_present[j] = False
                # Revert 4-AP updates
                for j in bit_to_4aps[bit]:
                    m = M4[j]
                    if is_4ap_present[j]:
                        others = m & ~bit_val
                        if (current_mask & others) == others:
                            is_4ap_present[j] = False
                continue
        else:
            # Removing bit:
            # 3-APs lost: 3-APs containing bit that were present
            delta_score = 0
            for j in bit_to_3aps[bit]:
                if is_3ap_present[j]:
                    delta_score -= 1
                    is_3ap_present[j] = False
            
            # Removing a bit cannot create a new 4-AP, so no validity issue
            # But we need to update is_4ap_present for 4-APs that contained this bit
            for j in bit_to_4aps[bit]:
                if is_4ap_present[j]:
                    m = M4[j]
                    others = m & ~bit_val
                    if (current_mask & others) == others:
                        is_4ap_present[j] = False
                    else:
                        is_4ap_present[j] = False
        
        new_score = current_score + delta_score
        
        # Acceptance
        if new_score > current_score or random.random() < np.exp(delta_score / T):
            # Apply the move
            if adding:
                current_mask |= bit_val
            else:
                current_mask &= ~bit_val
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
        else:
            # Revert the state changes
            if adding:
                for j in bit_to_3aps[bit]:
                    m = M3[j]
                    if is_3ap_present[j]:
                        others = m & ~bit_val
                        if (current_mask & others) == others:
                            is_3ap_present[j] = False
                for j in bit_to_4aps[bit]:
                    m = M4[j]
                    if is_4ap_present[j]:
                        others = m & ~bit_val
                        if (current_mask & others) == others:
                            is_4ap_present[j] = False
            else:
                for j in bit_to_3aps[bit]:
                    m = M3[j]
                    if not is_3ap_present[j]:
                        others = m & ~bit_val
                        if ((current_mask | bit_val) & others) == others:
                            is_3ap_present[j] = True
                for j in bit_to_4aps[bit]:
                    m = M4[j]
                    if not is_4ap_present[j]:
                        others = m & ~bit_val
                        if ((current_mask | bit_val) & others) == others:
                            is_4ap_present[j] = True
        
        T *= cooling
    
    # Convert best_mask to list
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
    return sorted(result)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    mask = 0
    for x in res:
        mask |= (1 << (x-1))
    print(f"Fitness: {_count_3aps(mask)}")
    print(f"Valid: {_is_valid_4ap_free(mask)}")
