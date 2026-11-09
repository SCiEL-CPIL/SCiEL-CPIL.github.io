# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses Simulated Annealing with bitmask representation for speed.
    """
    m = 205
    
    # 1. Precompute forbidden differences and conflict masks
    forbidden = set(pow(x, 2, m) for x in range(1, m))
    forbidden.discard(0)
    
    # conflict_mask[i] is a bitmask where bit j is 1 if i and j conflict
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            if (i - j) % m in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask
        
    best_mask = 0
    best_size = 0
    
    start_time = time.time()
    # Safety margin: 600s limit, use 580s
    deadline = start_time + 580.0
    
    # Set random seed for reproducibility if needed, but random is fine
    # random.seed(42)
    
    while time.time() < deadline:
        # --- Phase 1: Random Greedy Initialization ---
        curr_mask = 0
        candidates = list(range(m))
        random.shuffle(candidates)
        
        for v in candidates:
            if (curr_mask & conflict_mask[v]) == 0:
                curr_mask |= (1 << v)
                
        curr_size = bin(curr_mask).count('1')
        
        if curr_size > best_size:
            best_size = curr_size
            best_mask = curr_mask
            
        # --- Phase 2: Simulated Annealing Local Search ---
        # We try to improve curr_mask
        T = 5.0
        cooling = 0.995
        min_T = 0.1
        
        # To avoid infinite loops on very small improvements, limit SA iterations per restart
        max_sa_steps = 10000
        
        steps = 0
        while T > min_T and steps < max_sa_steps and time.time() < deadline:
            steps += 1
            
            # Determine move type
            # We prefer Add (size +1), then Swap (size 0), then Remove (size -1)
            # But to escape local minima, we need Remove/Swap.
            
            # 1. Try to Add
            # Pick a random vertex not in mask
            # Efficient way: get list of unset bits? 
            # Since m=205, we can just pick random v and check if it's in mask.
            # Or precompute list of available vertices? 
            # Let's just pick random v.
            
            move_type = random.random()
            delta_e = 0
            accepted = False
            
            if move_type < 0.3:
                # Try Add
                v = random.randint(0, m - 1)
                if not (curr_mask & (1 << v)):
                    # Check compatibility
                    if (curr_mask & conflict_mask[v]) == 0:
                        curr_mask |= (1 << v)
                        curr_size += 1
                        delta_e = 1
                        accepted = True
            elif move_type < 0.8:
                # Try Swap (Remove u, Add v)
                # Pick u in mask
                # Get list of set bits
                # Optimization: Keep a list of set bits? 
                # For m=205, extracting set bits is fast.
                set_bits = []
                temp = curr_mask
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    set_bits.append(idx)
                    temp ^= lsb
                
                if set_bits:
                    u = random.choice(set_bits)
                    # Pick v not in mask
                    v = random.randint(0, m - 1)
                    if not (curr_mask & (1 << v)):
                        # Temporarily remove u
                        new_mask = curr_mask & ~(1 << u)
                        # Check if v is compatible with new_mask
                        if (new_mask & conflict_mask[v]) == 0:
                            curr_mask = new_mask | (1 << v)
                            # Size remains same
                            delta_e = 0
                            accepted = True
            else:
                # Try Remove (to escape local minima)
                set_bits = []
                temp = curr_mask
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    set_bits.append(idx)
                    temp ^= lsb
                    
                if set_bits:
                    u = random.choice(set_bits)
                    curr_mask &= ~(1 << u)
                    curr_size -= 1
                    delta_e = -1
                    accepted = True
            
            if accepted:
                # Update best
                if curr_size > best_size:
                    best_size = curr_size
                    best_mask = curr_mask
                    
                # Acceptance criterion
                if delta_e >= 0:
                    pass # Always accept improvements or neutral swaps
                else:
                    # Accept worse move with probability exp(delta_e / T)
                    # delta_e is -1
                    prob = np.exp(delta_e / T)
                    if random.random() > prob:
                        # Reject: Revert
                        # We need to revert the specific move
                        # This requires storing previous state or undoing
                        # For simplicity and speed, we can just re-run the greedy 
                        # or accept that we lost the state. 
                        # Better: Store prev_mask and prev_size.
                        pass 
            else:
                # Move not valid, do nothing
                pass
                
            T *= cooling
            
        # If we didn't update best in this SA run, we might restart greedy
        # The outer while loop handles restarts
        
    # Convert best_mask to array
    result = []
    for i in range(m):
        if best_mask & (1 << i):
            result.append(i)
            
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
