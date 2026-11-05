import time
import random

# EVOLVE-BLOCK-START
def construct():
    n = 512
    deadline = time.time() + 85.0  # Safety margin
    
    # Precompute conflict masks for each x in 1..n
    # masks[x] is a list of bitmasks. Each bitmask represents the other 3 elements
    # of a 4-AP that includes x.
    # If we add x to set S, the number of new 4-APs is the number of masks M
    # such that S contains all bits in M.
    
    masks = [[] for _ in range(n + 1)]
    
    for x in range(1, n + 1):
        # Type 1: x, x+d, x+2d, x+3d
        # x is the first term.
        # Other terms: x+d, x+2d, x+3d
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            a = x + d
            b = x + 2 * d
            c = x + 3 * d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            masks[x].append(mask)
            
        # Type 2: x-d, x, x+d, x+2d
        # x is the second term.
        # Other terms: x-d, x+d, x+2d
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            a = x - d
            b = x + d
            c = x + 2 * d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            masks[x].append(mask)
            
        # Type 3: x-2d, x-d, x, x+d
        # x is the third term.
        # Other terms: x-2d, x-d, x+d
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            a = x - 2 * d
            b = x - d
            c = x + d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            masks[x].append(mask)
            
        # Type 4: x-3d, x-2d, x-d, x
        # x is the fourth term.
        # Other terms: x-3d, x-2d, x-d
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            a = x - 3 * d
            b = x - 2 * d
            c = x - d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            masks[x].append(mask)
            
    def count_conflicts(x, S):
        """Count how many 4-APs would be formed if x were in S (given current S)."""
        count = 0
        for m in masks[x]:
            if (S & m) == m:
                count += 1
        return count
        
    # Initialize with a greedy set to have a decent starting point
    S = 0
    aps = 0
    size = 0
    for x in range(1, n + 1):
        # Check if adding x creates any AP
        # If S is built incrementally, we only need to check conflicts with existing elements.
        # But count_conflicts(x, S) checks if the OTHER 3 are in S.
        # So if count_conflicts(x, S) == 0, we can add x.
        if count_conflicts(x, S) == 0:
            S |= (1 << (x - 1))
            size += 1
            # Update aps?
            # Since we only add if conflicts == 0, aps remains 0.
            
    best_S = S
    best_size = size
    best_aps = 0
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    penalty = 100.0  # High penalty to avoid APs
    
    while time.time() < deadline:
        # Random move
        x = random.randint(1, n)
        bit = 1 << (x - 1)
        
        if S & bit:
            # Removing x
            # Decrease aps by count_conflicts(x, S_without_x)?
            # No, count_conflicts(x, S) counts APs containing x and 3 others in S.
            # If x is in S, these are the APs that x participates in.
            # So removing x removes exactly count_conflicts(x, S) APs.
            delta_aps = -count_conflicts(x, S)
            delta_size = -1
            new_S = S & ~bit
        else:
            # Adding x
            delta_aps = count_conflicts(x, S)
            delta_size = 1
            new_S = S | bit
            
        new_aps = aps + delta_aps
        new_size = size + delta_size
        
        # Score: maximize size, minimize aps
        # We want to reach aps=0.
        # If new_aps == 0, score = new_size
        # If new_aps > 0, score = new_size - penalty * new_aps
        
        old_score = size - penalty * aps
        new_score = new_size - penalty * new_aps
        delta_score = new_score - old_score
        
        # Acceptance probability
        if delta_score > 0:
            accept = True
        else:
            if T > T_min:
                # Avoid overflow in exp
                if delta_score > -500 * T: 
                    prob = pow(2.718281828, delta_score / T)
                    accept = random.random() < prob
                else:
                    accept = False
            else:
                accept = False
                
        if accept:
            S = new_S
            aps = new_aps
            size = new_size
            
            # Update best if valid (aps == 0)
            if aps == 0 and size > best_size:
                best_size = size
                best_S = S
                best_aps = 0
                
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            # Optional: restart with higher T if stuck?
            # For now, just keep cooling or reset if no improvement for a while.
            # Simple approach: just keep running.
            
    # Return the best valid set found
    # If best_aps > 0, we might need to clean it up?
    # But we only update best_S if aps == 0.
    # If no valid set found (unlikely, greedy gives one), return greedy.
    
    result = []
    for i in range(n):
        if best_S & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
# EVOLVE-BLOCK-END
