"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct a large subset of [1, 512] with no 4-term AP.
    
    Strategy:
    1. Bitmask representation with precomputed conflict masks.
    2. Multiple greedy constructions with different orderings.
    3. Simulated Annealing with add, remove, and swap moves.
    4. Stagnation detection with adaptive restarts.
    5. Final polish pass.
    """
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 512
    # Bit i corresponds to element i+1
    
    # Precompute conflicts
    conflicts = [[] for _ in range(N)]
    
    for a in range(1, N - 2):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            conflicts[a - 1].append(mask_a)
            
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            conflicts[b - 1].append(mask_b)
            
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1))
            conflicts[c - 1].append(mask_c)
            
            mask_e = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflicts[e - 1].append(mask_e)
    
    def is_valid_addition(x, S):
        idx = x - 1
        for m in conflicts[idx]:
            if (S & m) == m:
                return False
        return True
    
    def greedy_construct_bitmask(order=None):
        if order is None:
            order = list(range(1, N + 1))
            random.shuffle(order)
        S = 0
        for x in order:
            if is_valid_addition(x, S):
                S |= (1 << (x - 1))
        return S
    
    def polish(S):
        """Try to add any remaining elements."""
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if not (S & (1 << (x - 1))):
                    if is_valid_addition(x, S):
                        S |= (1 << (x - 1))
                        changed = True
        return S
    
    # Phase 1: Multiple greedy restarts with different strategies
    best_S = 0
    best_size = 0
    
    # Strategy 1: Random order
    for _ in range(30):
        if time.time() > deadline - 30:
            break
        S = greedy_construct_bitmask()
        size = bin(S).count('1')
        if size > best_size:
            best_size = size
            best_S = S
    
    # Strategy 2: Weighted random (prefer elements with fewer conflicts)
    conflict_counts = [len(conflicts[i]) for i in range(N)]
    weights = [1.0 / (1 + conflict_counts[i]) for i in range(N)]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    for _ in range(30):
        if time.time() > deadline - 30:
            break
        order = list(range(1, N + 1))
        # Weighted shuffle
        for i in range(N - 1, 0, -1):
            j = random.choices(range(i + 1), weights=weights[:i + 1])[0]
            order[i], order[j] = order[j], order[i]
        S = greedy_construct_bitmask(order)
        size = bin(S).count('1')
        if size > best_size:
            best_size = size
            best_S = S
    
    # Phase 2: Simulated Annealing with multiple move types
    current_S = best_S
    current_size = best_size
    
    T = 2.0
    cooling = 0.99995
    min_T = 0.05
    
    # Track stagnation
    stagnant_steps = 0
    stagnation_threshold = 5000
    best_seen_in_sa = best_size
    sa_restarts = 0
    max_sa_restarts = 3
    
    while time.time() < deadline - 5 and T > min_T:
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add
            added = False
            for _ in range(15):
                x = random.randint(1, N)
                if not (current_S & (1 << (x - 1))):
                    if is_valid_addition(x, current_S):
                        current_S |= (1 << (x - 1))
                        current_size += 1
                        added = True
                        stagnant_steps = 0
                        break
        elif move_type < 0.7:
            # Try to remove (to enable future additions)
            removed = False
            for _ in range(15):
                x = random.randint(1, N)
                if current_S & (1 << (x - 1)):
                    if random.random() < np.exp(-1.0 / T):
                        current_S &= ~(1 << (x - 1))
                        current_size -= 1
                        removed = True
                        break
        else:
            # Swap: remove one, try to add one
            # Pick element to remove
            removed_x = None
            for _ in range(15):
                x = random.randint(1, N)
                if current_S & (1 << (x - 1)):
                    removed_x = x
                    break
            
            if removed_x is not None:
                # Temporarily remove
                temp_S = current_S & ~(1 << (removed_x - 1))
                temp_size = current_size - 1
                
                # Try to add a different element
                added_x = None
                for _ in range(15):
                    x = random.randint(1, N)
                    if x != removed_x and not (temp_S & (1 << (x - 1))):
                        if is_valid_addition(x, temp_S):
                            added_x = x
                            break
                
                if added_x is not None:
                    # Accept swap with some probability
                    # Size stays same, so accept with prob exp(0/T) = 1
                    # But we want to be selective to avoid random walks
                    if random.random() < 0.3:
                        current_S = temp_S | (1 << (added_x - 1))
                        # current_size unchanged
                        stagnant_steps = 0
                # If no valid addition, keep original (no change)
        
        # Cool down
        T *= cooling
        
        # Update best
        if current_size > best_size:
            best_size = current_size
            best_S = current_S
            best_seen_in_sa = best_size
        
        # Stagnation detection
        if current_size <= best_seen_in_sa:
            stagnant_steps += 1
        else:
            stagnant_steps = 0
        
        # If stuck, restart SA with a new initial point
        if stagnant_steps > stagnation_threshold and sa_restarts < max_sa_restarts:
            sa_restarts += 1
            # Create a new initial point by perturbing best_S
            new_S = best_S
            # Remove some random elements
            num_remove = random.randint(5, 15)
            for _ in range(num_remove):
                x = random.randint(1, N)
                if new_S & (1 << (x - 1)):
                    new_S &= ~(1 << (x - 1))
            
            # Polish
            new_S = polish(new_S)
            new_size = bin(new_S).count('1')
            
            if new_size >= best_size - 5:  # Accept if not too much worse
                current_S = new_S
                current_size = new_size
                T = 1.5  # Reset temperature
                stagnant_steps = 0
    
    # Phase 3: Final polish
    best_S = polish(best_S)
    best_size = bin(best_S).count('1')
    
    # Convert bitmask to list
    result = []
    for i in range(N):
        if best_S & (1 << i):
            result.append(i + 1)
    
    # Verification
    def has_4ap(s):
        s_set = set(s)
        for a in s_set:
            for b in s_set:
                if a >= b:
                    continue
                d = b - a
                if d == 0:
                    continue
                c = b + d
                e = c + d
                if c in s_set and e in s_set:
                    return True
        return False
    
    if has_4ap(result):
        return [1]
    
    return sorted(result)

# EVOLVE-BLOCK-END
