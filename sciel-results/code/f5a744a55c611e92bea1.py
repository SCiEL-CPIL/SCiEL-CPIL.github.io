"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _precompute_conflicts(N=180):
    """Precompute pair masks for each element y.
    pair_masks[y] is a list of integers, each representing a pair {a, b} such that {y, a, b} is a forbidden triple.
    """
    pair_masks = [[] for _ in range(N + 1)]
    
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a < b and a != c:
                    # Triple is (a, b, c)
                    # For a: pair is (b, c)
                    # For b: pair is (a, c)
                    # For c: pair is (a, b)
                    
                    mask_bc = (1 << b) | (1 << c)
                    mask_ac = (1 << a) | (1 << c)
                    mask_ab = (1 << a) | (1 << b)
                    
                    pair_masks[a].append(mask_bc)
                    pair_masks[b].append(mask_ac)
                    pair_masks[c].append(mask_ab)
    
    return pair_masks

def _can_add(y, S_mask, pair_masks):
    """Check if y can be added to S_mask without creating a forbidden triple."""
    for pm in pair_masks[y]:
        if (pm & S_mask) == pm:
            return False
    return True

def _greedy_fill(S_mask, pair_masks, deadline):
    """Greedily add elements to S_mask if possible."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for y in range(1, 181):
            if S_mask & (1 << y):
                continue
            if _can_add(y, S_mask, pair_masks):
                S_mask |= (1 << y)
                changed = True
    return S_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 180
    pair_masks = _precompute_conflicts(N)
    
    # Initial solution: Start with 91..180 as a base, then greedy fill
    current_mask = 0
    for i in range(91, 181):
        current_mask |= (1 << i)
        
    current_mask = _greedy_fill(current_mask, pair_masks, deadline)
    
    best_size = bin(current_mask).count("1")
    best_mask = current_mask
    
    # Local Search: Try removing 1 or 2 elements and refilling
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Try single removals
        for x in range(1, N + 1):
            if not (current_mask & (1 << x)):
                continue
            
            trial_mask = current_mask & ~(1 << x)
            trial_mask = _greedy_fill(trial_mask, pair_masks, deadline)
            
            if bin(trial_mask).count("1") > best_size:
                best_size = bin(trial_mask).count("1")
                best_mask = trial_mask
                current_mask = trial_mask
                improved = True
                break # Restart search with new best
        
        if time.time() > deadline - 2:
            break
            
        # Try pair removals (if time permits)
        # Sample random pairs to escape local minima
        if not improved:
            for _ in range(50): # Limit attempts per iteration
                if time.time() > deadline - 2:
                    break
                # Get list of set bits
                bits = [i for i in range(1, N+1) if current_mask & (1 << i)]
                if len(bits) < 2:
                    break
                i, j = random.sample(bits, 2)
                trial_mask = current_mask & ~(1 << i) & ~(1 << j)
                trial_mask = _greedy_fill(trial_mask, pair_masks, deadline)
                
                if bin(trial_mask).count("1") > best_size:
                    best_size = bin(trial_mask).count("1")
                    best_mask = trial_mask
                    current_mask = trial_mask
                    improved = True
                    break
                     
    # Convert best_mask to list
    result = [i for i in range(1, N + 1) if best_mask & (1 << i)]
    
    # Ensure non-empty
    if not result:
        return [180]
        
    return sorted(result)
# EVOLVE-BLOCK-END
