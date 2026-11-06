"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import log

# EVOLVE-BLOCK-START

def _check_insertion(seq, products, x, pos):
    """
    Check if inserting x at position pos in seq is valid.
    Returns True if valid, False otherwise.
    Also checks strictly increasing constraint.
    """
    n = len(seq)
    
    # Check strictly increasing
    if pos > 0 and x <= seq[pos - 1]:
        return False
    if pos < n and x >= seq[pos]:
        return False
        
    # Check product uniqueness
    # New products will be:
    # x
    # x * seq[pos-1]
    # x * seq[pos-1] * seq[pos-2]
    # ...
    # x * seq[pos-1] * ... * seq[0]
    
    # We need to check if any of these new products are already in `products`.
    # Note: The existing products set contains all products of the current seq.
    # The new products are disjoint from existing products in terms of indices, 
    # but their VALUES must not collide.
    
    # Calculate new products
    prod = x
    if prod in products:
        return False
    
    for i in range(pos - 1, -1, -1):
        prod *= seq[i]
        if prod in products:
            return False
            
    return True

def _do_insertion(seq, products, x, pos):
    """
    Insert x at position pos into seq and update products.
    """
    seq.insert(pos, x)
    
    prod = x
    products.add(prod)
    for i in range(pos - 1, -1, -1):
        prod *= seq[i]
        products.add(prod)

def _greedy_max_insertion(seq, products, available):
    """
    Repeatedly try to insert any available number into any valid position.
    Prioritizes smaller numbers? Or just first found?
    Let's try first found in ascending order of x, then ascending order of pos.
    """
    improved = True
    while improved:
        improved = False
        # Iterate through available numbers in ascending order
        for x in sorted(available):
            # Iterate through possible positions
            # To maintain strictly increasing, x can only be placed in specific slots.
            # But checking all pos is O(N).
            # We can find the range of valid pos for x.
            # However, product constraint is global.
            # Let's just check all positions from 0 to len(seq).
            for pos in range(len(seq) + 1):
                if _check_insertion(seq, products, x, pos):
                    _do_insertion(seq, products, x, pos)
                    available.remove(x)
                    improved = True
                    break # Try next x
            # If x couldn't be inserted, move to next x
    return seq, products, available

def _local_search_swap(seq, products, available, deadline):
    """
    Try to swap an element in seq with an element in available to maintain length 
    but potentially allow further insertions later.
    Or simply try to remove an element and see if we can insert two others.
    This is complex. Let's stick to simple swaps first.
    """
    best_len = len(seq)
    # We don't really want to swap if it doesn't lead to improvement.
    # A better local search: Remove a random element, then run greedy_max_insertion again.
    # If the resulting length is > best_len, keep it.
    
    # For now, let's just rely on multiple restarts with different initial seeds.
    pass

def _run_single_trial(seed_val, deadline):
    """
    Run a single search trial.
    1. Initialize with a small random sequence.
    2. Run greedy_max_insertion.
    3. Return the resulting sequence.
    """
    random.seed(seed_val)
    
    # Initial sequence: pick 3-5 random distinct numbers from [2, 128]
    # Sort them to ensure strictly increasing
    init_size = random.randint(3, 5)
    init_vals = random.sample(range(2, 129), init_size)
    init_vals.sort()
    
    seq = list(init_vals)
    products = set()
    
    # Calculate initial products
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            products.add(p)
            
    available = set(range(2, 129))
    for x in seq:
        available.remove(x)
        
    # Run greedy insertion
    seq, products, available = _greedy_max_insertion(seq, products, available)
    
    return seq

def construct():
    start = time.time()
    deadline = start + 80.0  # Safety margin
    
    best_seq = []
    best_len = 0
    
    # Run multiple trials with different random seeds
    # We want to explore different starting points
    num_trials = 100
    
    for i in range(num_trials):
        if time.time() > deadline - 10.0:
            break
            
        trial_seq = _run_single_trial(i, deadline)
        
        if len(trial_seq) > best_len:
            best_len = len(trial_seq)
            best_seq = trial_seq
            
            # If we found a very long sequence, we might break early?
            # Max possible is 127 (all numbers). But likely much shorter.
            # Let's see how long we can get.
            if best_len > 20:
                # Just a heuristic stop if we get very long
                pass
                
    # Final polish: Try to insert any remaining available numbers into best_seq
    # This handles cases where the greedy order missed a spot that became valid 
    # due to specific structure, though _greedy_max_insertion is exhaustive for 
    # single insertions.
    
    # Re-calculate products for best_seq to ensure validity
    # (In case best_seq was from an old trial and we want to try more insertions? 
    # No, _greedy_max_insertion runs until no more insertions are possible.
    # So best_seq is locally optimal for single insertions.)
    
    # However, we might have missed a sequence that required a "swap" to unlock.
    # Let's try a simple improvement: 
    # For each element in best_seq, try removing it, and then run _greedy_max_insertion 
    # on the remaining elements + the removed element (which is now available).
    # If the new length is greater, update best_seq.
    
    # This is expensive, so limit it.
    if time.time() < deadline - 5.0:
        current_best = list(best_seq)
        current_len = len(current_best)
        
        # Calculate products for current_best
        products = set()
        n = len(current_best)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= current_best[j]
                products.add(p)
                
        available = set(range(2, 129))
        for x in current_best:
            available.remove(x)
            
        # Try removing each element and re-inserting
        # We only do this if we have time
        for i in range(len(current_best)):
            if time.time() > deadline - 2.0:
                break
                
            x_to_remove = current_best[i]
            
            # Remove x_to_remove
            trial_seq = current_best[:i] + current_best[i+1:]
            
            # Recalculate products for trial_seq
            trial_products = set()
            tn = len(trial_seq)
            for k in range(tn):
                p = 1
                for l in range(k, tn):
                    p *= trial_seq[l]
                    trial_products.add(p)
                    
            trial_available = set(available)
            trial_available.add(x_to_remove)
            
            # Run greedy insertion on this new state
            trial_seq, trial_products, trial_available = _greedy_max_insertion(trial_seq, trial_products, trial_available)
            
            if len(trial_seq) > current_len:
                current_len = len(trial_seq)
                current_best = trial_seq
                # Update best
                if current_len > best_len:
                    best_len = current_len
                    best_seq = current_best
    
    # Final validation
    if not best_seq:
        return [2, 3, 4, 5] # Fallback
        
    # Verify strictly increasing
    for i in range(len(best_seq) - 1):
        if best_seq[i] >= best_seq[i+1]:
            return [2, 3, 4, 5]
            
    # Verify products
    prods = set()
    n = len(best_seq)
    count = 0
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= best_seq[j]
            if p in prods:
                return [2, 3, 4, 5]
            prods.add(p)
            count += 1
            
    if count != n * (n + 1) // 2:
        return [2, 3, 4, 5]
        
    return best_seq

# EVOLVE-BLOCK-END
