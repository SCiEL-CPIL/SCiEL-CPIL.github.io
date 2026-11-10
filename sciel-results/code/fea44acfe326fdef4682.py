"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Denominators are 2 to 48. Indices 0 to 46 correspond to denominators 2 to 48.
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocals as Fractions
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # Known good solution from analysis: indices [1, 2, 3, 4, 35, 42]
    # Indices: 1->3, 2->4, 3->5, 4->6, 35->37, 42->44
    # Sum: 1/3 + 1/4 + 1/5 + 1/6 + 1/37 + 1/44
    # Let's verify this is strictly below 1.
    # 1/3+1/4+1/5+1/6 = 20/60+15/60+12/60+10/60 = 57/60 = 19/20 = 0.95
    # 1/37 + 1/44 = (44+37)/(37*44) = 81/1628 ≈ 0.04975
    # Total ≈ 0.99975 < 1.
    # Deficit = 1 - (19/20 + 81/1628) = 1/20 - 81/1628 = (1628 - 1620) / (20*1628) = 8 / 32560 = 1 / 4070.
    
    initial_indices = [1, 2, 3, 4, 35, 42]
    current_indices = set(initial_indices)
    current_sum = sum(reciprocals[i] for i in current_indices)
    current_deficit = Fraction(1) - current_sum
    
    best_indices = set(current_indices)
    best_deficit = current_deficit
    
    # Local Search: Simulated Annealing or Hill Climbing with swaps
    # Move: Pick one index to remove and one to add (swap), or just add/remove if size constraint allows (but size isn't fixed, just distinct).
    # Actually, the problem doesn't fix the size. But adding elements increases sum, potentially crossing 1. Removing decreases sum.
    # To minimize deficit (get closer to 1 from below), we want to add elements that keep sum < 1, or swap elements to fine-tune.
    
    # Let's try a simple hill climbing with swaps and single adds/removes.
    # Since we want to be strictly below 1, adding an element is only valid if sum + 1/d < 1.
    # Swapping is generally safe if we check the new sum.
    
    temp = 0.01
    cooling = 0.99
    
    while time.time() < deadline:
        # Generate a neighbor
        # Option 1: Swap two elements (one in set, one not)
        # Option 2: Add one element
        # Option 3: Remove one element (unlikely to improve deficit, but might allow better swaps later? No, removing increases deficit directly)
        
        # Let's focus on Swaps and Adds.
        # Sample a candidate move
        move_type = random.random()
        
        if move_type < 0.7:
            # Swap
            if not current_indices:
                continue
            in_idx = random.choice(list(current_indices))
            out_candidates = [i for i in range(n) if i not in current_indices]
            if not out_candidates:
                continue
            out_idx = random.choice(out_candidates)
            
            new_indices = set(current_indices)
            new_indices.remove(in_idx)
            new_indices.add(out_idx)
            
            # Calculate new sum efficiently? 
            # current_sum - recip[in_idx] + recip[out_idx]
            new_sum = current_sum - reciprocals[in_idx] + reciprocals[out_idx]
            
            if new_sum < 1:
                new_deficit = Fraction(1) - new_sum
                # Accept if better or with probability exp(-delta/T)
                delta = new_deficit - current_deficit
                if delta < 0 or random.random() < pow(2.71828, -delta / temp):
                    current_indices = new_indices
                    current_sum = new_sum
                    current_deficit = new_deficit
                    
                    if current_deficit < best_deficit:
                        best_deficit = current_deficit
                        best_indices = set(current_indices)
        else:
            # Add
            out_candidates = [i for i in range(n) if i not in current_indices]
            if not out_candidates:
                continue
            add_idx = random.choice(out_candidates)
            
            new_indices = set(current_indices)
            new_indices.add(add_idx)
            new_sum = current_sum + reciprocals[add_idx]
            
            if new_sum < 1:
                new_deficit = Fraction(1) - new_sum
                delta = new_deficit - current_deficit
                if delta < 0 or random.random() < pow(2.71828, -delta / temp):
                    current_indices = new_indices
                    current_sum = new_sum
                    current_deficit = new_deficit
                    
                    if current_deficit < best_deficit:
                        best_deficit = current_deficit
                        best_indices = set(current_indices)
        
        temp *= cooling
        if temp < 0.0001:
            break
            
    # Return list of indices
    return sorted(list(best_indices))
# EVOLVE-BLOCK-END
