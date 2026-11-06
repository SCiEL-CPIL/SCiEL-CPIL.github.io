import random
from fractions import Fraction

def _get_sums(S):
    sums = {Fraction(0, 1)}
    for x in S:
        new_sums = set()
        for s in sums:
            new_sums.add(s + Fraction(1, x))
        if sums & new_sums:
            return None # Invalid
        sums |= new_sums
    return sums

def construct():
    seed = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_set = list(seed)
    best_size = len(best_set)
    
    all_elements = set(range(1, 23))
    
    # Heuristic: Larger numbers are "safer"
    # Sort candidates for addition: try larger numbers first?
    # Or random?
    
    for restart in range(50):
        # Initialize current set
        if restart == 0:
            current_set = list(best_set)
        else:
            # Random restart: pick a random valid set of size ~best_size/2
            # Or just a random small set
            size = max(1, best_size - random.randint(2, 5))
            candidates = random.sample(range(1, 23), size)
            if _get_sums(candidates) is not None:
                current_set = candidates
            else:
                current_set = list(best_set) # Fallback
        
        current_sums = _get_sums(current_set)
        if current_sums is None:
            continue
            
        improved = True
        while improved and len(current_set) < 22:
            improved = False
            available = list(all_elements - set(current_set))
            random.shuffle(available)
            
            # Try to add
            for x in available:
                # Check if adding x is valid
                # New sums = current_sums U {s + 1/x for s in current_sums}
                new_sums = set()
                for s in current_sums:
                    new_sums.add(s + Fraction(1, x))
                
                if current_sums & new_sums:
                    continue
                    
                # Valid addition
                current_set.append(x)
                current_sums = current_sums | new_sums
                improved = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                break # Restart inner loop to try more additions
            
            # If no addition, try swaps to enable future additions
            if not improved:
                in_set = list(current_set)
                out_set = list(all_elements - set(current_set))
                random.shuffle(in_set)
                random.shuffle(out_set)
                
                swap_made = False
                for i in in_set:
                    for o in out_set:
                        if i == o: continue
                        # Try swap
                        new_set = [y for y in current_set if y != i] + [o]
                        new_sums_candidate = _get_sums(new_set)
                        if new_sums_candidate is not None:
                            current_set = new_set
                            current_sums = new_sums_candidate
                            improved = True
                            swap_made = True
                            # Note: Swap does not increase size, 
                            # but changes the set. 
                            # We set improved=True to continue the while loop 
                            # and see if we can add something now.
                            break
                    if swap_made:
                        break
                # If no swap worked, we are stuck.
                # Break the while loop.
                
    return best_set
