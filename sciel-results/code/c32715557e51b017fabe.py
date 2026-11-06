import time
from math import gcd

_START = time.time()
_DEADLINE = 130.0  # Safety margin under 150s kill

def construct():
    """
    Constructs a set of pairwise disjoint residue classes [m, r] with distinct moduli m in [20, 200]
    to maximize the sum of 1/m.
    """
    # List to store chosen [m, r] pairs
    chosen = []
    
    # Helper to check if a pair (m, r) is disjoint from all in chosen
    def is_valid(m, r, chosen_list):
        for mk, rk in chosen_list:
            g = gcd(m, mk)
            if (r % g) == (rk % g):
                return False
        return True

    # Helper to find all valid residues for a given m
    def find_valid_residues(m, chosen_list):
        valid = []
        for r in range(m):
            if is_valid(m, r, chosen_list):
                valid.append(r)
        return valid

    # Iterate through moduli in ascending order to prioritize small m (high 1/m)
    for m in range(20, 201):
        # Check time budget
        if time.time() - _START > _DEADLINE:
            break
        
        valid_residues = find_valid_residues(m, chosen)
        
        if not valid_residues:
            continue
            
        if len(valid_residues) == 1:
            chosen.append([m, valid_residues[0]])
        else:
            # Heuristic: Pick the residue r that maximizes the number of valid residues for the next modulus (m+1)
            # This lookahead helps avoid dead-ends where a choice now blocks many future small moduli.
            best_r = valid_residues[0]
            max_future_options = -1
            
            next_m = m + 1
            # We only need to lookahead if next_m is within range
            if next_m <= 200:
                for r in valid_residues:
                    # Tentatively add (m, r)
                    chosen.append([m, r])
                    
                    # Count valid residues for next_m
                    future_valid = find_valid_residues(next_m, chosen)
                    count = len(future_valid)
                    
                    # Remove tentative addition
                    chosen.pop()
                    
                    if count > max_future_options:
                        max_future_options = count
                        best_r = r
                    # Tie-breaker: smallest residue
                    elif count == max_future_options and r < best_r:
                        best_r = r
            else:
                # If it's the last modulus, just pick the smallest valid one
                best_r = min(valid_residues)
                
            chosen.append([m, best_r])
            
    return chosen
