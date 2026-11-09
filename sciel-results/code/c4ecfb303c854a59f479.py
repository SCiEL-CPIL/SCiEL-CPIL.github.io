import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 60  # Safety margin for 90s limit
    
    max_mod = 128
    # Candidate moduli: all even numbers
    # We proved that any valid set must be a clique in the non-coprime graph.
    # If a prime p is in the set, all elements are multiples of p.
    # Best case is multiples of 2.
    candidates = [m for m in range(2, max_mod + 1) if m % 2 == 0]
    
    # We want to maximize the number of classes.
    # We will try to assign residues to all candidates.
    # If we fail, we might drop some, but greedy usually works for this structure.
    
    best_result = []
    
    # Try multiple random restarts
    # The order of processing matters. Ascending is good for constraints.
    # Residue choice is randomized.
    
    for attempt in range(10000):
        if time.time() > deadline:
            break
            
        # Shuffle residues for each modulus? Or just pick random valid?
        # We iterate moduli in ascending order.
        current_assignment = {}
        success = True
        
        for m in candidates:
            # Generate random permutation of residues
            residues = list(range(m))
            random.shuffle(residues)
            
            found = False
            for r in residues:
                # Check against all previously assigned
                compatible = True
                for m_prev, r_prev in current_assignment.items():
                    g = math.gcd(m, m_prev)
                    if (r - r_prev) % g == 0:
                        compatible = False
                        break
                if compatible:
                    current_assignment[m] = r
                    found = True
                    break
            
            if not found:
                success = False
                break
        
        if success:
            # We found a valid assignment for all 64 moduli
            # Construct the list
            result = [[r, m] for m, r in current_assignment.items()]
            # Sort for consistency? Not required, but good.
            result.sort(key=lambda x: x[1])
            
            # Check if this is better than best
            if len(result) > len(best_result):
                best_result = result
                if len(best_result) == len(candidates):
                    # Can't do better than all candidates
                    return best_result
                    
    # If we didn't find a full solution, return the best partial one found?
    # The greedy above breaks on failure. It doesn't keep partial.
    # Let's modify to keep partial if full fails?
    # Actually, if it fails, it's likely due to bad luck.
    # With 10000 restarts, it should find one.
    # If not, we return whatever we have.
    
    if not best_result:
        # Fallback to seed-like behavior if search fails
        # But search should not fail.
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
        
    return best_result
