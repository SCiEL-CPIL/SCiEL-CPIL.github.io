import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    base = 51975
    # Generate odd divisors > 1
    moduli = [d for d in range(3, base + 1, 2) if base % d == 0]
    # Sort moduli, maybe descending helps? Or random? 
    # Let's try descending for initial greedy, as large moduli cover fewer specific spots but are "heavier" in constraint satisfaction?
    # Actually, for covering, small moduli cover more residues. 
    # Let's stick to a heuristic order or just use the list.
    
    # Initial assignment: Greedy
    # Start with no coverage
    count_cover = np.zeros(base, dtype=np.int8)
    residues = {}
    
    # Greedy initialization
    # Process moduli in increasing order? 
    # If we pick small m first, they cover many. 
    # Let's try increasing order for init.
    for m in sorted(moduli):
        # Find r that maximizes coverage of currently uncovered (count_cover==0)
        # Score for r is sum(1 for x in r::m if count_cover[x]==0)
        best_r = 0
        best_score = -1
        # Optimization: if m is large, iterating m times is fine. m <= 51975.
        # But we have ~100 moduli. 100 * 50000 = 5M ops. Fine.
        for r in range(m):
            # Sum of count_cover[r::m] == 0
            # Using numpy slicing
            s = np.sum(count_cover[r::m] == 0)
            if s > best_score:
                best_score = s
                best_r = r
        residues[m] = best_r
        count_cover[best_r::m] += 1
        
    # Local Search
    # Repeat until no improvement or time limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate over moduli in random order or fixed?
        # Fixed order might get stuck in local minima. Random might help.
        # Let's use a fixed order but maybe shuffle occasionally?
        # For now, fixed order.
        for m in moduli:
            if time.time() > deadline:
                break
            
            r_old = residues[m]
            
            # Calculate K(x) = count_cover[x] - (1 if x in C_m else 0)
            # We want to find r' maximizing sum(1 for x in r'::m if K(x)==0)
            
            # Create mask for current class m
            # Instead of creating full mask, we can compute scores directly?
            # Z[x] = 1 if count_cover[x] - (1 if x % m == r_old else 0) == 0
            # This depends on r_old.
            
            # Efficient way:
            # 1. Subtract 1 from count_cover for current class to get count_others
            count_others = count_cover.copy()
            count_others[r_old::m] -= 1
            
            # Now we want to find r' maximizing sum(1 for x in r'::m if count_others[x]==0)
            # This is the same as the greedy step but on count_others.
            
            best_r = -1
            best_score = -1
            for r in range(m):
                s = np.sum(count_others[r::m] == 0)
                if s > best_score:
                    best_score = s
                    best_r = r
            
            # Compare best_score with current score
            # Current score is sum(1 for x in r_old::m if count_others[x]==0)
            # Which is exactly best_score if best_r == r_old?
            # No, best_score is the max over all r.
            # If best_r == r_old, no change.
            # If best_score > score(r_old), we improve.
            # score(r_old) is np.sum(count_others[r_old::m] == 0)
            
            current_score = np.sum(count_others[r_old::m] == 0)
            
            if best_score > current_score:
                # Update
                # Restore count_cover by adding back old class? 
                # Easier: 
                # count_cover currently has old class included.
                # We want to replace old with new.
                # count_cover[r_old::m] -= 1
                # count_cover[best_r::m] += 1
                
                count_cover[r_old::m] -= 1
                count_cover[best_r::m] += 1
                residues[m] = best_r
                improved = True
                # Optional: restart the loop over moduli if significant improvement?
                # For now, continue to next modulus.
        
        # If we made improvements, we might want to do another pass.
        # The while loop handles this.
        
    # Construct return value
    classes = [[m, residues[m]] for m in moduli]
    return classes
