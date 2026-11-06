import numpy as np
import time
import random

def construct():
    N = 51975
    divisors = [d for d in range(2, N + 1) if N % d == 0]
    
    residues = {d: 0 for d in divisors}
    
    # Precompute x % d for all d
    # This is a dictionary of lists
    x_mods = {}
    for d in divisors:
        x_mods[d] = np.arange(N) % d
        
    count = np.zeros(N, dtype=np.int8)
    
    for d in divisors:
        r = residues[d]
        # count[x] += 1 for x in range(r, N, d)
        count[r::d] += 1
        
    start_time = time.time()
    timeout = 110.0
    
    # Indices for uncovered
    # We will recompute this each iteration or maintain it?
    # Recomputing is O(N) in C, which is fast.
    
    while time.time() - start_time < timeout:
        improved = False
        shuffled_divs = divisors[:]
        random.shuffle(shuffled_divs)
        
        for d in shuffled_divs:
            if time.time() - start_time > timeout:
                break
                
            r_old = residues[d]
            
            # Get indices of uncovered residues
            uncovered_idx = np.nonzero(count == 0)[0]
            
            if len(uncovered_idx) == 0:
                return [[d, residues[d]] for d in divisors] # Perfect cover found!
                
            # Compute gain for each r
            # We want count of uncovered x such that x % d == r
            # x_mods[d] is array of x%d for all x
            # We need x_mods[d][uncovered_idx]
            mods = x_mods[d][uncovered_idx]
            gain = np.bincount(mods, minlength=d)
            
            # Compute loss for r_old
            # Count of x in range(r_old, N, d) such that count[x] == 1
            # x_old_indices = np.arange(r_old, N, d)
            # loss_old = np.sum(count[x_old_indices] == 1)
            # This creates a slice.
            x_old = np.arange(r_old, N, d)
            loss_old = np.sum(count[x_old] == 1)
            
            # Find best r
            best_r = np.argmax(gain)
            
            if best_r != r_old and gain[best_r] > loss_old:
                # Update count
                # Subtract old
                count[r_old::d] -= 1
                # Add new
                count[best_r::d] += 1
                
                residues[d] = best_r
                improved = True
        
        if time.time() - start_time > timeout:
            break
            
        if not improved:
            # Stuck. Try random restarts?
            # Or just break.
            # Let's try a few random perturbations
            for _ in range(10):
                if time.time() - start_time > timeout:
                    break
                d = random.choice(divisors)
                r_old = residues[d]
                r_new = random.randint(0, d-1)
                if r_new == r_old: continue
                
                # Evaluate change
                uncovered_idx = np.nonzero(count == 0)[0]
                if len(uncovered_idx) == 0: break
                
                mods = x_mods[d][uncovered_idx]
                gain = np.bincount(mods, minlength=d)
                
                x_old = np.arange(r_old, N, d)
                loss_old = np.sum(count[x_old] == 1)
                
                # For random move, we just check if it's better
                # delta = gain[r_new] - (loss_old if r_new != r_old else 0) 
                # Wait, the logic for delta is specific to the best response.
                # For a random move, we just calculate the new uncovered count.
                # New Uncovered = Old Uncovered - (Newly Covered) + (Newly Uncovered)
                # This is expensive to calculate exactly without updating.
                # Let's just update and check if score improved.
                
                old_score = np.sum(count == 0)
                
                count[r_old::d] -= 1
                count[r_new::d] += 1
                
                new_score = np.sum(count == 0)
                
                if new_score < old_score:
                    residues[d] = r_new
                    improved = True
                else:
                    # Revert
                    count[r_new::d] -= 1
                    count[r_old::d] += 1
                    break # If first random move fails, stop trying randoms for this pass?
                    
            if not improved:
                break

    return [[d, int(residues[d])] for d in divisors]
