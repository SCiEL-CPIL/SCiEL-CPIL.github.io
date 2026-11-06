import numpy as np
import time
import random

def construct():
    N = 51975
    # Find all divisors of N greater than 1
    divisors = [d for d in range(2, N + 1) if N % d == 0]
    
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    def greedy_init(order):
        """Greedy initialization: assign residues to cover as many uncovered as possible."""
        res = {}
        count = np.zeros(N, dtype=np.int8)
        for d in order:
            # Find residue class that covers most currently uncovered
            uncovered = np.nonzero(count == 0)[0]
            if len(uncovered) == 0:
                res[d] = 0
                continue
            mods = uncovered % d
            gains = np.bincount(mods, minlength=d)
            best_r = int(np.argmax(gains))
            res[d] = best_r
            count[best_r::d] += 1
        return res, count
    
    # Try multiple initialization orders
    best_residues = None
    best_count = None
    best_score = N  # worst case
    
    # Order 1: ascending divisors
    asc_divs = sorted(divisors)
    res1, count1 = greedy_init(asc_divs)
    score1 = int(np.sum(count1 == 0))
    if score1 < best_score:
        best_score = score1
        best_residues = res1
        best_count = count1
    
    # Order 2: descending divisors
    desc_divs = sorted(divisors, reverse=True)
    res2, count2 = greedy_init(desc_divs)
    score2 = int(np.sum(count2 == 0))
    if score2 < best_score:
        best_score = score2
        best_residues = res2
        best_count = count2
    
    # Order 3: random
    for _ in range(3):
        if time.time() - start_time > timeout:
            break
        rand_divs = divisors[:]
        random.shuffle(rand_divs)
        res3, count3 = greedy_init(rand_divs)
        score3 = int(np.sum(count3 == 0))
        if score3 < best_score:
            best_score = score3
            best_residues = res3
            best_count = count3
    
    # Now do local search from the best initialization
    residues = best_residues.copy()
    count = best_count.copy()
    
    if best_score == 0:
        return [[d, int(residues[d])] for d in divisors]
    
    while time.time() - start_time < timeout:
        improved = False
        shuffled_divs = divisors[:]
        random.shuffle(shuffled_divs)
        
        for d in shuffled_divs:
            if time.time() - start_time > timeout:
                break
                
            r_old = residues[d]
            
            # Calculate loss: number of x in class (d, r_old) that are covered exactly once
            x_old = np.arange(r_old, N, d)
            loss = int(np.sum(count[x_old] == 1))
            
            # Get indices of currently uncovered residues
            uncovered_idx = np.nonzero(count == 0)[0]
            
            if len(uncovered_idx) == 0:
                return [[d, int(residues[d])] for d in divisors]
                
            # For each uncovered x, x % d gives the residue class it falls into.
            mods = uncovered_idx % d
            gains = np.bincount(mods, minlength=d)
            
            # Find the best residue to switch to
            best_r = int(np.argmax(gains))
            max_gain = int(gains[best_r])
            
            # Check if switching improves the score
            if best_r != r_old and max_gain > loss:
                # Update count array
                count[r_old::d] -= 1
                count[best_r::d] += 1
                residues[d] = best_r
                improved = True
                
        if not improved:
            # Try random perturbations to escape local minima
            escaped = False
            for _ in range(20):
                if time.time() - start_time > timeout:
                    break
                d = random.choice(divisors)
                r_old = residues[d]
                r_new = random.randint(0, d - 1)
                if r_new == r_old:
                    continue
                
                x_old = np.arange(r_old, N, d)
                loss = int(np.sum(count[x_old] == 1))
                
                x_new = np.arange(r_new, N, d)
                gain = int(np.sum(count[x_new] == 0))
                
                # Accept if neutral or better, or with probability for worse moves (simulated annealing-like)
                if gain >= loss:
                    count[r_old::d] -= 1
                    count[r_new::d] += 1
                    residues[d] = r_new
                    escaped = True
                    break
                elif random.random() < 0.1:  # 10% chance to accept worse move
                    count[r_old::d] -= 1
                    count[r_new::d] += 1
                    residues[d] = r_new
                    escaped = True
                    break
                    
            if not escaped:
                # Try a larger perturbation: change 3 random divisors
                for _ in range(10):
                    if time.time() - start_time > timeout:
                        break
                    d1, d2, d3 = random.sample(divisors, 3)
                    r1_old, r2_old, r3_old = residues[d1], residues[d2], residues[d3]
                    r1_new = random.randint(0, d1 - 1)
                    r2_new = random.randint(0, d2 - 1)
                    r3_new = random.randint(0, d3 - 1)
                    
                    # Save old state
                    old_state = (r1_old, r2_old, r3_old)
                    
                    # Apply new state
                    count[r1_old::d1] -= 1
                    count[r2_old::d2] -= 1
                    count[r3_old::d3] -= 1
                    count[r1_new::d1] += 1
                    count[r2_new::d2] += 1
                    count[r3_new::d3] += 1
                    residues[d1] = r1_new
                    residues[d2] = r2_new
                    residues[d3] = r3_new
                    
                    # Check if this helped
                    current_score = int(np.sum(count == 0))
                    if current_score < best_score or random.random() < 0.2:
                        if current_score < best_score:
                            best_score = current_score
                            best_residues = residues.copy()
                        escaped = True
                        break
                    else:
                        # Revert
                        count[r1_new::d1] -= 1
                        count[r2_new::d2] -= 1
                        count[r3_new::d3] -= 1
                        count[r1_old::d1] += 1
                        count[r2_old::d2] += 1
                        count[r3_old::d3] += 1
                        residues[d1] = r1_old
                        residues[d2] = r2_old
                        residues[d3] = r3_old
                        
                if not escaped:
                    break
        else:
            # Had improvements, update best
            current_score = int(np.sum(count == 0))
            if current_score < best_score:
                best_score = current_score
                best_residues = residues.copy()
            if best_score == 0:
                break
    
    return [[d, int(best_residues[d])] for d in divisors]
