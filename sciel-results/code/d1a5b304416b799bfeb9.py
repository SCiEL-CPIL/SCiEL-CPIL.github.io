import time
import numpy as np
import random

def get_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted([d for d in divisors if d > 1])

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    divisors = get_divisors(N)
    
    best_uncovered = N
    best_classes = []
    
    # Run multiple iterations of Greedy + SA
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Greedy Construction with ratio-based selection
        covered = np.zeros(N, dtype=np.uint8)
        cover_count = np.zeros(N, dtype=np.int16)
        classes = []
        available_moduli = set(divisors)
        
        # Greedy Step: maximize ratio of new coverage to cost (1/m)
        while available_moduli:
            if time.time() > deadline - 20:
                break
            
            best_ratio = -1
            best_m = None
            best_r = None
            
            for m in list(available_moduli):
                max_cov = 0
                best_r_for_m = 0
                
                for r in range(m):
                    idx = np.arange(r, N, m)
                    cov = int(np.sum(covered[idx] == 0))
                    if cov > max_cov:
                        max_cov = cov
                        best_r_for_m = r
                
                # Ratio: new coverage / cost, where cost = 1/m (so ratio = cov * m)
                # This prioritizes larger moduli that cover more total residues
                ratio = max_cov * m
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_m = m
                    best_r = best_r_for_m
            
            if best_m is None or best_ratio <= 0:
                break
            
            m = best_m
            r = best_r
            idx = np.arange(r, N, m)
            covered[idx] = 1
            cover_count[idx] += 1
            classes.append([m, r])
            available_moduli.remove(m)
        
        cur_uncovered = int(np.sum(covered == 0))
        
        if cur_uncovered < best_uncovered:
            best_uncovered = cur_uncovered
            best_classes = [c[:] for c in classes]
        
        # Local Search (Simulated Annealing)
        if time.time() > deadline - 10:
            continue
        
        current_uncovered = cur_uncovered
        current_classes = [c[:] for c in classes]
        current_cover_count = cover_count.copy()
        
        T = 1.0
        cooling = 0.99
        min_T = 0.01
        max_steps = 8000  # Increased steps due to faster delta computation
        
        used_moduli = {c[0] for c in current_classes}
        unused_moduli = [m for m in divisors if m not in used_moduli]
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if T < min_T:
                break
            
            move_type = random.random()
            
            if move_type < 0.7:
                # Change residue
                if not current_classes:
                    continue
                i = random.randint(0, len(current_classes) - 1)
                m, old_r = current_classes[i]
                new_r = random.randint(0, m - 1)
                if new_r == old_r:
                    continue
                
                old_idx = np.arange(old_r, N, m)
                new_idx = np.arange(new_r, N, m)
                
                # Fast delta computation using set operations
                old_set = set(old_idx.tolist())
                new_set = set(new_idx.tolist())
                
                # Indices that lose coverage: in old_set but not in new_set, and cover_count becomes 0
                lost = [x for x in old_set - new_set if current_cover_count[x] == 1]
                # Indices that gain coverage: in new_set but not in old_set, and cover_count was 0
                gained = [x for x in new_set - old_set if current_cover_count[x] == 0]
                
                delta = len(lost) - len(gained)
                
                # Apply if accepting
                if delta < 0 or (delta == 0 and random.random() < 0.1) or (random.random() < np.exp(-delta / T)):
                    current_cover_count[old_idx] -= 1
                    current_cover_count[new_idx] += 1
                    current_uncovered += delta
                    current_classes[i][1] = new_r
                else:
                    # Reject: no state change needed since we didn't modify cover_count
                    pass
            
            else:
                # Swap modulus
                if not current_classes or not unused_moduli:
                    continue
                
                i = random.randint(0, len(current_classes) - 1)
                old_m, old_r = current_classes[i]
                new_m = random.choice(unused_moduli)
                new_r = random.randint(0, new_m - 1)
                
                old_idx = np.arange(old_r, N, old_m)
                new_idx = np.arange(new_r, N, new_m)
                
                old_set = set(old_idx.tolist())
                new_set = set(new_idx.tolist())
                
                lost = [x for x in old_set - new_set if current_cover_count[x] == 1]
                gained = [x for x in new_set - old_set if current_cover_count[x] == 0]
                
                delta = len(lost) - len(gained)
                
                if delta < 0 or (delta == 0 and random.random() < 0.1) or (random.random() < np.exp(-delta / T)):
                    current_cover_count[old_idx] -= 1
                    current_cover_count[new_idx] += 1
                    current_uncovered += delta
                    current_classes[i][0] = new_m
                    current_classes[i][1] = new_r
                    used_moduli.remove(old_m)
                    used_moduli.add(new_m)
                    unused_moduli.remove(new_m)
                    unused_moduli.append(old_m)
                else:
                    pass
            
            T *= cooling
            
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_classes = [c[:] for c in current_classes]
        
    return best_classes
