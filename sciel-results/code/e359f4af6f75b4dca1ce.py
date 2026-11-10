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
    deadline = start_time + 118  # Safety margin before 120s limit
    
    divisors = get_divisors(N)
    
    best_uncovered = N
    best_classes = []
    
    # Precompute index arrays for all (m, r) pairs to speed up delta calculations
    # For each modulus m, precompute np.arange(r, N, m) for all r in range(m)
    # This is memory intensive but N is small enough (51975)
    # Total memory: sum of N/m for all divisors m * m residues = sum of N for each m = num_divisors * N
    # num_divisors is about 48, so 48 * 51975 * 4 bytes = ~10MB, acceptable
    
    # Actually, let's be smarter. We only need the arrays when we use them.
    # But precomputing can save time. Let's precompute for small moduli which are used more.
    
    # For now, let's keep the on-the-fly approach but optimize the delta calculation.
    
    num_restarts = 15  # More restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 25:
            break
        
        # Greedy Construction with cost-aware selection
        covered = np.zeros(N, dtype=np.uint8)
        cover_count = np.zeros(N, dtype=np.int16)
        classes = []
        available_moduli = set(divisors)
        
        # Greedy Step: select class maximizing newly covered residues
        # The "cost" consideration: smaller moduli cover more residues per class,
        # so they have higher impact. We prioritize them implicitly by their gain.
        
        while available_moduli:
            if time.time() > deadline - 25:
                break
            
            best_gain = 0
            best_m = None
            best_r = None
            
            for m in list(available_moduli):
                # For efficiency, check residues that are likely to give high gain
                # Start with r=0 and check a few others
                max_cov = 0
                best_r_for_m = 0
                
                # Sample residues for larger moduli to save time
                if m > 100:
                    residues_to_check = [0, m//2, m//3, 1, m-1]
                    residues_to_check = [r % m for r in residues_to_check if 0 <= r < m]
                    # Deduplicate while preserving order
                    seen = set()
                    unique_residues = []
                    for r in residues_to_check:
                        if r not in seen:
                            seen.add(r)
                            unique_residues.append(r)
                    residues_to_check = unique_residues
                else:
                    residues_to_check = list(range(m))
                
                for r in residues_to_check:
                    idx = np.arange(r, N, m)
                    cov = int(np.sum(covered[idx] == 0))
                    if cov > max_cov:
                        max_cov = cov
                        best_r_for_m = r
                
                if max_cov > best_gain:
                    best_gain = max_cov
                    best_m = m
                    best_r = best_r_for_m
            
            if best_gain <= 0:
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
        if time.time() > deadline - 15:
            continue
            
        # Prepare for SA
        current_uncovered = cur_uncovered
        current_classes = [c[:] for c in classes]
        current_cover_count = cover_count.copy()
        
        # SA Parameters
        T = 5.0
        cooling = 0.995
        min_T = 0.01
        max_steps = 15000
        
        used_moduli = {c[0] for c in current_classes}
        unused_moduli = [m for m in divisors if m not in used_moduli]
        
        def compute_delta(old_m, old_r, new_m, new_r, cover_count):
            """
            Compute the change in uncovered count when replacing (old_m, old_r) with (new_m, new_r).
            If new_m is None, it's a removal.
            """
            if old_m is None:
                # Adding a class
                if new_m is None:
                    return 0
                new_idx = np.arange(new_r, N, new_m)
                counts_new = cover_count[new_idx]
                return -int(np.sum(counts_new == 0))
            
            if new_m is None:
                # Removing a class
                old_idx = np.arange(old_r, N, old_m)
                counts_old = cover_count[old_idx]
                return int(np.sum(counts_old == 1))
            
            # Swapping
            old_idx = np.arange(old_r, N, old_m)
            new_idx = np.arange(new_r, N, new_m)
            
            # Find intersection efficiently
            # Use boolean arrays for membership
            old_mask = np.zeros(N, dtype=bool)
            new_mask = np.zeros(N, dtype=bool)
            old_mask[old_idx] = True
            new_mask[new_idx] = True
            
            only_old_mask = old_mask & ~new_mask
            only_new_mask = new_mask & ~old_mask
            
            delta = 0
            if only_old_mask.any():
                only_old_indices = np.where(only_old_mask)[0]
                counts_only_old = cover_count[only_old_indices]
                delta += int(np.sum(counts_only_old == 1))
            
            if only_new_mask.any():
                only_new_indices = np.where(only_new_mask)[0]
                counts_only_new = cover_count[only_new_indices]
                delta -= int(np.sum(counts_only_new == 0))
            
            return delta

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
                
                delta = compute_delta(m, old_r, m, new_r, current_cover_count)
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_uncovered += delta
                    old_idx = np.arange(old_r, N, m)
                    new_idx = np.arange(new_r, N, m)
                    current_cover_count[old_idx] -= 1
                    current_cover_count[new_idx] += 1
                    current_classes[i][1] = new_r
                    
            elif move_type < 0.9:
                # Swap modulus
                if not current_classes or not unused_moduli:
                    continue
                
                i = random.randint(0, len(current_classes) - 1)
                old_m, old_r = current_classes[i]
                new_m = random.choice(unused_moduli)
                new_r = random.randint(0, new_m - 1)
                
                delta = compute_delta(old_m, old_r, new_m, new_r, current_cover_count)
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_uncovered += delta
                    old_idx = np.arange(old_r, N, old_m)
                    new_idx = np.arange(new_r, N, new_m)
                    current_cover_count[old_idx] -= 1
                    current_cover_count[new_idx] += 1
                    current_classes[i][0] = new_m
                    current_classes[i][1] = new_r
                    used_moduli.remove(old_m)
                    used_moduli.add(new_m)
                    unused_moduli.remove(new_m)
                    unused_moduli.append(old_m)
                    
            else:
                # Remove a class
                if not current_classes:
                    continue
                
                i = random.randint(0, len(current_classes) - 1)
                old_m, old_r = current_classes[i]
                
                delta = compute_delta(old_m, old_r, None, None, current_cover_count)
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_uncovered += delta
                    old_idx = np.arange(old_r, N, old_m)
                    current_cover_count[old_idx] -= 1
                    removed_class = current_classes.pop(i)
                    used_moduli.remove(removed_class[0])
                    unused_moduli.append(removed_class[0])
            
            T *= cooling
            
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_classes = [c[:] for c in current_classes]
                
    return best_classes
