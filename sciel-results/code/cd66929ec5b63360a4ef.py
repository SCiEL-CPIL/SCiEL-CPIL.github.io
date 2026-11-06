import time
import math
import numpy as np

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145.0  # 145s limit to be safe
    
    # 1. Find all divisors of 51975
    divisors = []
    for i in range(1, int(math.isqrt(base)) + 1):
        if base % i == 0:
            if i > 1:
                divisors.append(i)
            other = base // i
            if other > 1 and other != i:
                divisors.append(other)
    
    # Sort divisors
    divisors.sort()
    
    # 2. Precompute masks and residue lists for each (m, r)
    masks = {}
    residues = {}
    for m in divisors:
        masks[m] = [0] * m
        residues[m] = [[] for _ in range(m)]
        for r in range(m):
            mask_val = 0
            res_list = []
            for i in range(r, base, m):
                mask_val |= (1 << i)
                res_list.append(i)
            masks[m][r] = mask_val
            residues[m][r] = res_list
            
    # 3. Greedy Set Cover
    uncovered_mask = (1 << base) - 1
    remaining_moduli = set(divisors)
    classes = []
    used_moduli = set()
    
    # freq[x] = number of classes covering residue x
    freq = [0] * base
    freq_np = np.zeros(base, dtype=np.int32)
    
    while remaining_moduli:
        best_gain = 0
        best_m = None
        best_r = None
        
        # Find the modulus and residue that maximizes coverage of uncovered
        for m in remaining_moduli:
            # For this m, find r that maximizes intersection with uncovered
            for r in range(m):
                # Gain is number of bits in (uncovered_mask & masks[m][r])
                gain = (uncovered_mask & masks[m][r]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
                    
        if best_gain == 0:
            break
            
        # Select the best class
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        remaining_moduli.remove(best_m)
        
        # Update uncovered and freq
        m = best_m
        r = best_r
        for x in residues[m][r]:
            if freq[x] == 0:
                uncovered_mask &= ~(1 << x)
            freq[x] += 1
            freq_np[x] += 1
        
        # Time check
        if time.time() > deadline:
            break
            
    # 4. Local Search using NumPy for speed
    # Helper to calculate gain of adding (m, r) using numpy
    def calc_gain_np(m, r):
        arr = np.array(residues[m][r], dtype=np.int32)
        return int(np.sum(freq_np[arr] == 0))

    # Helper to calculate loss of removing (m, r) using numpy
    def calc_loss_np(m, r):
        arr = np.array(residues[m][r], dtype=np.int32)
        return int(np.sum(freq_np[arr] == 1))

    # Add phase: Add any class that reduces uncovered count
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_m = None
        best_r = None
        
        # Try adding an unused modulus
        for m in divisors:
            if m in used_moduli:
                continue
            for r in range(m):
                gain = calc_gain_np(m, r)
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
                    
        if best_gain > 0:
            m = best_m
            r = best_r
            classes.append([m, r])
            used_moduli.add(m)
            for x in residues[m][r]:
                if freq[x] == 0:
                    uncovered_mask &= ~(1 << x)
                freq[x] += 1
                freq_np[x] += 1
            improved = True

    # Swap phase: Try to swap a used class with an unused class
    # More iterations and better candidate selection
    for iteration in range(30):
        if time.time() > deadline:
            break
        
        best_swap_gain = 0
        best_old_idx = -1
        best_new_m = None
        best_new_r = None
        
        # Calculate gains for all unused moduli
        unused_candidates = []
        for m in divisors:
            if m in used_moduli:
                continue
            best_r_for_m = 0
            best_gain_for_m = -1
            for r in range(m):
                g = calc_gain_np(m, r)
                if g > best_gain_for_m:
                    best_gain_for_m = g
                    best_r_for_m = r
            if best_gain_for_m > 0:
                unused_candidates.append((best_gain_for_m, m, best_r_for_m))
        
        unused_candidates.sort(reverse=True)
        
        # Calculate losses for all used classes
        used_candidates = []
        for idx, (m, r) in enumerate(classes):
            loss = calc_loss_np(m, r)
            used_candidates.append((loss, idx, m, r))
        
        used_candidates.sort() # Lowest loss first
        
        # Try more candidates: top 8 unused with top 8 used
        for g_new, m_new, r_new in unused_candidates[:8]:
            for l_old, idx_old, m_old, r_old in used_candidates[:8]:
                net_gain = g_new - l_old
                if net_gain > best_swap_gain:
                    best_swap_gain = net_gain
                    best_old_idx = idx_old
                    best_new_m = m_new
                    best_new_r = r_new
        
        if best_swap_gain > 0 and best_old_idx != -1:
            # Perform swap
            m_old, r_old = classes[best_old_idx]
            
            # Remove old
            used_moduli.remove(m_old)
            for x in residues[m_old][r_old]:
                freq[x] -= 1
                freq_np[x] -= 1
                if freq[x] == 0:
                    uncovered_mask |= (1 << x)
            classes.pop(best_old_idx)
            
            # Add new
            used_moduli.add(best_new_m)
            for x in residues[best_new_m][best_new_r]:
                if freq[x] == 0:
                    uncovered_mask &= ~(1 << x)
                freq[x] += 1
                freq_np[x] += 1
            classes.append([best_new_m, best_new_r])
        else:
            # No improving swap found; try one more pass with different top-k
            if iteration >= 15:
                break
    
    return classes
