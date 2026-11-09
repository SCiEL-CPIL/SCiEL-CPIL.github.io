import numpy as np
import time
import random

def construct():
    N = 51975
    
    # Generate divisors
    divisors = []
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    
    # Filter moduli > 1
    moduli = [m for m in divisors if m > 1]
    moduli.sort()
    
    # Precompute indices for each modulus to speed up updates?
    # No, slicing r::m is fast in numpy.
    
    cover_count = np.zeros(N, dtype=np.int32)
    current_classes = []
    used_moduli = set()
    
    def get_uncovered_count():
        return np.sum(cover_count == 0)
    
    def add_class(m, r):
        cover_count[r::m] += 1
        current_classes.append((m, r))
        used_moduli.add(m)
        
    def remove_class(m, r):
        cover_count[r::m] -= 1
        current_classes.remove((m, r))
        used_moduli.remove(m)
        
    def change_residue(m, r_old, r_new):
        cover_count[r_old::m] -= 1
        cover_count[r_new::m] += 1
        # Update the class record
        for i, (cm, cr) in enumerate(current_classes):
            if cm == m and cr == r_old:
                current_classes[i] = (m, r_new)
                break

    # Initialization: Greedy
    # We want to minimize uncovered.
    # Greedy: Pick the class that reduces uncovered count the most.
    # To do this efficiently, we can just iterate through all possible (m, r) pairs 
    # and pick the best one, add it, and repeat.
    # There are ~48 moduli. For each m, there are m residues.
    # Total pairs = sum(m for m in moduli).
    # Sum of divisors of 51975?
    # sigma(51975) = sigma(27)*sigma(25)*sigma(7)*sigma(11)
    # sigma(27) = 1+3+9+27 = 40
    # sigma(25) = 1+5+25 = 31
    # sigma(7) = 8
    # sigma(11) = 12
    # Product = 40 * 31 * 8 * 12 = 12480.
    # 12480 pairs.
    # For each pair, calculating the improvement requires checking N/m positions.
    # Total work for one greedy step: sum(m * N/m) = sum(N) = 12480 * N? No.
    # For a specific m, testing all r costs m * (N/m) = N.
    # So testing all r for a specific m costs N operations.
    # Testing all m costs |moduli| * N = 48 * 51975 ~ 2.5M operations.
    # We can do this maybe 20-50 times in 120s?
    # 2.5M numpy ops is very fast, likely < 0.1s.
    # So we can do a "Full Greedy" initialization:
    # While there are unused moduli:
    #   Find the (m, r) pair among all unused m and all r that minimizes uncovered count.
    #   Add it.
    
    best_uncovered = get_uncovered_count()
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Full Greedy Init
    while len(current_classes) < len(moduli):
        best_gain = -1
        best_m = None
        best_r = None
        
        for m in moduli:
            if m in used_moduli:
                continue
            # Try all residues for this m
            # We can vectorize this?
            # For each r, count how many x in r::m have cover_count[x] == 0.
            # This is sum(cover_count[r::m] == 0).
            # Doing this in a loop for r in range(m) is m iterations of numpy ops.
            # Total N ops per m.
            
            # Optimization: 
            # The number of newly covered residues is the number of x in r::m 
            # such that cover_count[x] == 0.
            
            # Let's just loop r. It's fast enough.
            for r in range(m):
                # Count zeros in cover_count[r::m]
                # This creates a view, so it's fast.
                newly_covered = np.sum(cover_count[r::m] == 0)
                if newly_covered > best_gain:
                    best_gain = newly_covered
                    best_m = m
                    best_r = r
                    
        if best_gain <= 0:
            break
            
        add_class(best_m, best_r)
        best_uncovered = get_uncovered_count()
        
        if time.time() > deadline:
            break

    # Local Search
    # We will try to improve further.
    # Moves:
    # 1. Change residue of an existing class.
    # 2. Add a new class (if any unused).
    # 3. Remove a class.
    # 4. Swap.
    
    # To speed up, we can pre-compute the list of unused moduli.
    
    while time.time() < deadline:
        move_type = random.choice(['change', 'add', 'remove', 'swap'])
        
        if move_type == 'change' and current_classes:
            idx = random.randint(0, len(current_classes) - 1)
            m, r_old = current_classes[idx]
            if m == 1: continue
            r_new = random.randint(0, m - 1)
            if r_new == r_old:
                continue
            
            # Apply
            cover_count[r_old::m] -= 1
            cover_count[r_new::m] += 1
            new_uncovered = get_uncovered_count()
            
            if new_uncovered <= best_uncovered:
                # Accept
                current_classes[idx] = (m, r_new)
                if new_uncovered < best_uncovered:
                    best_uncovered = new_uncovered
            else:
                # Revert
                cover_count[r_new::m] -= 1
                cover_count[r_old::m] += 1
                
        elif move_type == 'add':
            unused = [m for m in moduli if m not in used_moduli]
            if unused:
                m = random.choice(unused)
                r = random.randint(0, m - 1)
                
                # Apply
                cover_count[r::m] += 1
                new_uncovered = get_uncovered_count()
                
                if new_uncovered <= best_uncovered:
                    # Accept
                    current_classes.append((m, r))
                    used_moduli.add(m)
                    if new_uncovered < best_uncovered:
                        best_uncovered = new_uncovered
                else:
                    # Revert
                    cover_count[r::m] -= 1
                    
        elif move_type == 'remove' and current_classes:
            idx = random.randint(0, len(current_classes) - 1)
            m, r = current_classes[idx]
            
            # Apply
            cover_count[r::m] -= 1
            new_uncovered = get_uncovered_count()
            
            if new_uncovered <= best_uncovered:
                # Accept
                current_classes.pop(idx)
                used_moduli.remove(m)
                if new_uncovered < best_uncovered:
                    best_uncovered = new_uncovered
            else:
                # Revert
                cover_count[r::m] += 1
                
        elif move_type == 'swap' and current_classes:
            unused = [m for m in moduli if m not in used_moduli]
            if unused:
                idx = random.randint(0, len(current_classes) - 1)
                m_old, r_old = current_classes[idx]
                m_new = random.choice(unused)
                r_new = random.randint(0, m_new - 1)
                
                # Apply
                cover_count[r_old::m_old] -= 1
                cover_count[r_new::m_new] += 1
                new_uncovered = get_uncovered_count()
                
                if new_uncovered <= best_uncovered:
                    # Accept
                    current_classes[idx] = (m_new, r_new)
                    used_moduli.remove(m_old)
                    used_moduli.add(m_new)
                    if new_uncovered < best_uncovered:
                        best_uncovered = new_uncovered
                else:
                    # Revert
                    cover_count[r_new::m_new] -= 1
                    cover_count[r_old::m_old] += 1
                    # Note: current_classes[idx] is still (m_old, r_old) in memory, which is correct for revert.
                    # But we need to ensure used_moduli is consistent. 
                    # Since we didn't change current_classes, used_moduli should not have changed.
                    # In the accept branch, we update used_moduli. In revert, we do nothing.
                    # This is correct.

    # Return in required format [[m, r], ...]
    return [[m, r] for m, r in current_classes]
