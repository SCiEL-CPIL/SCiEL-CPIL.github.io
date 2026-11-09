import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    N = 51975
    base = N
    
    # Generate divisors
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # Initialize coverage counts
    # counts[x] = number of classes covering x
    counts = np.zeros(N, dtype=np.int32)
    uncovered = N
    
    # State: list of [m, r]
    classes = []
    used_moduli = set()
    
    def update_add(m, r):
        nonlocal uncovered
        idx = np.arange(r, N, m)
        old_counts = counts[idx]
        # New counts will be old + 1
        # Uncovered decreases where old_counts was 0
        dec = np.sum(old_counts == 0)
        counts[idx] += 1
        uncovered -= dec
        
    def update_remove(m, r):
        nonlocal uncovered
        idx = np.arange(r, N, m)
        old_counts = counts[idx]
        # New counts will be old - 1
        # Uncovered increases where new_counts is 0 (i.e., old_counts was 1)
        inc = np.sum(old_counts == 1)
        counts[idx] -= 1
        uncovered += inc
        
    def find_best_residue(m, counts_arr):
        # Find residue r that covers the most currently uncovered residues
        # Z[x] = 1 if counts_arr[x] == 0 else 0
        Z = (counts_arr == 0).astype(np.int32)
        best_r = 0
        best_gain = -1
        
        max_possible = N // m
        
        if m <= 200:
            # Small m: iterate residues with numpy slicing
            for r in range(m):
                gain = np.sum(Z[r::m])
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
                    if best_gain == max_possible:
                        break
        else:
            # Large m: each residue class covers at most a few elements.
            # Iterate over uncovered positions and tally gains per residue.
            uncov_idx = np.nonzero(Z)[0]
            if len(uncov_idx) == 0:
                return 0, 0
            # For each uncovered position x, its residue class is x % m
            residues = uncov_idx % m
            # Count occurrences of each residue
            # Use bincount for efficiency
            gains = np.bincount(residues, minlength=m)
            best_r = int(np.argmax(gains))
            best_gain = int(gains[best_r])
            
        return best_r, best_gain

    # Greedy Initialization: process larger moduli first (they cover fewer residues,
    # so they're more "specialized" and less likely to be subsumed)
    greedy_order = sorted(divisors, reverse=True)
    
    for m in greedy_order:
        best_r, best_gain = find_best_residue(m, counts)
        if best_gain > 0:
            classes.append([m, best_r])
            used_moduli.add(m)
            update_add(m, best_r)
            
    # Simulated Annealing
    current_uncovered = uncovered
    best_uncovered = uncovered
    best_classes = [c.copy() for c in classes]
    
    # SA Parameters - more aggressive cooling for faster convergence
    T = 20.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute unused moduli
    all_divisors_set = set(divisors)
    
    while time.time() < deadline - 10:
        if T < T_min:
            break
            
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.55 and classes:
            # Change residue of a random class
            i = random.randint(0, len(classes) - 1)
            m, r = classes[i]
            new_r = random.randint(0, m - 1)
            if new_r == r:
                continue
                
            # Evaluate
            old_r = r
            # Remove old
            update_remove(m, old_r)
            # Add new
            update_add(m, new_r)
            
            delta = uncovered - current_uncovered
            current_uncovered = uncovered
            classes[i][1] = new_r
            
            if delta > 0:
                # Worse
                if random.random() < np.exp(-delta / T):
                    pass # Accept
                else:
                    # Reject
                    update_remove(m, new_r)
                    update_add(m, old_r)
                    classes[i][1] = old_r
                    current_uncovered = uncovered
            else:
                # Better or equal
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_classes = [c.copy() for c in classes]
                    
        elif move_type < 0.75 and len(classes) >= 2:
            # Swap residues of two random classes
            i, j = random.sample(range(len(classes)), 2)
            m_i, r_i = classes[i]
            m_j, r_j = classes[j]
            
            # Remove both
            update_remove(m_i, r_i)
            update_remove(m_j, r_j)
            
            # Add swapped
            update_add(m_i, r_j)
            update_add(m_j, r_i)
            
            delta = uncovered - current_uncovered
            current_uncovered = uncovered
            classes[i][1] = r_j
            classes[j][1] = r_i
            
            if delta > 0:
                if random.random() < np.exp(-delta / T):
                    pass
                else:
                    # Reject
                    update_remove(m_i, r_j)
                    update_remove(m_j, r_i)
                    update_add(m_i, r_i)
                    update_add(m_j, r_j)
                    classes[i][1] = r_i
                    classes[j][1] = r_j
                    current_uncovered = uncovered
            else:
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_classes = [c.copy() for c in classes]
                    
        elif move_type < 0.90:
            # Add a class
            unused = [m for m in divisors if m not in used_moduli]
            if unused:
                m = random.choice(unused)
                r = random.randint(0, m - 1)
                
                update_add(m, r)
                classes.append([m, r])
                used_moduli.add(m)
                
                delta = uncovered - current_uncovered
                current_uncovered = uncovered
                
                if delta > 0:
                    if random.random() < np.exp(-delta / T):
                        pass
                    else:
                        # Reject
                        update_remove(m, r)
                        classes.pop()
                        used_moduli.remove(m)
                        current_uncovered = uncovered
                else:
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_classes = [c.copy() for c in classes]
                        
        else:
            # Remove a class
            if classes:
                i = random.randint(0, len(classes) - 1)
                m, r = classes[i]
                
                update_remove(m, r)
                classes.pop(i)
                used_moduli.remove(m)
                
                delta = uncovered - current_uncovered
                current_uncovered = uncovered
                
                if delta > 0:
                    if random.random() < np.exp(-delta / T):
                        pass
                    else:
                        # Reject
                        classes.insert(i, [m, r])
                        used_moduli.add(m)
                        update_add(m, r)
                        current_uncovered = uncovered
                else:
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_classes = [c.copy() for c in classes]
                        
        T *= cooling
        
    # Polish: Local Search with multiple rounds
    # Restore best state
    counts = np.zeros(N, dtype=np.int32)
    uncovered = N
    for m, r in best_classes:
        update_add(m, r)
        
    classes = [c.copy() for c in best_classes]
    used_moduli = set(m for m, _ in classes)
    
    # Multiple passes of local search - cycle through all classes until no improvement
    polish_rounds = 0
    max_polish_rounds = 50
    while time.time() < deadline - 3 and polish_rounds < max_polish_rounds:
        polish_rounds += 1
        improved = False
        for i in range(len(classes)):
            if time.time() > deadline - 3:
                break
            m, r = classes[i]
            
            # Remove current class to evaluate new residue
            update_remove(m, r)
            
            # Find best residue using fast method
            best_r, best_gain = find_best_residue(m, counts)
            
            if best_r != r:
                # Apply best
                update_add(m, best_r)
                classes[i][1] = best_r
                improved = True
            else:
                # Restore original
                update_add(m, r)
        
        if not improved:
            break
    
    # Second polish pass: try adding any unused modulus that helps, then re-optimize
    if time.time() < deadline - 3:
        for m in divisors:
            if m in used_moduli:
                continue
            best_r, best_gain = find_best_residue(m, counts)
            if best_gain > 0:
                update_add(m, best_r)
                classes.append([m, best_r])
                used_moduli.add(m)
        
        # Re-run local search after additions
        if time.time() < deadline - 2:
            for _ in range(5):
                if time.time() > deadline - 2:
                    break
                improved = False
                for i in range(len(classes)):
                    if time.time() > deadline - 2:
                        break
                    m, r = classes[i]
                    update_remove(m, r)
                    best_r, best_gain = find_best_residue(m, counts)
                    if best_r != r:
                        update_add(m, best_r)
                        classes[i][1] = best_r
                        improved = True
                    else:
                        update_add(m, r)
                if not improved:
                    break
    
    # Final cleanup: remove any class that doesn't cover any unique residue
    if time.time() < deadline - 1:
        for i in range(len(classes) - 1, -1, -1):
            if time.time() > deadline - 1:
                break
            m, r = classes[i]
            # Check if removing this class increases uncovered count
            update_remove(m, r)
            inc = uncovered - (N - sum(1 for x in range(N) if counts[x] > 0))
            # Simpler: just check if this class covers any residue it's the only one for
            # Actually, just check if removing it makes things worse
            # If removing it increases uncovered, keep it; otherwise remove
            if uncovered > best_uncovered:
                # This class is helpful, restore it
                update_add(m, r)
            else:
                # This class is not helpful, leave it removed
                classes.pop(i)
                used_moduli.remove(m)
    
    return classes
