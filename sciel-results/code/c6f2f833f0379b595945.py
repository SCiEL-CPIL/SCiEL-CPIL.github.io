import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
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
        
    # Greedy Initialization
    # Sort divisors in ascending order to cover more residues initially
    greedy_order = divisors.copy()
    # Maybe try a few different orders? Let's stick to ascending for now, 
    # but we can shuffle if time permits.
    
    for m in greedy_order:
        best_r = 0
        best_uncovered = uncovered + 1
        # We can sample residues if m is large, but m is small enough to iterate all
        # N=51975, max m=51975/3=17325. Iterating 17k residues for each of 47 moduli is ~800k ops.
        # Each op is O(N/m). This might be slow in pure Python loop, but we are just finding best_r.
        # We can use numpy to evaluate all residues at once? 
        # No, because we need to update counts to see the effect.
        # However, we can estimate the gain.
        # Gain = number of x in progression r where counts[x] == 0.
        # We can compute this for all r in parallel using numpy?
        # Let's try.
        
        # Create a matrix of indices? Too large.
        # Just loop r.
        # To speed up, we can break early if we find a perfect cover (gain == N/m)
        max_gain = N // m
        for r in range(m):
            idx = np.arange(r, N, m)
            gain = np.sum(counts[idx] == 0)
            if gain > best_uncovered - uncovered: # This logic is wrong, best_uncovered is absolute
                pass
            
        # Correct logic:
        best_gain = -1
        for r in range(m):
            idx = np.arange(r, N, m)
            gain = np.sum(counts[idx] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
                
        if best_gain > 0:
            classes.append([m, best_r])
            used_moduli.add(m)
            update_add(m, best_r)
            
    # Simulated Annealing
    current_uncovered = uncovered
    best_uncovered = uncovered
    best_classes = [c.copy() for c in classes]
    
    # SA Parameters
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute unused moduli
    all_divisors_set = set(divisors)
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
            
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.5 and classes:
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
                    
        elif move_type < 0.7 and len(classes) >= 2:
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
                    
        elif move_type < 0.85:
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
        
    # Polish: Local Search
    # Restore best state
    # We need to reconstruct counts and uncovered from best_classes
    counts = np.zeros(N, dtype=np.int32)
    uncovered = N
    for m, r in best_classes:
        update_add(m, r)
        
    classes = [c.copy() for c in best_classes]
    used_moduli = set(m for m, _ in classes)
    
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for i in range(len(classes)):
            if time.time() > deadline - 1:
                break
            m, r = classes[i]
            best_r = r
            best_uncovered_local = uncovered
            
            # Try all residues
            for new_r in range(m):
                if new_r == r:
                    continue
                # Evaluate
                update_remove(m, r)
                update_add(m, new_r)
                
                if uncovered < best_uncovered_local:
                    best_uncovered_local = uncovered
                    best_r = new_r
                    
                # Revert
                update_remove(m, new_r)
                update_add(m, r)
                
            if best_r != r:
                # Apply best
                update_remove(m, r)
                update_add(m, best_r)
                classes[i][1] = best_r
                improved = True
                uncovered = best_uncovered_local
                # Restart loop as other residues might be better now
                break
                
    return classes
