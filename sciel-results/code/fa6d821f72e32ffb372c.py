import time
import random
import numpy as np
from math import log

def _compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    n = len(U)
    if n == 0 or 0 not in U:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    
    # Compute sumset and diffset
    # For small n, direct set comprehension is fast
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
            
    len_sum = len(sumset)
    len_diff = len(diffset)
    
    if len_sum == 0:
        return 0.0
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
        
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
        
    return 1 + log(ratio) / denominator

def _compute_score_fast(U):
    """Compute the score for a given set U using FFT."""
    U = sorted(set(U))
    if not U or 0 not in U:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    
    # Determine FFT size
    L = 1
    while L < 2 * max_u + 1:
        L *= 2
        
    A = np.zeros(L)
    for u in U:
        A[u] = 1.0
        
    F = np.fft.fft(A)
    
    # Sumset
    S_counts = np.fft.ifft(F * F).real
    len_sum = np.sum(S_counts > 0.5)
    
    # Difference set
    D_counts = np.fft.ifft(F * np.conj(F)).real
    pos_diffs = np.sum(D_counts[0:max_u+1] > 0.5)
    neg_diffs = np.sum(D_counts[L-max_u:L] > 0.5)
    len_diff = pos_diffs + neg_diffs
    
    if len_sum == 0:
        return 0.0
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
        
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
        
    return 1 + log(ratio) / denominator

def _generate_greedy_sidon(max_val, target_size):
    """Generate a greedy Sidon set."""
    U = [0]
    for x in range(1, max_val + 1):
        if len(U) >= target_size:
            break
        # Check if x can be added to maintain Sidon property
        # Sidon: all sums a+b (a<=b) are distinct
        # Equivalent: no a+b = c+d with {a,b} != {c,d}
        # Check collisions with existing elements
        valid = True
        current_sums = set()
        for a in U:
            for b in U:
                if a >= b:
                    current_sums.add(a + b)
        
        # New sums involving x
        new_sums = []
        for a in U:
            s = a + x
            new_sums.append(s)
            if s in current_sums:
                valid = False
                break
        if valid:
            # Check for collisions among new sums themselves?
            # a+x = b+x implies a=b, so no collision among new sums if a,b distinct.
            # But we must check if a+x = b+x is possible? No, x is constant.
            # We must check if a+x = c+d where c,d in U?
            # This is covered by checking if s in current_sums.
            # Wait, current_sums contains a+b for a,b in U.
            # If a+x = b+c, and b,c in U, then s is in current_sums?
            # No, current_sums is built from U. If b,c in U, then b+c is in current_sums.
            # So checking s in current_sums is correct.
            
            # Also need to check if x+x collides?
            # x+x = 2x. Is 2x in current_sums?
            # 2x = a+b. If so, collision.
            # My loop adds a+x for all a in U.
            # If a=x, then x+x. But x is not in U yet.
            # So I need to explicitly check 2x.
            if 2*x in current_sums:
                valid = False
            
            if valid:
                U.append(x)
                # Update current_sums for next iterations?
                # To be efficient, we should maintain the set of sums.
                # But for small target_size, recomputing is fine.
                
    return U

def _sa_search(initial_set, max_val, timeout=10.0):
    """Simulated Annealing to find a good set."""
    start_time = time.time()
    
    current = sorted(set(initial_set))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score_fast(current)
    
    best = list(current)
    best_score = current_score
    
    T = 0.1
    T_min = 1e-6
    cooling = 0.999
    
    while time.time() - start_time < timeout:
        if T < T_min:
            break
            
        # Generate neighbor
        move_type = random.random()
        new_set = list(current)
        
        if move_type < 0.5:
            # Change an element
            if len(new_set) <= 1:
                continue
            idx = random.randint(0, len(new_set) - 1)
            if idx == 0 and new_set[0] == 0:
                # Don't change 0
                idx = random.randint(1, len(new_set) - 1) if len(new_set) > 1 else 0
            old_val = new_set[idx]
            # Try nearby values
            delta = random.randint(-5, 5)
            new_val = old_val + delta
            if new_val < 0:
                new_val = random.randint(0, max_val)
            if new_val > max_val:
                new_val = random.randint(0, max_val)
            if new_val in new_set and new_val != old_val:
                continue
            new_set[idx] = new_val
        elif move_type < 0.75:
            # Add an element
            new_elem = random.randint(0, max_val)
            if new_elem not in new_set:
                new_set.append(new_elem)
            else:
                continue
        else:
            # Remove an element (not 0)
            removable = [x for x in new_set if x != 0]
            if not removable:
                continue
            rem = random.choice(removable)
            new_set.remove(rem)
            
        new_set = sorted(set(new_set))
        if 0 not in new_set or len(new_set) < 2:
            continue
            
        new_score = _compute_score_fast(new_set)
        
        # Acceptance
        if new_score >= current_score or random.random() < np.exp((new_score - current_score) / T):
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = list(current)
                
        T *= cooling
        
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    best_result = [0, 1, 3, 7]
    best_score = _compute_score_fast(best_result)
    
    # Generate diverse initial sets
    initial_sets = []
    
    # Greedy Sidon sets of various sizes
    for size in range(3, 20):
        s = _generate_greedy_sidon(200, size)
        if len(s) >= 2:
            initial_sets.append(s)
            
    # Random sets
    for _ in range(10):
        s = {0}
        for _ in range(random.randint(5, 15)):
            s.add(random.randint(1, 100))
        initial_sets.append(sorted(s))
        
    # Specific promising structures
    initial_sets.append([0, 1, 3, 9])
    initial_sets.append([0, 1, 4, 9])
    initial_sets.append([0, 1, 3, 7, 12])
    
    # Run SA on each initial set
    for init in initial_sets:
        if time.time() > deadline - 5:
            break
            
        # Determine max_val for search
        # Keep it small to maintain high score, but allow some growth
        max_val = max(max(init) * 2, 100)
        
        # Time budget per run
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        sa_timeout = min(15.0, remaining * 0.2)
        
        result, score = _sa_search(init, max_val, timeout=sa_timeout)
        
        if score > best_score:
            best_score = score
            best_result = result
            
    # Final refinement on the best result
    if time.time() < deadline - 5:
        max_val_refine = max(max(best_result) * 2, 100)
        remaining = deadline - time.time()
        result, score = _sa_search(best_result, max_val_refine, timeout=min(20.0, remaining - 2))
        if score > best_score:
            best_score = score
            best_result = result
            
    # Ensure validity
    final_score = _compute_score_fast(best_result)
    if final_score < 1.0:
        best_result = [0, 1, 3, 7]
        
    return best_result
