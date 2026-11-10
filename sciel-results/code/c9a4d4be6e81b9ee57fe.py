import numpy as np
import time
import random

def construct():
    base = 51975
    # Get divisors
    divisors = []
    for i in range(2, int(base**0.5) + 1):
        if base % i == 0:
            divisors.append(i)
            if i != base // i:
                divisors.append(base // i)
    divisors.append(base)
    divisors = sorted(divisors)
    
    # State
    coverage = np.zeros(base, dtype=np.int8)
    classes = []
    used_moduli = set()
    
    # Greedy Init
    # Sort divisors by ascending size (small m covers more)
    for m in divisors:
        # Find best r
        best_r = 0
        best_gain = -1
        # Optimization: We can sample r if m is large? No, m <= 51975.
        # But sum of m is ~120k. Checking all r for all m is 120k checks.
        # Each check is O(N/m) operations in numpy?
        # No, we can vectorize the gain calculation for all r?
        # Gain(r) = sum(coverage[r::m] == 0)
        # This is hard to vectorize over r directly.
        # We'll just loop r.
        for r in range(m):
            idx = np.arange(r, base, m)
            gain = np.sum(coverage[idx] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        # Add class
        idx = np.arange(best_r, base, m)
        coverage[idx] += 1
        classes.append([m, best_r])
        used_moduli.add(m)
        
    current_score = np.sum(coverage == 0)
    best_score = current_score
    best_classes = [list(c) for c in classes]
    
    # Local Search / Simulated Annealing
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    T = 10.0 # Initial temperature
    cooling = 0.99
    
    while time.time() - start_time < timeout:
        # Pick a move
        move_type = random.choice(['change', 'add', 'remove'])
        
        if move_type == 'change' and classes:
            # Pick a random class
            i = random.randint(0, len(classes) - 1)
            m, r_old = classes[i]
            # Pick a new residue
            r_new = random.randint(0, m - 1)
            if r_new == r_old: continue
            
            # Calculate delta
            idx_old = np.arange(r_old, base, m)
            idx_new = np.arange(r_new, base, m)
            
            # Temporarily apply
            coverage[idx_old] -= 1
            coverage[idx_new] += 1
            new_score = np.sum(coverage == 0)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                current_score = new_score
                classes[i] = [m, r_new]
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [list(c) for c in classes]
            else:
                # Revert
                coverage[idx_new] -= 1
                coverage[idx_old] += 1
                
        elif move_type == 'add':
            available = [m for m in divisors if m not in used_moduli]
            if not available: continue
            m = random.choice(available)
            r = random.randint(0, m - 1)
            
            idx = np.arange(r, base, m)
            coverage[idx] += 1
            new_score = np.sum(coverage == 0)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                classes.append([m, r])
                used_moduli.add(m)
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [list(c) for c in classes]
            else:
                coverage[idx] -= 1
                
        elif move_type == 'remove' and len(classes) > 1:
            i = random.randint(0, len(classes) - 1)
            m, r = classes[i]
            idx = np.arange(r, base, m)
            
            # Check if removal is safe (doesn't increase uncovered)
            # Actually, SA allows it.
            coverage[idx] -= 1
            new_score = np.sum(coverage == 0)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                classes.pop(i)
                used_moduli.remove(m)
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [list(c) for c in classes]
            else:
                coverage[idx] += 1
                
        T *= cooling
        
    return best_classes
