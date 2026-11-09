import time
import numpy as np
import random

def _get_divisors(n):
    divisors = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted([d for d in divisors if d > 1])

def _score_classes_fast(classes, base=51975):
    """Fast scoring using numpy."""
    covered = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return base - int(covered.sum())

def _greedy_construct(base=51975, order='ascending'):
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    if order == 'descending':
        divisors = divisors[::-1]
    elif order == 'random':
        random.shuffle(divisors)
    
    for m in divisors:
        # Find residue that covers most uncovered residues
        best_r = 0
        best_count = -1
        for r in range(m):
            count = int((covered[r::m] == 0).sum())
            if count > best_count:
                best_count = count
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    return classes

def _local_search(classes, base=51975, time_limit=10, max_iterations=1000):
    """
    Efficient local search using cover counts.
    """
    start_time = time.time()
    n = len(classes)
    
    # Build initial cover counts
    cover_count = np.zeros(base, dtype=np.int32)
    for m, r in classes:
        cover_count[r::m] += 1
    
    current_score = int(np.sum(cover_count == 0))
    best_score = current_score
    best_classes = [c[:] for c in classes]
    
    improved = True
    iteration = 0
    while improved and time.time() - start_time < time_limit and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Shuffle order of classes to break symmetry
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time > time_limit:
                break
            
            m_i, r_i = classes[i]
            
            # Try changing residue for class i
            for new_r in range(m_i):
                if new_r == r_i:
                    continue
                
                # 1. Remove old residue coverage
                old_residues = r_i + np.arange(0, base, m_i)
                cover_count[old_residues] -= 1
                newly_uncovered = int(np.sum(cover_count[old_residues] == 0))
                
                # 2. Add new residue coverage
                new_residues = new_r + np.arange(0, base, m_i)
                cover_count[new_residues] += 1
                newly_covered = int(np.sum(cover_count[new_residues] == 1))
                
                delta_score = newly_uncovered - newly_covered
                new_score = current_score + delta_score
                
                if new_score < best_score:
                    best_score = new_score
                    best_classes = [c[:] for c in classes]
                    current_score = new_score
                    classes[i][1] = new_r
                    improved = True
                else:
                    # Revert
                    cover_count[old_residues] += 1
                    cover_count[new_residues] -= 1
                    
    return best_classes

def _multi_start_local_search(base=51975, time_limit=60, num_starts=5):
    """Try multiple greedy orderings and local search on each."""
    start_time = time.time()
    best_classes = None
    best_score = float('inf')
    
    orders = ['ascending', 'descending', 'random', 'random', 'random']
    for idx, order in enumerate(orders):
        if time.time() - start_time > time_limit:
            break
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5:
            break
        
        # Allocate time for this start
        ls_time = min(remaining_time * 0.7, 15)
        
        classes = _greedy_construct(base, order=order)
        classes = _local_search(classes, base, time_limit=ls_time)
        score = _score_classes_fast(classes, base)
        
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
    
    return best_classes

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145
    
    # Use multi-start local search with different greedy orderings
    classes = _multi_start_local_search(base, time_limit=min(100, int(deadline - time.time()) - 10))
    
    # Final polish with remaining time
    if time.time() < deadline - 5 and classes is not None:
        remaining = int(deadline - time.time()) - 2
        if remaining > 3:
            classes = _local_search(classes, base, time_limit=min(remaining, 20))
    
    return classes
