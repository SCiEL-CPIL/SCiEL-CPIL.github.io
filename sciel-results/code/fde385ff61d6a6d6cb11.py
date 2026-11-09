import numpy as np
import time
import random
from itertools import combinations

def _compute_max_rep(A):
    """Compute the maximum number of unordered pair representations for sums in [128, 256]."""
    A_set = set(A)
    # Count representations for each target sum in [128, 256]
    counts = [0] * 129  # index 0 corresponds to sum 128, index 128 to sum 256
    
    # For each pair (a, b) with a <= b, if a + b in [128, 256], increment count
    for a in A:
        for b in A:
            if a <= b:
                s = a + b
                if 128 <= s <= 256:
                    counts[s - 128] += 1
    
    return max(counts) if counts else 0

def _is_valid(A):
    """Check if every integer in [128, 256] is a sum of two elements of A."""
    A_set = set(A)
    for target in range(128, 257):
        found = False
        for a in A:
            if a <= target and (target - a) in A_set and (target - a) >= a:
                found = True
                break
            elif (target - a) in A_set:
                found = True
                break
        if not found:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = 128
    # Candidate universe: [0, 128]
    universe = list(range(n + 1))
    
    # Initial heuristic: start with a symmetric set that's likely to cover the range
    # Try sets of various sizes and structures
    
    best_A = None
    best_score = float('inf')
    
    # Strategy: Use simulated annealing or local search
    # State: boolean array of length 129, x[i] = 1 if i in A
    
    def random_initial_set(min_size=30, max_size=60):
        size = random.randint(min_size, max_size)
        return set(random.sample(universe, size))
    
    def local_search(A, time_limit):
        """Improve A using local search (add/remove/swap elements)."""
        A = set(A)
        A_list = list(A)
        
        if not A:
            return A, float('inf')
        
        current_score = _compute_max_rep(A)
        if not _is_valid(A):
            # Penalize invalid sets
            current_score += 1000
            return A, current_score
        
        end_time = time.time() + time_limit
        no_improvement_count = 0
        
        while time.time() < end_time:
            improved = False
            candidates = list(A)
            
            # Try removing elements
            for a in candidates:
                if len(A) <= 1:
                    break
                A.remove(a)
                new_score = _compute_max_rep(A)
                if not _is_valid(A):
                    new_score += 1000
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    no_improvement_count = 0
                else:
                    A.add(a)
            
            # Try adding elements
            if not improved:
                for a in universe:
                    if a not in A:
                        A.add(a)
                        new_score = _compute_max_rep(A)
                        if not _is_valid(A):
                            new_score += 1000
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            no_improvement_count = 0
                        else:
                            A.remove(a)
            
            if not improved:
                no_improvement_count += 1
                if no_improvement_count > 5:
                    break
            else:
                no_improvement_count = 0
        
        return A, current_score
    
    # Multiple restarts with different initial sets
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart % 3 == 0:
            # Symmetric-ish initial set
            size = random.randint(35, 55)
            A = random_initial_set(size, size)
        elif restart % 3 == 1:
            # Dense lower half
            A = set(random.sample(range(0, 70), random.randint(30, 45)))
            A.update(random.sample(range(70, 129), random.randint(15, 25)))
        else:
            # Sparse
            A = random_initial_set(25, 40)
        
        # Local search with decreasing time budget
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        time_budget = min(remaining * 0.4, 3.0)
        
        A_improved, score = local_search(A, time_budget)
        
        if _is_valid(A_improved):
            actual_score = _compute_max_rep(A_improved)
            if actual_score < best_score:
                best_score = actual_score
                best_A = A_improved
        else:
            # Keep trying even if invalid, but track best valid
            pass
    
    # If no valid set found, try a deterministic fallback
    if best_A is None or not _is_valid(best_A):
        # Fallback: the original seed
        best_A = list(range(64, 129))
    
    return list(best_A)
