"""Rewritten program: dual-view construction via Sidon-like difference sets."""
import numpy as np
from itertools import combinations
import math

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0 to maximize
    the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1).
    
    Strategy: Use a "dual-view" approach where we define U by its difference structure.
    We optimize by selecting elements that minimize |U+U| (additive energy) while
    keeping |U-U| large (distinct differences).
    """
    
    def compute_score(U):
        if not U or 0 not in U:
            return 0.0
        U_set = set(U)
        max_u = max(U_set)
        if max_u == 0:
            return 0.0
        # Compute |U+U| and |U-U|
        sum_set = set()
        diff_set = set()
        for a in U_set:
            for b in U_set:
                sum_set.add(a + b)
                diff_set.add(a - b)
        n_diff = len(diff_set)
        n_sum = len(sum_set)
        if n_sum == 0:
            return 0.0
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        return 1.0 + math.log(n_diff / n_sum) / log_denom
    
    def greedy_construct(max_size=15, max_val=200):
        """
        Greedily construct U by adding elements that maximize score improvement.
        Start with {0} and at each step add the element that gives best marginal gain.
        """
        U = [0]
        U_set = {0}
        
        # Precompute current sums and diffs
        sum_set = {0}
        diff_set = {0}
        
        for _ in range(max_size - 1):
            best_elem = None
            best_score = -float('inf')
            
            # Try candidate elements
            for candidate in range(1, max_val + 1):
                if candidate in U_set:
                    continue
                
                # Compute new sums and diffs
                new_sum_set = set(sum_set)
                new_diff_set = set(diff_set)
                
                for a in U_set:
                    new_sum_set.add(a + candidate)
                    new_diff_set.add(a - candidate)
                    new_diff_set.add(candidate - a)
                
                # Compute score
                max_u = max(max(U_set), candidate)
                n_diff = len(new_diff_set)
                n_sum = len(new_sum_set)
                
                if n_sum == 0 or max_u == 0:
                    continue
                
                log_denom = math.log(2 * max_u + 1)
                if log_denom <= 0:
                    continue
                
                score = 1.0 + math.log(n_diff / n_sum) / log_denom
                
                if score > best_score:
                    best_score = score
                    best_elem = candidate
            
            if best_elem is None:
                break
            
            # Add the best element
            U.append(best_elem)
            U_set.add(best_elem)
            
            # Update sums and diffs
            for a in list(U_set):
                sum_set.add(a + best_elem)
                diff_set.add(a - best_elem)
                diff_set.add(best_elem - a)
        
        return sorted(U)
    
    # Try different parameter combinations
    candidates = []
    
    # Various greedy constructions with different max_size and max_val
    for max_size in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        for max_val in [50, 100, 150, 200, 300, 500]:
            U = greedy_construct(max_size, max_val)
            candidates.append(U)
    
    # Also include some known good structures
    # Sidon-like sequences
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86, 100, 115])
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86])
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61])
    
    # Evaluate all candidates
    best_score = 0.0
    best_U = [0, 1, 3]
    
    for U in candidates:
        if not U or 0 not in U:
            continue
        score = compute_score(U)
        if score > best_score:
            best_score = score
            best_U = U
    
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
