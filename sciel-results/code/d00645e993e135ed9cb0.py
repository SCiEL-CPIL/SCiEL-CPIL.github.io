"""Improved program for the suite task; EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = 256
    
    def check_pair_conflict(a, b, c, selected):
        """Check if a divides b+c (for a < b <= c)."""
        return (b + c) % a == 0
    
    def is_valid_set(S):
        """Check if set S satisfies the property."""
        S_sorted = sorted(S)
        for i, a in enumerate(S_sorted):
            for j in range(i + 1, len(S_sorted)):
                b = S_sorted[j]
                for k in range(j, len(S_sorted)):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def count_conflicts(S, candidate):
        """Count how many conflicts candidate creates with S."""
        S_sorted = sorted(S)
        conflicts = 0
        for a in S_sorted:
            if a >= candidate:
                continue
            # Check pairs (a, b, c) where a < b <= c, and one of b,c is candidate
            # Case 1: b = candidate, c in S_sorted with c >= candidate
            for c in S_sorted:
                if c < candidate:
                    continue
                if (candidate + c) % a == 0:
                    conflicts += 1
            # Case 2: c = candidate, b in S_sorted with candidate <= b <= candidate (only b=candidate)
            # This is covered by case 1 when c = candidate
            # But also need to check where a < b <= c and c = candidate, b in S
            # Actually let me redo this more carefully
        # Let me just do a full check for simplicity
        S_new = S + [candidate]
        S_sorted = sorted(S_new)
        for i, a in enumerate(S_sorted):
            for j in range(i + 1, len(S_sorted)):
                b = S_sorted[j]
                for k in range(j, len(S_sorted)):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return True
        return False
    
    # Strategy: Start with the known good tail [171, 256]
    S = list(range(171, 257))
    best_set = set(S)
    best_size = len(S)
    
    # Try to add elements from below 171
    for candidate in range(1, 171):
        if time.time() > deadline:
            break
        
        S_new = S + [candidate]
        S_sorted = sorted(S_new)
        valid = True
        for i, a in enumerate(S_sorted):
            if not valid:
                break
            for j in range(i + 1, len(S_sorted)):
                if not valid:
                    break
                b = S_sorted[j]
                for k in range(j, len(S_sorted)):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        valid = False
                        break
        
        if valid:
            S = S_new
            best_set = set(S)
            best_size = len(S)
    
    # Try swaps: remove one element from S, add one from outside
    if time.time() < deadline - 5:
        for remove_candidate in range(171, 257):
            if time.time() > deadline:
                break
            
            S_without = [x for x in S if x != remove_candidate]
            for add_candidate in range(1, 171):
                if time.time() > deadline:
                    break
                if add_candidate in S_without:
                    continue
                
                S_new = S_without + [add_candidate]
                S_sorted = sorted(S_new)
                valid = True
                for i, a in enumerate(S_sorted):
                    if not valid:
                        break
                    for j in range(i + 1, len(S_sorted)):
                        if not valid:
                            break
                        b = S_sorted[j]
                        for k in range(j, len(S_sorted)):
                            c = S_sorted[k]
                            if (b + c) % a == 0:
                                valid = False
                                break
                
                if valid and len(S_new) > best_size:
                    S = S_new
                    best_set = set(S)
                    best_size = len(S)
    
    return sorted(list(best_set))
# EVOLVE-BLOCK-END
