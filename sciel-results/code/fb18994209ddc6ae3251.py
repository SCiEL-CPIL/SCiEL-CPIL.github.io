import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def is_valid_paley(p):
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        out_masks = [0] * p
        for v in range(p):
            mask = 0
            for u in range(p):
                if v == u:
                    continue
                diff = (u - v) % p
                if diff in squares:
                    mask |= (1 << u)
            out_masks[v] = mask
            
        # Check all 4-sets
        for S in combinations(range(p), 4):
            s_mask = 0
            for v in S:
                s_mask |= (1 << v)
            
            found = False
            for v in range(p):
                if (out_masks[v] & s_mask) == s_mask:
                    found = True
                    break
            if not found:
                return False
        return True

    def build_paley_orientation(p):
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return orientation

    def validate_orientation(n, orientation):
        # Reconstruct adjacency matrix
        A = np.zeros((n, n), dtype=bool)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orientation[idx] == 0:
                    A[i, j] = True
                    A[j, i] = False
                else:
                    A[i, j] = False
                    A[j, i] = True
                idx += 1
        
        # Check all 4-sets
        # For each 4-set, check if any vertex dominates it
        # Dominated by v if A[v, u] is True for all u in S
        for S in combinations(range(n), 4):
            dominated = False
            for v in range(n):
                if all(A[v, u] for u in S):
                    dominated = True
                    break
            if not dominated:
                return False
        return True

    best_n = 67
    best_orientation = build_paley_orientation(67)
    
    # Check smaller Paley candidates
    candidates = [31, 43, 47, 59]
    for p in candidates:
        if time.time() > deadline:
            break
        if is_valid_paley(p):
            best_n = p
            best_orientation = build_paley_orientation(p)
            break

    # Try to find a non-Paley solution for n < 67?
    # Given time constraints and complexity, we skip this to ensure reliability.
    # If we have time, we could try a few random tournaments for n=66.
    # But validation is O(n^5). For n=66, it's slow.
    # We will stick to the reliable Paley 67.

    # Final validation of the chosen orientation
    if time.time() < deadline - 5:
        if not validate_orientation(best_n, best_orientation):
            # Fallback to 67 if validation fails (should not happen)
            best_n = 67
            best_orientation = build_paley_orientation(67)

    return {'n': best_n, 'orientation': best_orientation}
