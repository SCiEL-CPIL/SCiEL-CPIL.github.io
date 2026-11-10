"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    def _check_sidon_and_disjoint(A, B, a, b):
        """Check if adding a to A and b to B maintains Sidon property and disjoint differences."""
        # Check Sidon for A with a added
        if a in A:
            return False  # a already in A
        # Check Sidon for B with b added
        if b in B:
            return False
        
        # Check sums for A
        A_new = A + [a]
        sums_A = set()
        for i in range(len(A_new)):
            for j in range(i, len(A_new)):
                s = A_new[i] + A_new[j]
                if s in sums_A:
                    return False
                sums_A.add(s)
        
        # Check sums for B
        B_new = B + [b]
        sums_B = set()
        for i in range(len(B_new)):
            for j in range(i, len(B_new)):
                s = B_new[i] + B_new[j]
                if s in sums_B:
                    return False
                sums_B.add(s)
        
        # Check differences
        diff_A = set()
        for i in range(len(A_new)):
            for j in range(len(A_new)):
                if i != j:
                    d = A_new[i] - A_new[j]
                    if d != 0:
                        diff_A.add(d)
        
        diff_B = set()
        for i in range(len(B_new)):
            for j in range(len(B_new)):
                if i != j:
                    d = B_new[i] - B_new[j]
                    if d != 0:
                        diff_B.add(d)
        
        # Disjoint nonzero differences
        if diff_A & diff_B:
            return False
        
        return True
    
    def _greedy_random():
        """Randomized greedy construction."""
        A = []
        B = []
        used_sums_A = set()
        used_sums_B = set()
        diff_A = set()
        diff_B = set()
        
        # Randomize order of candidates
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            # Try adding to A
            if x not in A:
                # Check Sidon for A
                valid_A = True
                for a in A:
                    s1 = x + x
                    s2 = x + a
                    if s1 in used_sums_A or s2 in used_sums_A:
                        valid_A = False
                        break
                if valid_A:
                    # Check differences
                    new_diffs = set()
                    for a in A:
                        d1 = x - a
                        d2 = a - x
                        new_diffs.add(d1)
                        new_diffs.add(d2)
                    new_diffs.add(0)
                    new_diffs.discard(0)
                    
                    if not (new_diffs & diff_B):
                        # Add to A
                        A.append(x)
                        used_sums_A.add(x + x)
                        for a in A[:-1]:
                            used_sums_A.add(x + a)
                        diff_A |= new_diffs
                        continue
            
            # Try adding to B
            if x not in B:
                # Check Sidon for B
                valid_B = True
                for b in B:
                    s1 = x + x
                    s2 = x + b
                    if s1 in used_sums_B or s2 in used_sums_B:
                        valid_B = False
                        break
                if valid_B:
                    # Check differences
                    new_diffs = set()
                    for b in B:
                        d1 = x - b
                        d2 = b - x
                        new_diffs.add(d1)
                        new_diffs.add(d2)
                    new_diffs.add(0)
                    new_diffs.discard(0)
                    
                    if not (new_diffs & diff_A):
                        # Add to B
                        B.append(x)
                        used_sums_B.add(x + x)
                        for b in B[:-1]:
                            used_sums_B.add(x + b)
                        diff_B |= new_diffs
        
        return A, B
    
    best_A = []
    best_B = []
    best_size = 0
    
    # Run multiple random restarts
    for restart in range(1000):
        if time.time() > deadline:
            break
        
        A, B = _greedy_random()
        size = len(A) + len(B)
        
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
