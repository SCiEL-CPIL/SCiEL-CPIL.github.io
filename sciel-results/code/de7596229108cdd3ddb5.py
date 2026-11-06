import time
import random

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    best_size = 0
    best_A = []
    best_B = []
    
    candidates = list(range(1, 257))
    
    while time.time() - start_time < timeout:
        random.shuffle(candidates)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in candidates:
            # Check if x can be added to A
            valid_A = True
            # Check sums: x+a for a in A, and 2x
            # Optimization: Check if any (x+a) is in sums_A
            for a in A:
                if (x + a) in sums_A:
                    valid_A = False
                    break
            if valid_A and (2 * x) in sums_A:
                valid_A = False
            
            # Check diffs: |x-a| for a in A must not be in diffs_B
            if valid_A:
                for a in A:
                    d = abs(x - a)
                    if d in diffs_B:
                        valid_A = False
                        break
            
            # Check if x can be added to B
            valid_B = True
            for b in B:
                if (x + b) in sums_B:
                    valid_B = False
                    break
            if valid_B and (2 * x) in sums_B:
                valid_B = False
            
            if valid_B:
                for b in B:
                    d = abs(x - b)
                    if d in diffs_A:
                        valid_B = False
                        break
            
            # Decision logic
            if valid_A and valid_B:
                # Balance sizes: add to the smaller set
                if len(A) <= len(B):
                    # Add to A
                    A.append(x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                        diffs_A.add(abs(x - a))
                    sums_A.add(2 * x)
                else:
                    # Add to B
                    B.append(x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                        diffs_B.add(abs(x - b))
                    sums_B.add(2 * x)
            elif valid_A:
                # Add to A
                A.append(x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                    diffs_A.add(abs(x - a))
                sums_A.add(2 * x)
            elif valid_B:
                # Add to B
                B.append(x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                    diffs_B.add(abs(x - b))
                sums_B.add(2 * x)
            # Else: cannot add to either
            
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = list(A)
            best_B = list(B)

    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
