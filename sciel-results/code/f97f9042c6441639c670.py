import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 58.0  # Safety margin for 60s limit

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_total = 0
    best_A = []
    best_B = []
    
    # We will use random restarts with different seeds/orderings
    # and a robust local search.
    
    for seed in range(100):
        if time.time() > deadline - 2.0:
            break
            
        random.seed(seed)
        
        # Greedy construction
        A = []
        B = []
        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0
        
        # Shuffle order to escape bad local minima
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        for x in order:
            if time.time() > deadline - 2.0:
                break
            
            # Try adding to A
            can_add_A = True
            # Check Sidon property for A
            # Sums: x+x and x+a for all a in A
            if (sums_A >> (x + x)) & 1:
                can_add_A = False
            else:
                for a in A:
                    if (sums_A >> (x + a)) & 1:
                        can_add_A = False
                        break
                    d = abs(x - a)
                    if (diffs_A >> d) & 1 or (diffs_B >> d) & 1:
                        can_add_A = False
                        break
            
            if can_add_A:
                # Add to A
                A.append(x)
                sums_A |= (1 << (x + x))
                for a in A[:-1]:
                    sums_A |= (1 << (x + a))
                    d = abs(x - a)
                    diffs_A |= (1 << d)
            else:
                # Try adding to B
                can_add_B = True
                if (sums_B >> (x + x)) & 1:
                    can_add_B = False
                else:
                    for b in B:
                        if (sums_B >> (x + b)) & 1:
                            can_add_B = False
                            break
                        d = abs(x - b)
                        if (diffs_B >> d) & 1 or (diffs_A >> d) & 1:
                            can_add_B = False
                            break
                
                if can_add_B:
                    B.append(x)
                    sums_B |= (1 << (x + x))
                    for b in B[:-1]:
                        sums_B |= (1 << (x + b))
                        d = abs(x - b)
                        diffs_B |= (1 << d)
        
        total = len(A) + len(B)
        
        # Local Search
        # We try to improve the set by swapping elements.
        # We represent the state as lists and bitmasks.
        # To speed up local search, we might just do a few iterations.
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            
            # Rebuild bitmasks for current A, B to ensure consistency
            # (Actually, we maintained them above, but if we do swaps, we need to update)
            # For simplicity and speed, let's define helper functions to rebuild.
            
            def rebuild(A, B):
                sA = 0; dA = 0
                for i in range(len(A)):
                    for j in range(i, len(A)):
                        sA |= 1 << (A[i] + A[j])
                        if i != j:
                            dA |= 1 << abs(A[i] - A[j])
                sB = 0; dB = 0
                for i in range(len(B)):
                    for j in range(i, len(B)):
                        sB |= 1 << (B[i] + B[j])
                        if i != j:
                            dB |= 1 << abs(B[i] - B[j])
                return sA, dA, sB, dB

            sums_A, diffs_A, sums_B, diffs_B = rebuild(A, B)
            
            # Try 1-for-1 swaps
            # Remove x from A, add y to A
            for i in range(len(A)):
                if time.time() > deadline - 1.0: break
                x = A[i]
                # Temporarily remove x
                A_temp = A[:i] + A[i+1:]
                sA_temp, dA_temp, _, _ = rebuild(A_temp, B)
                
                # Find a y not in A_temp or B that can be added to A_temp
                for y in range(1, N + 1):
                    if y in A_temp or y in B:
                        continue
                    # Check if y can be added to A_temp
                    ok = True
                    if (sA_temp >> (y + y)) & 1: ok = False
                    else:
                        for a in A_temp:
                            if (sA_temp >> (y + a)) & 1: ok = False; break
                            d = abs(y - a)
                            if (dA_temp >> d) & 1 or (diffs_B >> d) & 1: ok = False; break
                    
                    if ok:
                        # Perform swap
                        A_temp.append(y)
                        # Verify global validity (diffs disjoint)
                        # dA_temp is diffs of A_temp. diffs_B is diffs of B.
                        # We need dA_temp & diffs_B == 0
                        if (dA_temp & diffs_B) == 0:
                            A = A_temp
                            # Update bitmasks
                            sums_A, diffs_A, sums_B, diffs_B = rebuild(A, B)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Try swaps in B
                for i in range(len(B)):
                    if time.time() > deadline - 1.0: break
                    x = B[i]
                    B_temp = B[:i] + B[i+1:]
                    _, _, sB_temp, dB_temp = rebuild(A, B_temp)
                    
                    for y in range(1, N + 1):
                        if y in A or y in B_temp:
                            continue
                        ok = True
                        if (sB_temp >> (y + y)) & 1: ok = False
                        else:
                            for b in B_temp:
                                if (sB_temp >> (y + b)) & 1: ok = False; break
                                d = abs(y - b)
                                if (dB_temp >> d) & 1 or (diffs_A >> d) & 1: ok = False; break
                        
                        if ok:
                            B_temp.append(y)
                            if (dB_temp & diffs_A) == 0:
                                B = B_temp
                                sums_A, diffs_A, sums_B, diffs_B = rebuild(A, B)
                                improved = True
                                break
                    if improved:
                        break
        
        if len(A) + len(B) > best_total:
            best_total = len(A) + len(B)
            best_A = A[:]
            best_B = B[:]
            
    # Ensure non-empty
    if not best_A: best_A = [1]
    if not best_B: best_B = [2]
    
    return [best_A, best_B]
