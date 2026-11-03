import numpy as np
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill

    nm = n * m  # 6
    mp = m * p  # 27
    pn = p * n  # 18

    # Build the target tensor as a set of (a,b,c) with value 1
    # a = i*m + j, b = j*p + k, c = k*n + i
    target_entries = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries[(a, b, c)] = 1

    target_set = set(target_entries.keys())

    # Schoolbook decomposition
    rank_sb = n * m * p  # 54
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    def verify_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose the tensor."""
        r = len(U[0])

        # Check all target entries
        for (a, b, c), val in target_entries.items():
            s = 0.0
            for t in range(r):
                s += U[a][t] * V[b][t] * W[c][t]
            if abs(s - val) > tol:
                return False

        # Check that non-target entries are zero
        # Total entries: nm * mp * pn = 6 * 27 * 18 = 2916
        # Target entries: 54
        # Non-target: 2862
        if r <= 40:
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        if (a, b, c) not in target_set:
                            s = 0.0
                            for t in range(r):
                                s += U[a][t] * V[b][t] * W[c][t]
                            if abs(s) > tol:
                                return False
        else:
            # For larger r, sample non-target entries
            non_target = [(a, b, c) for a in range(nm) for b in range(mp) for c in range(pn)
                          if (a, b, c) not in target_set]
            random.shuffle(non_target)
            sample_size = min(500, len(non_target))
            for (a, b, c) in non_target[:sample_size]:
                s = 0.0
                for t in range(r):
                    s += U[a][t] * V[b][t] * W[c][t]
                if abs(s) > tol:
                    return False

        return True

    # Strategy: Model as 2x3 matrix multiplication over 9-dim algebra of 3x3 diagonal matrices
    # This gives a rank-18 decomposition by tensoring the optimal rank-6 decomposition of 2x3x3
    # with the standard rank-9 basis of the diagonal algebra.
    
    # First, let's try to find a rank-6 decomposition for 2x3x3 matrix multiplication
    # Then tensor it with the diagonal algebra basis
    
    # For 2x3x3: n=2, m=3, p=3, schoolbook rank = 18
    # Let's try to find a lower rank decomposition for 2x3x3
    
    # Actually, let's think about this differently.
    # The 2x3x9 multiplication can be seen as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For each j, we have a 2x1 times 1x9 multiplication
    # This is an outer product of A[:,j] (length 2) and B[j,:] (length 9)
    
    # The key insight from the strategy: model as 2x3 matrix multiplication over 9-dim algebra
    # of 3x3 diagonal matrices. The diagonal algebra has dimension 9 (3 diagonal entries, each with 3 positions? No, 3x3 diagonal has dimension 3)
    
    # Wait, 3x3 diagonal matrices form a 3-dimensional algebra, not 9-dimensional.
    # Let me reconsider. The strategy says "9-dimensional algebra of 3x3 diagonal matrices" - that's incorrect.
    # 3x3 diagonal matrices form a 3-dimensional commutative algebra.
    
    # Let me try a different approach. Let's use the fact that we can decompose the problem
    # by exploiting the structure more carefully.
    
    # For the 2x3x9 case, let's try ranks from 18 up to 54 and see if we can find a valid decomposition
    # using a structured approach.
    
    # Approach: For each j in {0,1,2}, the contribution is A[:,j] * B[j,:]
    # This is a rank-2 tensor in the (i,k) space for each j
    # But we need to decompose the full multiplication tensor, not just C
    
    # Let's try a more systematic search for lower ranks using random initialization + gradient descent
    # on the tensor reconstruction error.
    
    def try_rank(r, num_attempts=10, time_budget=10):
        """Try to find a rank-r decomposition using random search."""
        start = time.time()
        
        for attempt in range(num_attempts):
            if time.time() > start + time_budget or time.time() > deadline - 20:
                break
            
            # Random initialization with small values
            U = np.random.uniform(-1, 1, (nm, r))
            V = np.random.uniform(-1, 1, (mp, r))
            W = np.random.uniform(-1, 1, (pn, r))
            
            # Compute the tensor reconstruction error
            # T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
            # We want this to equal 1 for target entries and 0 otherwise
            
            # Use gradient descent to minimize the error
            lr = 0.01
            for step in range(100):
                if time.time() > start + time_budget or time.time() > deadline - 20:
                    break
                
                # Compute gradient w.r.t. U, V, W
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # For target entries, we want T[a,b,c] = 1
                for (a, b, c), val in target_entries.items():
                    # dT/da = V[b,t]*W[c,t], dT/db = U[a,t]*W[c,t], dT/dc = U[a,t]*V[b,t]
                    # Error = T - val, so gradient of (T-val)^2 is 2*(T-val)*dT/dparam
                    T = np.sum(U[a] * V[b] * W[c])
                    err = T - val
                    grad_U[a] += 2 * err * V[b] * W[c]
                    grad_V[b] += 2 * err * U[a] * W[c]
                    grad_W[c] += 2 * err * U[a] * V[b]
                
                # For non-target entries, we want T[a,b,c] = 0
                # Sample some non-target entries to keep computation manageable
                non_target_sample = 100
                for _ in range(non_target_sample):
                    a = random.randint(0, nm - 1)
                    b = random.randint(0, mp - 1)
                    c = random.randint(0, pn - 1)
                    if (a, b, c) not in target_set:
                        T = np.sum(U[a] * V[b] * W[c])
                        err = T - 0
                        grad_U[a] += 2 * err * V[b] * W[c]
                        grad_V[b] += 2 * err * U[a] * W[c]
                        grad_W[c] += 2 * err * U[a] * V[b]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            # Check if we found a valid decomposition
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify_decomposition(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try ranks from 18 to 53
    for r in range(18, 54):
        if time.time() > deadline - 25:
            break
        
        result = try_rank(r, num_attempts=5, time_budget=15)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    # If we didn't find anything better, return schoolbook
    return best_result
