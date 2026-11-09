import numpy as np
import sys
import time
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = start_time + 110.0

    # 1. Construct Schoolbook 3x3x8 (Rank 72)
    U_school = np.zeros((9, 72))
    V_school = np.zeros((24, 72))
    W_school = np.zeros((24, 72))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_rank = 72
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # 2. Try to find a better 3x3x3 decomposition (Rank < 27)
    # If we find rank R for 3x3x3, we can build 3x3x8 with rank 2*R + 18 (using schoolbook for 3x3x2)
    # Target: R <= 26 -> Rank 70
    # R <= 25 -> Rank 68
    # R <= 23 -> Rank 64

    def find_333_decomposition(target_rank):
        """
        Attempts to find a rank-`target_rank` decomposition for 3x3x3 matrix multiplication.
        Returns (U, V, W) or None.
        """
        if time.time() > DEADLINE - 20:
            return None
            
        # Initialize with schoolbook (rank 27) and try to merge/remove
        # We will use a random restart strategy with local optimization
        
        for attempt in range(50):
            if time.time() > DEADLINE - 20:
                break
                
            np.random.seed(attempt)
            
            # Start with random initialization scaled to be close to schoolbook
            # Or just random
            U = np.random.randn(9, target_rank) * 0.5
            V = np.random.randn(9, target_rank) * 0.5
            W = np.random.randn(9, target_rank) * 0.5
            
            # Define the tensor T for 3x3x3
            # T[i*m+j, j*p+k, k*n+i] = 1
            # Here n=3, m=3, p=3
            
            def error_func(params):
                # params is flattened U, V, W
                U_f = params[:9*target_rank].reshape(9, target_rank)
                V_f = params[9*target_rank:18*target_rank].reshape(9, target_rank)
                W_f = params[18*target_rank:].reshape(9, target_rank)
                
                err = 0.0
                # Check non-zeros
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            a = i*3 + j
                            b = j*3 + k
                            c = k*3 + i
                            val = np.dot(U_f[a], V_f[b] * W_f[c])
                            err += (val - 1.0)**2
                # Check zeros (sample)
                # There are 27 non-zeros out of 9*9*9=729.
                # We check a subset of zeros to keep it fast
                np.random.seed(123) # Fixed seed for consistency within error calc? No, use global
                for _ in range(50):
                    a = np.random.randint(0, 9)
                    b = np.random.randint(0, 9)
                    c = np.random.randint(0, 9)
                    i = a // 3
                    j = a % 3
                    jj = b // 3
                    kk = b % 3
                    ckk = c // 3
                    ii = c % 3
                    if (i, j, kk) != (i, j, kk): # This logic is messy, let's just check all zeros if fast enough
                        pass
                    # Better: precompute zero indices
                    # But for speed in loop, we skip zero check or do it rarely
                return err

            # Use scipy.optimize if available, else simple gradient descent
            try:
                from scipy.optimize import minimize
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                res = minimize(error_func, x0, method='BFGS', options={'maxiter': 100})
                if res.fun < 1e-8:
                    U_f = res.x[:9*target_rank].reshape(9, target_rank)
                    V_f = res.x[9*target_rank:18*target_rank].reshape(9, target_rank)
                    W_f = res.x[18*target_rank:].reshape(9, target_rank)
                    # Snap to nearest integer/half-integer
                    U_s = np.round(U_f * 2) / 2
                    V_s = np.round(V_f * 2) / 2
                    W_s = np.round(W_f * 2) / 2
                    # Verify
                    if verify_333(U_s, V_s, W_s):
                        return U_s, V_s, W_s
            except:
                pass
                
        return None

    def verify_333(U, V, W, tol=1e-9):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i*3 + j
                    b = j*3 + k
                    c = k*3 + i
                    if abs(np.dot(U[a], V[b] * W[c]) - 1.0) > tol:
                        return False
        # Check zeros
        for a in range(9):
            for b in range(9):
                for c in range(9):
                    i = a // 3
                    j = a % 3
                    jj = b // 3
                    kk = b % 3
                    ckk = c // 3
                    ii = c % 3
                    # Zero if not (i==ii and j==jj and k==kk) ??
                    # T[a,b,c] is 1 if a=i*m+j, b=j*p+k, c=k*n+i
                    # So a=i*3+j, b=j*3+k, c=k*3+i
                    # Check if (a,b,c) is in target
                    if not (a == i*3+j and b == j*3+k and c == k*3+i):
                         # This is always true by construction of i,j,k from a,b,c?
                         # No. a,b,c are indices.
                         # i,j from a. jj,kk from b. ckk,ii from c.
                         # Target is 1 if i==ii and j==jj and k==kk.
                         pass
        # Simpler check:
        for a in range(9):
            for b in range(9):
                for c in range(9):
                    val = np.dot(U[a], V[b] * W[c])
                    i = a // 3
                    j = a % 3
                    jj = b // 3
                    kk = b % 3
                    ckk = c // 3
                    ii = c % 3
                    if i == ii and j == jj and kk == ckk:
                        if abs(val - 1.0) > tol: return False
                    else:
                        if abs(val) > tol: return False
        return True

    # Try ranks 23, 24, 25, 26
    found_333 = None
    for r in [23, 24, 25, 26]:
        if time.time() > DEADLINE - 30:
            break
        res = find_333_decomposition(r)
        if res:
            found_333 = res
            break

    if found_333:
        U3, V3, W3 = found_333
        r3 = U3.shape[1]
        
        # Construct 3x3x8
        # Use two 3x3x3 decompositions for columns 0-5
        # Use schoolbook 3x3x2 for columns 6-7
        r_total = 2 * r3 + 18
        
        U_new = np.zeros((9, r_total))
        V_new = np.zeros((24, r_total))
        W_new = np.zeros((24, r_total))
        
        # Part 1: 3x3x3 for B[:, 0:3]
        # U, V, W from found_333
        # V indices: j*3+k. We need to map to j*8+k.
        # For k in 0..2, V[j*3+k] -> V_new[j*8+k]
        U_new[:, 0:r3] = U3
        for j in range(3):
            for k in range(3):
                V_new[j*8+k, 0:r3] = V3[j*3+k]
        for k in range(3):
            for i in range(3):
                W_new[k*3+i, 0:r3] = W3[k*3+i]
        
        # Part 2: 3x3x3 for B[:, 3:6]
        # Shift V indices by 3
        U_new[:, r3:2*r3] = U3
        for j in range(3):
            for k in range(3):
                V_new[j*8+(k+3), r3:2*r3] = V3[j*3+k]
        for k in range(3):
            for i in range(3):
                W_new[k*3+i, r3:2*r3] = W3[k*3+i]
        
        # Part 3: Schoolbook 3x3x2 for B[:, 6:8]
        # Rank 18
        col_offset = 2 * r3
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U_new[i*3+j, col_offset] = 1
                    V_new[j*8+(k+6), col_offset] = 1
                    W_new[k*3+i, col_offset] = 1
                    col_offset += 1
                    
        # Verify full 3x3x8
        if verify_full(U_new, V_new, W_new):
            best_rank = r_total
            best_U = U_new
            best_V = V_new
            best_W = W_new

    # Convert to list of lists
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    new_row.append(int(round(x)))
                else:
                    if abs(2*x - round(2*x)) < 1e-9:
                        new_row.append(round(2*x)/2.0)
                    else:
                        new_row.append(round(x, 6))
            result.append(new_row)
        return result

    U_result = to_list(best_U)
    V_result = to_list(best_V)
    W_result = to_list(best_W)

    return {"U": U_result, "V": V_result, "W": W_result}

def verify_full(U, V, W, tol=1e-9):
    n, m, p = 3, 3, 8
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > tol:
                    return False
    # Sample zeros
    np.random.seed(42)
    for _ in range(100):
        a = np.random.randint(0, n*m)
        b = np.random.randint(0, m*p)
        c = np.random.randint(0, p*n)
        i = a // m
        j = a % m
        jj = b // p
        kk = b % p
        ckk = c // n
        ii = c % n
        if not (i == ii and j == jj and kk == ckk):
            val = np.dot(U[a], V[b] * W[c])
            if abs(val) > tol:
                return False
    return True

# EVOLVE-BLOCK-END
