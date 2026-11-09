"""Matrix multiplication 2,3,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    # Build the target tensor T as a 3D array
    # T[x, y, z] = 1 if x = i*m+j, y = j*p+k, z = k*n+i for some i,j,k, else 0
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition
    def schoolbook():
        U = np.zeros((n*m, schoolbook_rank))
        V = np.zeros((m*p, schoolbook_rank))
        W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W
    
    # Verification function
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        try:
            T_recon = np.einsum('xr,yr,zr->xyz', U, V, W)
            if np.max(np.abs(T_recon - T)) < tol:
                return True
        except:
            pass
        return False
    
    # Convert to list of lists with integer/half-integer values
    def to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    clean_row.append(int(round(val)))
                elif abs(val*2 - round(val*2)) < 1e-9:
                    clean_row.append(round(val*2)/2.0)
                else:
                    clean_row.append(val)
            result.append(clean_row)
        return result
    
    if verify(U, V, W):
        best_U, best_V, best_W = U, V, W
        best_rank = schoolbook_rank
    
    # Strategy 1: Try to find rank-9 decomposition for 2x3 x 3x3, then use twice
    # For 2x3 x 3x3: A is 2x3, B is 3x3, C is 2x3
    # Schoolbook rank = 2*3*3 = 18
    # Try to find rank 9, 10, 11, ... for this subproblem
    
    def find_2x3x3_decomp(target_rank):
        """Try to find a decomposition of 2x3 x 3x3 with given rank."""
        n2, m2, p2 = 2, 3, 3
        T_sub = np.zeros((n2*m2, m2*p2, p2*n2), dtype=float)
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    T_sub[i*m2+j, j*p2+k, k*n2+i] = 1.0
        
        def verify_sub(U, V, W, tol=1e-9):
            U = np.array(U, dtype=float)
            V = np.array(V, dtype=float)
            W = np.array(W, dtype=float)
            try:
                T_recon = np.einsum('xr,yr,zr->xyz', U, V, W)
                if np.max(np.abs(T_recon - T_sub)) < tol:
                    return True
            except:
                pass
            return False
        
        # Try random search with structured initialization
        import random
        random.seed(42)
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            U_try = np.zeros((n2*m2, target_rank))
            V_try = np.zeros((m2*p2, target_rank))
            W_try = np.zeros((p2*n2, target_rank))
            
            # Use a structured approach: each rank-1 term should contribute
            # to a specific (i,j,k) pattern
            for r in range(target_rank):
                # Choose which (i,j,k) this term primarily handles
                # But allow some mixing
                if r < n2*m2*p2:
                    i = r // (m2*p2)
                    j = (r // p2) % m2
                    k = r % p2
                    U_try[i*m2+j, r] = 1
                    V_try[j*p2+k, r] = 1
                    W_try[k*n2+i, r] = 1
                else:
                    # Additional terms: try to cancel or combine
                    # Random sparse
                    for _ in range(3):
                        idx_u = random.randint(0, n2*m2-1)
                        U_try[idx_u, r] = random.choice([-1, 1])
                    for _ in range(3):
                        idx_v = random.randint(0, m2*p2-1)
                        V_try[idx_v, r] = random.choice([-1, 1])
                    for _ in range(3):
                        idx_w = random.randint(0, p2*n2-1)
                        W_try[idx_w, r] = random.choice([-1, 1])
            
            if verify_sub(U_try, V_try, W_try):
                return U_try, V_try, W_try
        
        return None
    
    # Try to find rank-9 for 2x3 x 3x3
    if time.time() < deadline - 20:
        for target_rank in [9, 10, 11, 12]:
            if time.time() > deadline - 15:
                break
            result = find_2x3x3_decomp(target_rank)
            if result is not None:
                U_sub, V_sub, W_sub = result
                # Now use this for both blocks of 3x6
                # Block 1: columns 0-2 of B, output columns 0-2
                # Block 2: columns 3-5 of B, output columns 3-5
                U_full = np.zeros((n*m, target_rank * 2))
                V_full = np.zeros((m*p, target_rank * 2))
                W_full = np.zeros((p*n, target_rank * 2))
                
                # Copy for block 1 (k=0,1,2)
                for r in range(target_rank):
                    U_full[:, r] = U_sub[:, r]
                    V_full[:, r] = V_sub[:, r]
                    W_full[:, r] = W_sub[:, r]
                
                # Copy for block 2 (k=3,4,5)
                # Need to shift indices: j*p+k where k is 0,1,2 -> j*p+3+k
                for r in range(target_rank):
                    U_full[:, target_rank + r] = U_sub[:, r]
                    # V: shift k index by 3
                    for row in range(m2*p2):
                        j = row // p2
                        k = row % p2
                        V_full[j*p + (k+3), target_rank + r] = V_sub[row, r]
                    # W: shift k index by 3 in output
                    for row in range(p2*n2):
                        k = row // n2
                        i = row % n2
                        W_full[(k+3)*n + i, target_rank + r] = W_sub[row, r]
                
                if verify(U_full, V_full, W_full):
                    best_rank = target_rank * 2
                    best_U, best_V, best_W = U_full, V_full, W_full
                    break
    
    # Strategy 2: Direct rank reduction from schoolbook
    # Try to find linear combinations that reduce rank
    if time.time() < deadline - 10:
        # For each pair of schoolbook columns, check if their sum/difference
        # can replace both
        # This is equivalent to checking if T has rank < 36
        
        # Try greedy: for each rank from 35 down, try to find a solution
        import random
        random.seed(123)
        
        for target_rank in range(35, max(17, best_rank - 5), -1):
            if time.time() > deadline - 5:
                break
            if target_rank >= best_rank:
                continue
            
            attempts = 0
            max_attempts = 10
            found = False
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                attempts += 1
                
                # Start from schoolbook and perturb
                U_try = np.zeros((n*m, target_rank))
                V_try = np.zeros((m*p, target_rank))
                W_try = np.zeros((p*n, target_rank))
                
                # Initialize with first target_rank schoolbook columns
                for r in range(min(target_rank, schoolbook_rank)):
                    i = r // (m*p)
                    j = (r // p) % m
                    k = r % p
                    U_try[i*m+j, r] = 1
                    V_try[j*p+k, r] = 1
                    W_try[k*n+i, r] = 1
                
                # Try random perturbations on the last few columns
                for r in range(max(0, target_rank - 5), target_rank):
                    for _ in range(2):
                        idx_u = random.randint(0, n*m-1)
                        U_try[idx_u, r] += random.choice([-1, 1])
                    for _ in range(2):
                        idx_v = random.randint(0, m*p-1)
                        V_try[idx_v, r] += random.choice([-1, 1])
                    for _ in range(2):
                        idx_w = random.randint(0, p*n-1)
                        W_try[idx_w, r] += random.choice([-1, 1])
                
                if verify(U_try, V_try, W_try):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_try, V_try, W_try
                    found = True
                    break
            
            if found:
                break
    
    # Return the best solution found
    U_out = to_clean_list(best_U)
    V_out = to_clean_list(best_V)
    W_out = to_clean_list(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
