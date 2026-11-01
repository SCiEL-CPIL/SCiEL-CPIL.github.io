"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for matrix multiplication 2x4 by 4x7.
    Uses Strassen blocks with cross-block optimization to minimize rank.
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    def check_time():
        return time.time() < deadline
    
    # Build the initial rank-50 decomposition
    rank_init = 50
    U = np.zeros((n * m, rank_init))
    V = np.zeros((m * p, rank_init))
    W = np.zeros((p * n, rank_init))
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1
        U[d_idx, col] += 1
        V[e_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx, col] += 1
        U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1
        W[c11_idx, col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1
        V[h_idx, col] -= 1
        W[c01_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1
        V[e_idx, col] -= 1
        W[c00_idx, col] += 1
        W[c10_idx, col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx, col] += 1
        U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1
        W[c01_idx, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1
        U[a_idx, col] -= 1
        V[e_idx, col] += 1
        V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1
        U[d_idx, col] -= 1
        V[g_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx, col] += 1
        V[e_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1
        
        U[b_idx, col] += 1
        V[g_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1
        
        U[c_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1
        col += 1
        
        U[d_idx, col] += 1
        V[g_idx, col] += 1
        W[c10_idx, col] += 1
        col += 1

    # Build the initial rank-50 decomposition
    # Use 2x2x2 Strassen for 6 columns (3 blocks per j-group, 2 j-groups) = 42
    # Use 2x2x1 schoolbook for column 6 (2 blocks) = 8
    # Total: 50
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)
    
    # Try an alternative construction that might have lower rank
    # Use a 2x4x2 Strassen-like block for the first 2 columns
    # and a 2x4x2 for the next 2, etc.
    
    # Actually, let's try: 2x4*4x7 can be decomposed as:
    # - 2x4*4x4 using two 2x2*4x4 blocks, each with rank 14 (two 2x2x2 Strassen)
    # - 2x4*4x3 using two 2x2*4x3 blocks
    # For 2x2*4x3: split into 2x2*2x3 + 2x2*2x3
    # Each 2x2*2x3: use schoolbook = 12, or try a Strassen-like approach
    # Let's try: for 2x2*2x3, we have 6 products. Can we do it in fewer?
    # The tensor rank of 2x2*2x3 is at least 6 (since it's a 2x2x3 tensor)
    # Actually, the rank of the 2x2x3 multiplication tensor is 6 (it's minimal)
    # So 2x2*4x3 = 2*6 = 12
    # Total: 2*14 + 2*12 = 28 + 24 = 52. Worse.
    
    # Let's try yet another approach: use the fact that 7 = 4 + 3
    # and 4 = 2 + 2, 3 = 2 + 1
    # C = A(2x4) * B(4x7)
    # Split B into B1(4x4) and B2(4x3)
    # Split B1 into B11(4x2), B12(4x2)
    # Split B2 into B21(4x2), B22(4x1)
    # C1 = A*B1 = A*B11 + A*B12
    # C2 = A*B2 = A*B21 + A*B22
    # Each 2x4*4x2: split A into A1(2x2), A2(2x2)
    # 2x2*4x2: this is like 2x2*2x2 + 2x2*2x2 = 14
    # So 2x4*4x2 = 14
    # 2x4*4x4 = 2*14 = 28
    # 2x4*4x2 = 14
    # 2x4*4x1: schoolbook = 8
    # Total: 28 + 14 + 8 = 50. Same as before.
    
    # The current construction is already optimal for this block structure.
    # Let's try to find cross-block dependencies.
    
    def to_integer_list(mat):
        """Convert numpy matrix to list of lists with rounded integers."""
        return [[int(round(mat[i, j])) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    def verify(U_mat, V_mat, W_mat):
        """Verify that U, V, W correctly decompose the tensor."""
        # The tensor T[i*m+j, j*p+k, k*n+i] = 1
        # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # Use numpy for speed
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute the dot product efficiently
                    val = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to reduce rank by finding linear dependencies among the rank-1 terms
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1
    # Our decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r = U[:,r], v_r = V[:,r], w_r = W[:,r]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_init
    
    # Strategy: Try to find if any column is redundant by checking
    # if the tensor can still be represented without it.
    
    U_work = U.copy()
    V_work = V.copy()
    W_work = W.copy()
    active = list(range(rank_init))
    
    # Try removing columns one at a time, checking if the tensor is still valid
    # by verifying all 2*4*7 = 56 entries
    for iteration in range(rank_init):
        if not check_time():
            break
        
        removed = False
        for idx in active[:]:
            if not check_time():
                break
            
            test_active = [a for a in active if a != idx]
            if len(test_active) == 0:
                break
            
            U_test = U_work[:, test_active]
            V_test = V_work[:, test_active]
            W_test = W_work[:, test_active]
            
            if verify(U_test, V_test, W_test):
                U_work = U_test
                V_work = V_test
                W_work = W_test
                active = test_active
                removed = True
                break
        
        if not removed:
            break
    
    if len(active) < best_rank:
        best_U = U_work
        best_V = V_work
        best_W = W_work
        best_rank = len(active)
    
    # Try to find a lower-rank decomposition using a different approach
    # For 2x4*4x7, we can try to use the fact that the tensor has
    # a specific structure that might allow for a lower-rank decomposition
    
    # Approach: Use the fact that 7 = 4 + 3 and try to optimize the 4x3 part
    # 2x4*4x3: split into 2x2*4x3 + 2x2*4x3
    # Each 2x2*4x3: split into 2x2*2x3 + 2x2*2x3
    # Each 2x2*2x3: the tensor rank is 6 (minimal)
    # So 2x2*4x3 = 12, and 2x4*4x3 = 24
    
    # But we can try to overlap the 2x2*2x3 blocks with the Strassen blocks
    # from the 4x4 part
    
    # For now, let's try a randomized search: try to find a valid
    # decomposition of rank 49 by randomly permuting and checking
    
    if best_rank > 49 and check_time():
        # Try to find a rank-49 decomposition
        # This is a non-convex problem, so we'll use a simple heuristic:
        # Try to find a subset of 49 columns that can represent the tensor
        
        # For each pair of columns, check if removing one still works
        # This is what we already did, but let's try more aggressively
        
        # Try removing two columns at a time
        for i in range(len(active)):
            if not check_time():
                break
            for j in range(i+1, len(active)):
                if not check_time():
                    break
                
                test_active = [a for idx, a in enumerate(active) if idx not in (i, j)]
                if len(test_active) < 49:
                    continue
                
                U_test = U_work[:, test_active]
                V_test = V_work[:, test_active]
                W_test = W_work[:, test_active]
                
                if verify(U_test, V_test, W_test):
                    U_work = U_test
                    V_work = V_test
                    W_work = W_test
                    active = test_active
                    best_U = U_work
                    best_V = V_work
                    best_W = W_work
                    best_rank = len(active)
                    break
            if best_rank <= 49:
                break
    
    # Convert to output format
    U_list = to_integer_list(best_U)
    V_list = to_integer_list(best_V)
    W_list = to_integer_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
