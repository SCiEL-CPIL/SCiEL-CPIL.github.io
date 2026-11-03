"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _strassen_2x2():
    """Return U, V, W for 2x2x2 matrix multiplication, rank 7."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 0: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: (A10+A11)B00 -> C10 += 1, C11 -= 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Term 2: A00(B01-B11) -> C01 += 1, C11 += 1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Term 3: A11(B10-B00) -> C10 += 1, C00 += 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    
    # Term 4: (A00+A01)B11 -> C01 -= 1, C11 += 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = -1; W[3][4] = 1
    
    # Term 5: (A10-A00)(B00+B01) -> C11 += 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 6: (A01-A11)(B10+B11) -> C00 += 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _strassen_4x4():
    """Return U, V, W for 4x4x4 matrix multiplication, rank 49."""
    U_inner, V_inner, W_inner = _strassen_2x2()
    
    outer_A = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1, (1,1):-1},
    ]
    outer_B = [
        {(0,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(1,0):1, (0,0):-1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (1,1):1},
    ]
    outer_C = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):-1},
        {(0,1):1, (1,1):1},
        {(1,0):1, (0,0):1},
        {(0,1):-1, (1,1):1},
        {(1,1):1},
        {(0,0):1},
    ]
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for t in range(7):
        for s in range(7):
            col = t*7 + s
            for i in range(4):
                for j in range(4):
                    bi, bj = i//2, j//2
                    li, lj = i%2, j%2
                    coeff = outer_A[t].get((bi,bj), 0)
                    if coeff != 0:
                        U[i*4+j][col] = coeff * U_inner[li*2+lj][s]
            for i in range(4):
                for j in range(4):
                    bi, bj = i//2, j//2
                    li, lj = i%2, j%2
                    coeff = outer_B[t].get((bi,bj), 0)
                    if coeff != 0:
                        V[i*4+j][col] = coeff * V_inner[li*2+lj][s]
            for i in range(4):
                for j in range(4):
                    bi, bj = i//2, j//2
                    li, lj = i%2, j%2
                    coeff = outer_C[t].get((bi,bj), 0)
                    if coeff != 0:
                        W[i*4+j][col] = coeff * W_inner[li*2+lj][s]
    
    return U, V, W

def _build_base_65():
    """Build the rank-65 solution: Strassen 4x4x4 (rank 49) + schoolbook 4x4x1 (rank 16)."""
    n, m, p = 4, 4, 5
    U_444, V_444, W_444 = _strassen_4x4()
    
    # Schoolbook 4x4x1
    U_441 = [[0]*16 for _ in range(16)]
    V_441 = [[0]*16 for _ in range(4)]
    W_441 = [[0]*16 for _ in range(4)]
    col = 0
    for i in range(4):
        for j in range(4):
            U_441[i*4+j][col] = 1
            V_441[j][col] = 1
            W_441[i][col] = 1
            col += 1
    
    rank = 65
    U_full = [[0]*rank for _ in range(16)]
    V_full = [[0]*rank for _ in range(20)]
    W_full = [[0]*rank for _ in range(20)]
    
    for i in range(16):
        for c in range(49):
            U_full[i][c] = U_444[i][c]
        for c in range(16):
            U_full[i][49+c] = U_441[i][c]
    
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V_full[j*5+k][c] = V_444[j*4+k][c]
    for j in range(4):
        for c in range(16):
            V_full[j*5+4][49+c] = V_441[j][c]
    
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W_full[k*4+i][c] = W_444[k*4+i][c]
    for i in range(4):
        for c in range(16):
            W_full[16+i][49+c] = W_441[i][c]
    
    return U_full, V_full, W_full

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reshape to 3D: U (n, m, rank), V (m, p, rank), W (p, n, rank)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute sum over rank of U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This is equivalent to: for each (i,j,k), sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
    
    # Efficient computation using einsum
    # We want T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This can be computed as:
    # First compute M[j,k,c] = V[j,k,c] * W[k,i,c] for each i... no that's not right.
    
    # Let's do it step by step:
    # For fixed i: sum_j,k T[i,j,k] where T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Alternative: reshape and use matmul
    # T[i,j,k] = sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c])
    
    # Let's compute it directly with numpy broadcasting
    # U_3d shape: (n, m, rank)
    # V_3d shape: (m, p, rank)  
    # W_3d shape: (p, n, rank)
    
    # We need: sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
    
    # This is a 4th order tensor contraction. Let's use einsum.
    # But the indices are tricky because W has k and i swapped in position.
    
    # Let's transpose W: W_3d_t[i,k,c] = W_3d[k,i,c]
    W_3d_t = W_3d.transpose(1, 0, 2)  # Now shape is (n, p, rank), W_3d_t[i,k,c] = W_3d[k,i,c]
    
    # Now we want: sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d_t[i,k,c]
    # = sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d_t[i,k,c]
    
    # This is: for each i, compute sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d_t[i,k,c] over j,k,c
    # For fixed i: result[i,j,k] = sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d_t[i,k,c]
    
    # We can compute this as:
    # For fixed i and j: sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d_t[i,k,c]
    # = sum_c U_3d[i,j,c] * (V_3d[j,k,c] * W_3d_t[i,k,c])
    
    # Let's use einsum:
    # np.einsum('ijc,jkc,ikc->ijk', U_3d, V_3d, W_3d_t)
    
    result = np.einsum('ijc,jkc,ikc->ijk', U_3d, V_3d, W_3d_t, optimize=True)
    
    return np.allclose(result, 1.0, atol=tol)

def _try_reduce_rank(U, V, W, n, m, p, max_removals=10, deadline=None):
    """Try to reduce rank by removing linearly dependent columns."""
    rank = len(U[0])
    current_U, current_V, current_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    current_rank = rank
    
    for attempt in range(max_removals):
        if deadline and time.time() > deadline - 10:
            break
        
        # Try removing each column and check if the remaining can still satisfy
        found_removal = False
        for c in range(current_rank):
            if deadline and time.time() > deadline - 10:
                break
            
            # Temporarily remove column c
            test_U = [row[:c] + row[c+1:] for row in current_U]
            test_V = [row[:c] + row[c+1:] for row in current_V]
            test_W = [row[:c] + row[c+1:] for row in current_W]
            
            if _verify_decomposition(test_U, test_V, test_W, n, m, p):
                current_U, current_V, current_W = test_U, test_V, test_W
                current_rank -= 1
                found_removal = True
                break
        
        if not found_removal:
            break
    
    return current_U, current_V, current_W, current_rank

def _try_rank_reduction_via_linear_combination(U, V, W, n, m, p, deadline=None):
    """
    Try to find a lower rank decomposition by searching for linear combinations
    of existing terms that can replace pairs of terms.
    """
    rank = len(U[0])
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = rank
    
    if deadline and time.time() > deadline - 30:
        return best_U, best_V, best_W, best_rank
    
    # Try to see if we can reduce by 1 by finding a column that is a linear combination
    # of others in a way that preserves the tensor identity.
    
    # For each pair of columns (a, b), check if there exists a scalar alpha such that
    # replacing column a with (column a - alpha * column b) and removing column b
    # still gives a valid decomposition.
    
    # This is expensive, so let's be selective.
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Try to find if any column is redundant
    for c_remove in range(rank):
        if deadline and time.time() > deadline - 30:
            break
        
        # Remove column c_remove and verify
        test_U = [row[:c_remove] + row[c_remove+1:] for row in best_U]
        test_V = [row[:c_remove] + row[c_remove+1:] for row in best_V]
        test_W = [row[:c_remove] + row[c_remove+1:] for row in best_W]
        
        if _verify_decomposition(test_U, test_V, test_W, n, m, p):
            best_U, best_V, best_W = test_U, test_V, test_W
            best_rank -= 1
            U_arr = np.array(best_U, dtype=np.float64)
            V_arr = np.array(best_V, dtype=np.float64)
            W_arr = np.array(best_W, dtype=np.float64)
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Build base rank-65 solution
    U_full, V_full, W_full = _build_base_65()
    
    # Verify base solution
    if not _verify_decomposition(U_full, V_full, W_full, n, m, p):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to reduce rank using column removal
    U_red, V_red, W_red, final_rank = _try_reduce_rank(U_full, V_full, W_full, n, m, p, max_removals=10, deadline=deadline)
    
    # Verify reduced solution
    if _verify_decomposition(U_red, V_red, W_red, n, m, p):
        return {"U": U_red, "V": V_red, "W": W_red}
    
    # Return base solution if reduction failed verification
    return {"U": U_full, "V": V_full, "W": W_full}

# EVOLVE-BLOCK-END
