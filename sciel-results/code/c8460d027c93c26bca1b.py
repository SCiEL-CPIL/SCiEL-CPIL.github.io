"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import linprog
from itertools import combinations

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _build_rank14():
    """Return the known rank-14 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices
    # A: 0:00, 1:01, 2:10, 3:11
    # B: 0:00, 1:01, 2:02, 3:03, 4:10, 5:11, 6:12, 7:13
    # W (C): 0:00, 1:10, 2:01, 3:11, 4:02, 5:12, 6:03, 7:13
    # W index is k*n + i.

    # Strassen for B1 (cols 0,1)
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Strassen for B2 (cols 2,3)
    # N1
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return U, V, W


def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Tensor T[i,j,k] = 1 if i,j,k correspond to C[i,j] = A[i,:]*B[:,j]
    # A is n x m, B is m x p, C is n x p
    # T index: (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # For each output C[i,k], sum over j: A[i,j] * B[j,k]
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal delta(i,k) contribution
    # Actually, the decomposition is: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But that's not right. Let me think again.
    
    # Standard tensor decomposition for matrix multiplication:
    # T = sum_r U_r ⊗ V_r ⊗ W_r
    # where (U_r)_{i,j} * (V_r)_{j,k} * (W_r)_{k,i} = delta for the multiplication tensor
    
    # For C = A * B: C[i,k] = sum_j A[i,j] B[j,k]
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k (since every triple contributes)
    # Wait no. The tensor is T[i,j,k] = 1 if this is a valid multiplication step.
    # For matrix multiplication, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] B[j,k]
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(len(U[0])):
                    total += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _try_rank13():
    """Attempt to find a rank-13 decomposition via linear programming."""
    n, m, p = DIMENSIONS
    target_rank = 13
    
    # Build the constraint matrix
    # For each (i, j, k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is bilinear in the variables, so we can't directly use linprog.
    # Instead, we fix one factor and solve for others, or use a different approach.
    
    # Alternative: fix U and W, solve for V, or use iterative methods.
    # For now, return None to indicate no rank-13 found.
    return None


def construct():
    start_time = time.time()
    deadline = 145  # 120s limit + margin
    
    # Try to find rank-13 decomposition
    result_13 = _try_rank13()
    if result_13 is not None:
        return result_13
    
    # Fall back to known rank-14 decomposition
    U, V, W = _build_rank14()
    
    if time.time() > deadline:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
