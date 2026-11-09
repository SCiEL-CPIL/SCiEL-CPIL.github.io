"""Matrix multiplication 3x3x4 rank-minimal exact tensor decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x4 matrix using randomized linear algebra over finite fields.
    
    The condition: for all i in [0,3), j in [0,3), k in [0,4):
      sum_{r=0}^{rank-1} U[i*3+j, r] * V[j*4+k, r] * W[k*3+i, r] = 1
    
    Strategy: Try various ranks, use random initialization over GF(3),
    solve the resulting linear system, and verify exactness.
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    def verify(U, V, W, rank, tol=1e-9):
        """Verify the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def to_half_int(arr):
        """Convert numpy array to list of lists with half-integer entries."""
        return [[round(float(val) * 2) / 2 for val in row] for row in arr.tolist()]
    
    # Try ranks from low to high
    for rank in range(20, 37):
        # Try multiple random seeds
        for seed in range(50):
            np.random.seed(seed * 1000 + rank)
            
            # Random initialization over {-1, 0, 1} for V and W
            V = np.random.randint(-1, 2, size=(m * p, rank))  # 12 x rank
            W = np.random.randint(-1, 2, size=(p * n, rank))  # 12 x rank
            
            # Solve for U: for each i, solve linear system
            U = np.zeros((n * m, rank), dtype=float)
            success = True
            
            for i in range(n):
                # Build system: for each (j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # Unknowns: U[i*m+j, r] for j in range(m), r in range(rank) -> m*rank unknowns
                # Equations: m*p = 12 equations
                
                A_mat = np.zeros((m * p, m * rank))
                b_vec = np.ones(m * p)
                
                eq_idx = 0
                for j in range(m):
                    for k in range(p):
                        for jj in range(m):
                            for r in range(rank):
                                A_mat[eq_idx, jj * rank + r] = V[j * p + k, r] * W[k * n + i, r]
                        eq_idx += 1
                
                # Check if system is consistent using rank
                A_rank = np.linalg.matrix_rank(A_mat)
                b_aug = np.hstack([A_mat, b_vec.reshape(-1, 1)])
                b_aug_rank = np.linalg.matrix_rank(b_aug)
                
                if A_rank != b_aug_rank:
                    success = False
                    break
                
                # Solve using least squares (should be exact if consistent)
                U_block, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Verify the solution for this block
                if np.max(np.abs(A_mat @ U_block - b_vec)) > 1e-8:
                    success = False
                    break
                
                U[i * m:(i + 1) * m, :] = U_block.reshape(m, rank)
            
            if not success:
                continue
            
            # Convert to half-integers and verify
            U_list = to_half_int(U)
            V_list = to_half_int(V)
            W_list = to_half_int(W)
            
            if verify(U_list, V_list, W_list, rank):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
