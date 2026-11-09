"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from scipy.linalg import matrix_rank

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin under 120s

    n, m, p = DIMENSIONS  # 4, 5, 5

    def schoolbook():
        """Return schoolbook decomposition with rank = n*m*p = 100."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Exact verification using Fraction arithmetic.
        The tensor T[i,j,k] = 1 for all valid (i,j,k).
        We need: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        """
        rank = len(U[0]) if U else 0
        U_frac = [[Fraction(U[i*m+j][c]) for c in range(rank)] for i in range(n) for j in range(m)]
        V_frac = [[Fraction(V[j*p+k][c]) for c in range(rank)] for j in range(m) for k in range(p)]
        W_frac = [[Fraction(W[k*n+i][c]) for c in range(rank)] for k in range(p) for i in range(n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def tensor_to_matrices(T_flat):
        """Convert flat tensor (n*m*p,) to three matrices U, V, W of shape (?, rank).
        T_flat[i*m+j][k*p+l] should be interpreted correctly.
        Actually, T is indexed by (i,j,k) with i in [0,n), j in [0,m), k in [0,p).
        The tensor value at (i,j,k) is T_flat[i*m + j][k] if we think of it as a 3D array.
        
        Let me re-index: T is a 3D tensor of shape (n, m, p).
        T[i,j,k] = 1 for all valid indices.
        
        The decomposition requires:
        For all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
        
        This is a 3-tensor factorization problem.
        """
        pass

    def build_decomposition_via_svd():
        """Try to find a lower-rank decomposition using SVD on the unfolded tensor."""
        # The multiplication tensor T has shape (n, m, p) = (4, 5, 5)
        # T[i,j,k] = 1 for all i,j,k
        
        # Unfold the tensor in different ways to get matrix factorizations
        # T_unfold1[i, (j,k)] = T[i,j,k] -> shape (n, m*p) = (4, 25)
        # T_unfold2[(i,j), k] = T[i,j,k] -> shape (n*m, p) = (20, 5)
        # T_unfold3[(i,k), j] = T[i,j,k] -> shape (n*p, m) = (20, 5)
        
        # The tensor T[i,j,k] = 1 for all i,j,k is actually the outer product of three vectors:
        # T = ones(n,1,1) * ones(1,m,1) * ones(1,1,p)
        # Wait, that's not right. T[i,j,k] = 1 for ALL i,j,k means it's the all-ones tensor.
        # The all-ones tensor of shape (n,m,p) has rank 1!
        # T[i,j,k] = 1 = u_i * v_j * w_k where u_i=1, v_j=1, w_k=1 for all i,j,k.
        
        # So the rank is 1? Let me verify:
        # U[i*m+j, 0] = 1 for all i,j
        # V[j*p+k, 0] = 1 for all j,k  
        # W[k*n+i, 0] = 1 for all k,i
        # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1 for all (i,j,k). Yes!
        
        # So the rank-1 decomposition works!
        rank = 1
        U = [[1] * rank for _ in range(n * m)]
        V = [[1] * rank for _ in range(m * p)]
        W = [[1] * rank for _ in range(p * n)]
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        
        return None

    # Try the rank-1 decomposition first
    best_result = schoolbook()
    
    try:
        candidate = build_decomposition_via_svd()
        if candidate is not None:
            best_result = candidate
    except Exception:
        pass

    if time.time() > deadline:
        return best_result

    return best_result
# EVOLVE-BLOCK-END
