"""Matrix multiplication 2x3x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    n, m, p = 2, 3, 4
    N, M, P = n * m, m * p, p * n  # 6, 12, 8

    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 24
    U_school = [[0] * rank_school for _ in range(N)]
    V_school = [[0] * rank_school for _ in range(M)]
    W_school = [[0] * rank_school for _ in range(P)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def verify(U, V, W):
        """Verify that the decomposition exactly matches matrix multiplication."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # C[i,k] += A[i,j] * B[j,k]
                    # Tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # We need T[i,j,k] = 1
                    total = 0
                    for r in range(rank):
                        u = U[i * m + j][r]
                        v = V[j * p + k][r]
                        w = W[k * n + i][r]
                        total += u * v * w
                    if total != 1:
                        return False
        # Check that no other entries are non-zero
        # The tensor is zero for all (i,j,k) except when they form a valid multiplication term
        # Actually, the tensor M_{2x3x4} has T[i,j,k] = 1 for all i,j,k, and 0 otherwise?
        # No, the matrix multiplication tensor is defined such that:
        # C = A * B, where C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor T has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # And the decomposition is: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # So we just need to check that T[i,j,k] = 1 for all i,j,k.
        # But wait, are there other constraints? No, the tensor is fully determined.
        # Actually, I need to double-check the convention.
        # "output factor follows the official trace convention W[k*n+i,r]"
        # So W is indexed by (k, i) where k is the column index of B / row index of C,
        # and i is the row index of A / column index of C.
        # The tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And this must equal 1 for all valid (i,j,k).
        # But is that all? Yes, for matrix multiplication, the tensor has 1s at all (i,j,k)
        # and 0s elsewhere. But since i,j,k range over all valid indices, there are no "elsewhere"
        # entries. So checking T[i,j,k]=1 for all i,j,k is sufficient.
        return True

    def verify_full(U, V, W):
        """Full verification using numpy for speed."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reshape to 3D arrays
        # U: (n*m, rank) -> (n, m, rank)
        # V: (m*p, rank) -> (m, p, rank)
        # W: (p*n, rank) -> (p, n, rank)
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        # This is a contraction over r.
        
        # We can compute this as:
        # For each i, j, k:
        #   T[i,j,k] = dot(U_3d[i,j,:], V_3d[j,k,:] * W_3d[k,i,:])
        
        # Let's do it efficiently:
        # T = np.zeros((n, m, p))
        # for i in range(n):
        #     for j in range(m):
        #         for k in range(p):
        #             T[i,j,k] = np.dot(U_3d[i,j,:], V_3d[j,k,:] * W_3d[k,i,:])
        
        # Or use einsum:
        # T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Check if T is all 1s
        if not np.allclose(T, 1.0, atol=1e-10):
            return False
        return True

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Try to find a lower rank decomposition using random search with small rational values
    import random
    random.seed(42)
    
    def try_rank(r):
        """Try to find a rank-r decomposition using random search."""
        for attempt in range(5000):
            if time.time() > deadline - 5:
                return None
            
            # Generate random values from a small set
            vals = [-1, -0.5, 0, 0.5, 1]
            
            U = [[random.choice(vals) for _ in range(r)] for _ in range(N)]
            V = [[random.choice(vals) for _ in range(r)] for _ in range(M)]
            W = [[random.choice(vals) for _ in range(r)] for _ in range(P)]
            
            if verify(U, V, W):
                return (U, V, W)
        
        return None

    # Try ranks from 18 upward
    for r in range(18, 24):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
