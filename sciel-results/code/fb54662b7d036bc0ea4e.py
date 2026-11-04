import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 12 decomposition
    # C_0 = A_0 B (1x4x8) -> Rank 4
    # C_1 = A_1 B (1x4x8) -> Rank 4
    # C_2 = A_2 B (1x4x8) -> Rank 4
    
    rank = 12
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # For each row i of A (0 to 2)
    for i in range(n):
        # For each column j of A (0 to 3)
        for j in range(m):
            c = i * m + j
            # U[i*m + j, c] = 1
            U[i * m + j, c] = 1
            # V[j*p + k, c] for all k?
            # M_c = A[i,j] * (sum_k B[j,k] * W[k,i,c])
            # We want C[i,k] = sum_j A[i,j] B[j,k]
            # C[i,k] = sum_c W[k,i,c] M_c
            # M_c = A[i,j] * 1
            # So V[j*p + k, c] should be 0?
            # If V is 0, M_c = 0.
            # We need K_c(B) to be 1?
            # If K_c(B) = 1, then M_c = A[i,j].
            # Then C[i,k] = sum_c W[k,i,c] A[i,j].
            # This requires W[k,i,c] = B[j,k]? No, W is constant.
            
            # The rank-m decomposition for 1xm xp is:
            # M_j = A_j * 1? No.
            # M_j = A_j * (linear form of B).
            # Actually, rank is m.
            # M_j = A_j * B_{j,1}?
            # Let's use the identity:
            # C_k = sum_j A_j B_{j,k}
            # This is sum_j (A_j) (B_{j,k})
            # This is not bilinear in A and B with scalar M_j.
            # Unless we allow M_j to depend on k? No.
            
            # Wait, the rank of 1xm xp is m.
            # M_j = A_j * (sum_k B_{j,k} delta_{k,1})?
            # No, the standard result is rank m.
            # M_j = A_j * 1 is not possible because K_c(B) must be non-zero.
            # If K_c(B) = 1, then M_c = A_j.
            # Then C_k = sum_c W_{k,c} A_j.
            # This implies C_k is linear in A only.
            
            # Correct decomposition for 1xm xp:
            # Rank is m.
            # M_j = A_j * B_{j,1}?
            # C_1 = M_1.
            # C_2 = ?
            
            # Actually, for 1xm xp, the rank is m.
            # M_j = A_j * (B_{j,1} + ... + B_{j,p})?
            # No.
            
            # Let's stick to schoolbook for safety if rank 12 is hard.
            # Schoolbook rank 96.
            # But 12 is much better.
            
            # Re-evaluating rank of 1xm xp.
            # C_k = sum_j A_j B_{j,k}
            # M_j = A_j * B_{j,1}
            # C_1 = M_1 + ...
            # This is not separable.
            
            # The rank of the tensor T_{j,k} = 1 is 1?
            # T_{j,k} = sum_c U_{j,c} V_{k,c}
            # 1 = sum_c U_{j,c} V_{k,c}
            # U_j = 1, V_k = 1/delta?
            # If U_j=1, V_k=1, sum=1. Rank 1.
            # So 1xm xp has rank 1?
            # M_1 = (sum_j A_j) (sum_k B_{j,k})?
            # C_k = W_{k,1} M_1.
            # C_k = W_{k,1} (sum_j A_j) (sum_k B_{j,k})
            # This is not sum_j A_j B_{j,k}.
            
            # So rank is m.
            # M_j = A_j * 1? No.
            
            # I will use schoolbook rank 96 to be safe.
            pass

    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
