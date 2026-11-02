import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _build_rank_14_3x3x2():
    """
    Build a rank-14 decomposition for 3x3x2 matrix multiplication.
    
    T[i,j,k] = 1 for i,j in {0,1,2}, k in {0,1}
    """
    n, m, p = 3, 3, 2
    rank = 14
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen for the 2x2x2 subproblem: A[1:3,1:3] * B[1:3,:] -> C[1:3,:]
    # A indices: (1,1)=4, (1,2)=5, (2,1)=7, (2,2)=8
    # B indices: (1,0)=2, (1,1)=3, (2,0)=5, (2,1)=6
    # C indices: (0,1)=1, (0,2)=2, (1,1)=4, (1,2)=5
    
    r = 0
    
    # M1 = (a11+a22)*(b10+b21)
    U[4, r] = 1; U[8, r] = 1
    V[2, r] = 1; V[6, r] = 1
    W[1, r] = 1; W[4, r] = 1
    r += 1
    
    # M2 = (a21+a22)*b10
    U[7, r] = 1; U[8, r] = 1
    V[2, r] = 1
    W[4, r] = 1
    r += 1
    
    # M3 = a11*(b11-b21)
    U[4, r] = 1
    V[3, r] = 1; V[6, r] = -1
    W[2, r] = 1
    r += 1
    
    # M4 = a22*(b20-b10)
    U[8, r] = 1
    V[5, r] = 1; V[2, r] = -1
    W[1, r] = 1; W[4, r] = 1
    r += 1
    
    # M5 = (a11+a12)*b21
    U[4, r] = 1; U[5, r] = 1
    V[6, r] = 1
    W[1, r] = -1; W[2, r] = 1
    r += 1
    
    # M6 = (a21-a11)*(b10+b20)
    U[7, r] = 1; U[4, r] = -1
    V[2, r] = 1; V[5, r] = 1
    W[5, r] = 1
    r += 1
    
    # M7 = (a12-a22)*(b11+b21)
    U[5, r] = 1; U[8, r] = -1
    V[3, r] = 1; V[6, r] = 1
    W[1, r] = 1
    r += 1
    
    # Boundary terms
    # Term 7: A[0,0]*B[0,k] for k=0,1 -> C[0,k]
    U[0, r] = 1
    V[0, r] = 1; V[1, r] = 1
    W[0, r] = 1; W[3, r] = 1
    r += 1
    
    # Term 8: A[1,0]*B[0,k] for k=0,1 -> C[1,k]
    U[3, r] = 1
    V[0, r] = 1; V[1, r] = 1
    W[1, r] = 1; W[4, r] = 1
    r += 1
    
    # Term 9: A[2,0]*B[0,k] for k=0,1 -> C[2,k]
    U[6, r] = 1
    V[0, r] = 1; V[1, r] = 1
    W[2, r] = 1; W[5, r] = 1
    r += 1
    
    # Term 10: A[0,2]*B[2,k] for k=0,1 -> C[0,k]
    U[2, r] = 1
    V[4, r] = 1; V[5, r] = 1
    W[0, r] = 1; W[3, r] = 1
    r += 1
    
    # Term 11: A[1,2]*B[2,k] for k=0,1 -> C[1,k]
    U[5, r] = 1
    V[4, r] = 1; V[5, r] = 1
    W[1, r] = 1; W[4, r] = 1
    r += 1
    
    # Term 12: A[2,2]*B[2,k] for k=0,1 -> C[2,k]
    U[8, r] = 1
    V[4, r] = 1; V[5, r] = 1
    W[2, r] = 1; W[5, r] = 1
    r += 1
    
    # Term 13: A[0,1]*B[1,k] for k=0,1 -> C[0,k]
    U[1, r] = 1
    V[2, r] = 1; V[3, r] = 1
    W[0, r] = 1; W[3, r] = 1
    r += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify if U, V, W exactly decompose the tensor."""
    rank = U.shape[1]
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_reconstructed, T_target, atol=1e-9)

def _build_schoolbook(n, m, p):
    R_school = n * m * p
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, r] = 1.0
                V_school[j * p + k, r] = 1.0
                W_school[k * n + i, r] = 1.0
                r += 1
    return U_school, V_school, W_school

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    deadline = time.time() + 110
    
    # Build three rank-14 decompositions for the three 3x3x2 blocks
    U_parts = []
    V_parts = []
    W_parts = []
    
    for block in range(3):
        U14, V14, W14 = _build_rank_14_3x3x2()
        U_parts.append(U14)
        V_parts.append(V14)
        W_parts.append(W14)
    
    # Combine them
    U_combined = np.hstack(U_parts)
    V_combined = np.hstack(V_parts)
    W_combined = np.hstack(W_parts)
    
    # Verify
    if _verify_decomposition(U_combined, V_combined, W_combined, n, m, p):
        return {"U": U_combined.tolist(), "V": V_combined.tolist(), "W": W_combined.tolist()}
    
    # Fallback to schoolbook
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
