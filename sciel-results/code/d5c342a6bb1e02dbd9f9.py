"""Matrix multiplication 4,4,6 tensor decomposition with rank minimization."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]
_DEADLINE = 140.0  # safety margin before 150s kill

def _strassen_2x2():
    """Return (U,V,W) rank-7 decomposition for 2x2x2 matrix mult.
    U shape (4,7), V shape (4,7), W shape (4,7).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A11+A22)(B11+B22)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (A21+A22)B11
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    # M3 = A11(B21-B22)
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[1,2] = 1
    
    # M4 = A22(B12-B22)
    U[3,3] = 1
    V[1,3] = 1; V[3,3] = -1
    W[0,3] = 1; W[3,3] = 1
    
    # M5 = (A11+A12)B22
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 via Kronecker product of two Strassen 2x2.
    U: (16, 49), V: (16, 49), W: (16, 49).
    Convention: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c].
    """
    U2, V2, W2 = _strassen_2x2()
    
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    for s in range(7):  # outer Strassen step
        for t in range(7):  # inner Strassen step
            c = s * 7 + t
            
            # For each outer step s and inner step t:
            # U contribution: for each (I, cA) in outer A-blocks and (i, cA2) in inner A:
            #   U[I*2+i, J*2+j, c] += cA * cA2
            # where J is from outer B-blocks and j from inner B
            # This is the Kronecker product structure.
            
            # Outer step s defines: A-blocks, B-blocks, C-blocks
            # Inner step t defines: sub-blocks within each 2x2
            
            # For the Kronecker product:
            # U[(I*2+i)*4 + (J*2+j), c] = U2[I*2+i, s] * U2[i*2+j, t]... 
            # No, that's not right either. Let me think more carefully.
            
            # The 4x4x4 tensor T4[i,j,k] = sum over (I,J,K,i',j',k') of 
            #   T2[I,J,K] * T2[i',j',k'] * delta constraints
            # where i = I*2+i', j = J*2+j', k = K*2+k'
            
            # So T4[i,j,k] = T2[i//2, j//2, k//2] * T2[i%2, j%2, k%2]
            # Wait, that's not quite right. The Kronecker product of tensors:
            # (T ⊗ S)[i1,i2, j1,j2, k1,k2] = T[i1,j1,k1] * S[i2,j2,k2]
            
            # For matrix multiplication: T[i,j,k] = 1 for all valid indices.
            # The Kronecker product decomposition:
            # If T2 has rank-7 decomposition: T2[i,j,k] = sum_s U2[i*2+j,s]*V2[j*2+k,s]*W2[k*2+i,s]
            # Then T4[i,j,k] where i=I*2+i', j=J*2+j', k=K*2+k':
            # T4[I*2+i', J*2+j', K*2+k'] = T2[I,J,K] * T2[i',j',k']
            # = (sum_s U2[I*2+J,s]*V2[J*2+K,s]*W2[K*2+I,s]) * (sum_t U2[i'*2+j',t]*V2[j'*2+k',t]*W2[k'*2+i',t])
            # = sum_{s,t} [U2[I*2+J,s]*U2[i'*2+j',t]] * [V2[J*2+K,s]*V2[j'*2+k',t]] * [W2[K*2+I,s]*W2[k'*2+i',t]]
            
            # So for each (s,t) pair:
            # U4[(I*2+i')*4 + (J*2+j'), s*7+t] += U2[I*2+J, s] * U2[i'*2+j', t]
            # V4[(J*2+j')*4 + (K*2+k'), s*7+t] += V2[J*2+K, s] * V2[j'*2+k', t]
            # W4[(K*2+k')*4 + (I*2+i'), s*7+t] += W2[K*2+I, s] * W2[k'*2+i', t]
            
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        for i_prime in range(2):
                            for j_prime in range(2):
                                for k_prime in range(2):
                                    i = I * 2 + i_prime
                                    j = J * 2 + j_prime
                                    k = K * 2 + k_prime
                                    
                                    u_val = U2[I*2+J, s] * U2[i_prime*2+j_prime, t]
                                    v_val = V2[J*2+K, s] * V2[j_prime*2+k_prime, t]
                                    w_val = W2[K*2+I, s] * W2[k_prime*2+i_prime, t]
                                    
                                    u_idx = i * 4 + j
                                    v_idx = j * 4 + k
                                    w_idx = k * 4 + i
                                    
                                    U[u_idx, c] += u_val
                                    V[v_idx, c] += v_val
                                    W[w_idx, c] += w_val
    
    return U, V, W

def _build_4x4x2():
    """Build rank-32 decomposition for 4x4x2 via schoolbook.
    U: (16, 32), V: (8, 32), W: (8, 32).
    Convention: U[i*4+j, c], V[j*2+k, c], W[k*4+i, c].
    """
    n, m, p = 4, 4, 2
    rank = n * m * p
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((8, rank), dtype=int)
    W = np.zeros((8, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _combine_4x4x4_and_4x4x2(U444, V444, W444, U442, V442, W442):
    """Combine 4x4x4 (k=0..3) and 4x4x2 (k=4..5) decompositions.
    Returns U: (16, 81), V: (24, 81), W: (24, 81).
    Convention: U[i*4+j, c], V[j*6+k, c], W[k*4+i, c].
    """
    rank444 = U444.shape[1]
    rank442 = U442.shape[1]
    total_rank = rank444 + rank442
    
    U = np.zeros((16, total_rank), dtype=int)
    V = np.zeros((24, total_rank), dtype=int)
    W = np.zeros((24, total_rank), dtype=int)
    
    # Part 1: 4x4x4 (k=0..3)
    for c in range(rank444):
        U[:, c] = U444[:, c]
        # V444 has shape (16, rank444), indices j*4+k for j in 0..3, k in 0..3
        # In the full V, indices are j*6+k
        for j in range(4):
            for k in range(4):
                V[j*6 + k, c] = V444[j*4 + k, c]
        # W444 has shape (16, rank444), indices k*4+i for k in 0..3, i in 0..3
        # In the full W, indices are k*4+i
        for k in range(4):
            for i in range(4):
                W[k*4 + i, c] = W444[k*4 + i, c]
    
    # Part 2: 4x4x2 (k=4..5)
    offset = rank444
    for c in range(rank442):
        cc = offset + c
        U[:, cc] = U442[:, c]
        # V442 has shape (8, rank442), indices j*2+k for j in 0..3, k in 0..1
        # In the full V, indices are j*6+(4+k)
        for j in range(4):
            for k in range(2):
                V[j*6 + (4+k), cc] = V442[j*2 + k, c]
        # W442 has shape (8, rank442), indices k*4+i for k in 0..1, i in 0..3
        # In the full W, indices are (4+k)*4+i
        for k in range(2):
            for i in range(4):
                W[(4+k)*4 + i, cc] = W442[k*4 + i, c]
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify decomposition. Returns max error."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(Ua[i*m+j, :] * Va[j*p+k, :] * Wa[k*n+i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _try_remove_redundant(U, V, W, n, m, p, max_time=30.0):
    """Try to remove redundant columns."""
    Ua = np.array(U, dtype=float).copy()
    Va = np.array(V, dtype=float).copy()
    Wa = np.array(W, dtype=float).copy()
    rank = Ua.shape[1]
    t_start = time.time()
    
    removed = 0
    while removed < rank and (time.time() - t_start) < max_time:
        progress = False
        for c in range(rank - removed):
            if (time.time() - t_start) > max_time:
                break
            # Try removing column c
            mask = [i for i in range(rank - removed) if i != c]
            if not mask:
                break
            
            U_test = Ua[:, mask]
            V_test = Va[:, mask]
            W_test = Wa[:, mask]
            
            # Quick check
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U_test[i*m+j, :] * V_test[j*p+k, :] * W_test[k*n+i, :])
                        err = abs(s - 1.0)
                        if err > max_err:
                            break
                    if max_err > 1e-10:
                        break
                if max_err > 1e-10:
                    break
            
            if max_err < 1e-10:
                Ua, Va, Wa = U_test, V_test, W_test
                rank = rank - 1
                removed += 1
                progress = True
                break
        
        if not progress:
            break
    
    return Ua.tolist(), Va.tolist(), Wa.tolist()

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Build 4x4x4 via Kronecker Strassen (rank 49)
    U444, V444, W444 = _build_4x4x4_strassen()
    
    # Build 4x4x2 (rank 32)
    U442, V442, W442 = _build_4x4x2()
    
    # Combine: 4x4x4 + 4x4x2 = 4x4x6 (rank 81)
    U, V, W = _combine_4x4x4_and_4x4x2(U444, V444, W444, U442, V442, W442)
    
    # Verify
    max_err = _verify(U, V, W, n, m, p)
    
    # Try to reduce rank if time permits
    if (time.time() - t0) < _DEADLINE - 10:
        U, V, W = _try_remove_redundant(U, V, W, n, m, p, max_time=_DEADLINE - time.time() - 5)
    
    return {"U": U, "V": V, "W": W}
