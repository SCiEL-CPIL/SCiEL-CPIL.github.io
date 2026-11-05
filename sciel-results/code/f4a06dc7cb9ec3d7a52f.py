import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def _strassen_2x2x2():
    """Return U, V, W for Strassen 2x2x2 decomposition (rank 7).
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r] for i,j,k in {0,1}, r in 0..6
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 products:
    # M0 = (A00+A11)*(B00+B11)
    # M1 = (A10+A11)*B00
    # M2 = A00*(B01-B11)
    # M3 = A11*(B10-B00)
    # M4 = (A00+A01)*B11
    # M5 = (A10-A00)*(B00+B01)
    # M6 = (A01-A11)*(B10+B11)
    #
    # C00 = M0 + M3 - M5 - M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M2 + M1 + M4
    
    # U: A indices (i*2+j)
    # V: B indices (j*2+k)
    # W: C indices (k*2+i)
    
    # M0: (A00+A11)*(B00+B11)
    # U: A00=0, A11=3 -> U[0,0]=1, U[3,0]=1
    # V: B00=0, B11=3 -> V[0,0]=1, V[3,0]=1
    # W: C00=0, C11=3 -> W[0,0]=1, W[3,0]=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M1: (A10+A11)*B00
    # U: A10=2, A11=3 -> U[2,1]=1, U[3,1]=1
    # V: B00=0 -> V[0,1]=1
    # W: C10=2, C11=3 -> W[2,1]=1, W[3,1]=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M2: A00*(B01-B11)
    # U: A00=0 -> U[0,2]=1
    # V: B01=1, B11=3 -> V[1,2]=1, V[3,2]=-1
    # W: C01=1, C11=3 -> W[1,2]=1, W[3,2]=-1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M3: A11*(B10-B00)
    # U: A11=3 -> U[3,3]=1
    # V: B10=2, B00=0 -> V[2,3]=1, V[0,3]=-1
    # W: C00=0, C10=2 -> W[0,3]=1, W[2,3]=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M4: (A00+A01)*B11
    # U: A00=0, A01=1 -> U[0,4]=1, U[1,4]=1
    # V: B11=3 -> V[3,4]=1
    # W: C01=1, C11=3 -> W[1,4]=1, W[3,4]=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # M5: (A10-A00)*(B00+B01)
    # U: A10=2, A00=0 -> U[2,5]=1, U[0,5]=-1
    # V: B00=0, B01=1 -> V[0,5]=1, V[1,5]=1
    # W: C00=0, C10=2 -> W[0,5]=-1, W[2,5]=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = -1; W[2, 5] = 1
    
    # M6: (A01-A11)*(B10+B11)
    # U: A01=1, A11=3 -> U[1,6]=1, U[3,6]=-1
    # V: B10=2, B11=3 -> V[2,6]=1, V[3,6]=1
    # W: C00=0, C01=1 -> W[0,6]=-1, W[1,6]=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1
    
    return U, V, W

def _schoolbook_block(a, b, c):
    """Return U, V, W for schoolbook a×b×c decomposition.
    U[i*b+j, r], V[j*c+k, r], W[k*a+i, r]
    """
    rank = a * b * c
    U = np.zeros((a * b, rank))
    V = np.zeros((b * c, rank))
    W = np.zeros((c * a, rank))
    col = 0
    for i in range(a):
        for j in range(b):
            for k in range(c):
                U[i * b + j, col] = 1
                V[j * c + k, col] = 1
                W[k * a + i, col] = 1
                col += 1
    return U, V, W

def _block_decomposition(n, m, p):
    """Decompose M_{n,m,p} using block decomposition with Strassen for 2x2x2."""
    n, m, p = 3, 5, 7
    
    # Split dimensions
    n_splits = [2, 1]  # n = 2 + 1
    m_splits = [2, 2, 1]  # m = 2 + 2 + 1
    p_splits = [2, 2, 2, 1]  # p = 2 + 2 + 2 + 1
    
    assert sum(n_splits) == n
    assert sum(m_splits) == m
    assert sum(p_splits) == p
    
    # Precompute Strassen
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2()
    
    # Collect all terms
    all_U = []
    all_V = []
    all_W = []
    
    for a_idx, a in enumerate(n_splits):
        for b_idx, b in enumerate(m_splits):
            for c_idx, c in enumerate(p_splits):
                if a == 2 and b == 2 and c == 2:
                    # Use Strassen
                    U_block = strassen_U.copy()
                    V_block = strassen_V.copy()
                    W_block = strassen_W.copy()
                else:
                    # Use schoolbook
                    U_block, V_block, W_block = _schoolbook_block(a, b, c)
                
                # Determine offsets
                n_off = sum(n_splits[:a_idx])
                m_off = sum(m_splits[:b_idx])
                p_off = sum(p_splits[:c_idx])
                
                # U: index i*b + j, where i is within [0,a), j within [0,b)
                # Global U index: (n_off + i)*m + (m_off + j)
                # But U_block has shape (a*b, rank), with local index i*b+j
                # We need to map local (i*b+j) to global (n_off+i)*m + (m_off+j)
                U_full = np.zeros((n * m, U_block.shape[1]))
                for i in range(a):
                    for j in range(b):
                        local_idx = i * b + j
                        global_idx = (n_off + i) * m + (m_off + j)
                        U_full[global_idx, :] = U_block[local_idx, :]
                
                # V: index j*c + k, where j within [0,b), k within [0,c)
                # Global V index: (m_off + j)*p + (p_off + k)
                V_full = np.zeros((m * p, V_block.shape[1]))
                for j in range(b):
                    for k in range(c):
                        local_idx = j * c + k
                        global_idx = (m_off + j) * p + (p_off + k)
                        V_full[global_idx, :] = V_block[local_idx, :]
                
                # W: index k*a + i, where k within [0,c), i within [0,a)
                # Global W index: (p_off + k)*n + (n_off + i)
                W_full = np.zeros((p * n, W_block.shape[1]))
                for k in range(c):
                    for i in range(a):
                        local_idx = k * a + i
                        global_idx = (p_off + k) * n + (n_off + i)
                        W_full[global_idx, :] = W_block[local_idx, :]
                
                all_U.append(U_full)
                all_V.append(V_full)
                all_W.append(W_full)
    
    # Concatenate
    U = np.hstack(all_U)
    V = np.hstack(all_V)
    W = np.hstack(all_W)
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U, V, W = _block_decomposition(n, m, p)
    
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
