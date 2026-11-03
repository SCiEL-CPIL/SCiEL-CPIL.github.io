import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the full tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try block decomposition with Strassen
    # Split: n=1+2, m=2+2, p=3+4
    
    def build_subtensor(n_s, m_s, p_s, use_strassen=False):
        """Build U,V,W for multiplying n_s x m_s by m_s x p_s."""
        if use_strassen and n_s == 2 and m_s == 2 and p_s == 2:
            # Strassen decomposition
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B11-B21)
            # M4 = A22(B21-B11)  # wait, let me get this right
            # Standard Strassen:
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B11-B21)
            # M4 = A22(B21-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            # C11 = M1+M4-M5+M7
            # C12 = M3+M5
            # C21 = M2+M4
            # C22 = M1-M2+M3+M6
            
            rank = 7
            U = np.zeros((n_s*m_s, rank), dtype=np.float64)
            V = np.zeros((m_s*p_s, rank), dtype=np.float64)
            W = np.zeros((p_s*n_s, rank), dtype=np.float64)
            
            # A is 2x2: A[i,j], B is 2x2: B[j,k], C is 2x2: C[i,k]
            # U index: i*m_s+j, V index: j*p_s+k, W index: k*n_s+i
            
            # M1 = (A11+A22)(B11+B22)
            # A11: i=0,j=0; A22: i=1,j=1
            # B11: j=0,k=0; B22: j=1,k=1
            U[0*2+0, 0] = 1  # A11
            U[1*2+1, 0] = 1  # A22
            V[0*2+0, 0] = 1  # B11
            V[1*2+1, 0] = 1  # B22
            # C11 += M1, C22 += M1
            W[0*2+0, 0] = 1  # C11 (k=0,i=0)
            W[1*2+1, 0] = 1  # C22 (k=1,i=1)
            
            # M2 = (A21+A22)B11
            U[1*2+0, 1] = 1  # A21
            U[1*2+1, 1] = 1  # A22
            V[0*2+0, 1] = 1  # B11
            # C21 += M2
            W[0*2+1, 1] = 1  # C21 (k=0,i=1)
            
            # M3 = A11(B11-B21)
            U[0*2+0, 2] = 1  # A11
            V[0*2+0, 2] = 1  # B11
            V[1*2+0, 2] = -1  # B21
            # C12 += M3
            W[0*2+0, 2] = 1  # C12 (k=0,i=0)
            
            # M4 = A22(B21-B11)
            U[1*2+1, 3] = 1  # A22
            V[1*2+0, 3] = 1  # B21
            V[0*2+0, 3] = -1  # B11
            # C11 += M4, C21 += M4
            W[0*2+0, 3] = 1  # C11
            W[0*2+1, 3] = 1  # C21
            
            # M5 = (A11+A12)B22
            U[0*2+0, 4] = 1  # A11
            U[0*2+1, 4] = 1  # A12
            V[1*2+1, 4] = 1  # B22
            # C11 -= M5, C12 += M5
            W[0*2+0, 4] = -1  # C11
            W[1*2+0, 4] = 1  # C12
            
            # M6 = (A21-A11)(B11+B12)
            U[1*2+0, 5] = 1  # A21
            U[0*2+0, 5] = -1  # A11
            V[0*2+0, 5] = 1  # B11
            V[0*2+1, 5] = 1  # B12
            # C22 += M6
            W[1*2+1, 5] = 1  # C22
            
            # M7 = (A12-A22)(B21+B22)
            U[0*2+1, 6] = 1  # A12
            U[1*2+1, 6] = -1  # A22
            V[1*2+0, 6] = 1  # B21
            V[1*2+1, 6] = 1  # B22
            # C11 += M7
            W[0*2+0, 6] = 1  # C11
            
            return U, V, W
        else:
            # Schoolbook
            rank = n_s * m_s * p_s
            U = np.zeros((n_s*m_s, rank), dtype=np.float64)
            V = np.zeros((m_s*p_s, rank), dtype=np.float64)
            W = np.zeros((p_s*n_s, rank), dtype=np.float64)
            col = 0
            for i in range(n_s):
                for j in range(m_s):
                    for k in range(p_s):
                        U[i*m_s+j, col] = 1
                        V[j*p_s+k, col] = 1
                        W[k*n_s+i, col] = 1
                        col += 1
            return U, V, W
    
    # Try different splits
    best_rank = n * m * p
    best_result = None
    
    splits = [
        (1, 2, 2, 2, 3, 4),
        (1, 2, 2, 2, 4, 3),
        (2, 1, 2, 2, 3, 4),
        (2, 1, 2, 2, 4, 3),
        (1, 2, 1, 3, 3, 4),
        (1, 2, 1, 3, 4, 3),
        (2, 1, 1, 3, 3, 4),
        (2, 1, 1, 3, 4, 3),
        (1, 2, 2, 2, 2, 5),
        (1, 2, 2, 2, 5, 2),
        (2, 1, 2, 2, 2, 5),
        (2, 1, 2, 2, 5, 2),
    ]
    
    for n1, n2, m1, m2, p1, p2 in splits:
        if n1 + n2 != n or m1 + m2 != m or p1 + p2 != p:
            continue
        
        # Build 4 sub-decompositions
        use_s11 = (n1 == 2 and m1 == 2 and p1 == 2)
        use_s12 = (n1 == 2 and m2 == 2 and p2 == 2)
        use_s21 = (n2 == 2 and m1 == 2 and p1 == 2)
        use_s22 = (n2 == 2 and m2 == 2 and p2 == 2)
        
        U11, V11, W11 = build_subtensor(n1, m1, p1, use_s11)
        U12, V12, W12 = build_subtensor(n1, m2, p2, use_s12)
        U21, V21, W21 = build_subtensor(n2, m1, p1, use_s21)
        U22, V22, W22 = build_subtensor(n2, m2, p2, use_s22)
        
        r11, r12, r21, r22 = U11.shape[1], U12.shape[1], U21.shape[1], U22.shape[1]
        total_rank = r11 + r12 + r21 + r22
        
        if total_rank >= best_rank:
            continue
        
        # Assemble full U, V, W
        # U has shape (n*m, total_rank)
        # Rows: for sub (1,1): i in [0,n1), j in [0,m1) → row i*m+j
        #        for sub (1,2): i in [0,n1), j in [m1,m1+m2) → row i*m+j
        #        for sub (2,1): i in [n1,n1+n2), j in [0,m1) → row i*m+j
        #        for sub (2,2): i in [n1,n1+n2), j in [m1,m1+m2) → row i*m+j
        
        U = np.zeros((n*m, total_rank), dtype=np.float64)
        V = np.zeros((m*p, total_rank), dtype=np.float64)
        W = np.zeros((p*n, total_rank), dtype=np.float64)
        
        col_offset = 0
        # Sub (1,1): n1 x m1 x p1
        # U11 has shape (n1*m1, r11), rows indexed by i_local*m1+j_local
        # In full U, row = i*m + j where i=i_local, j=j_local
        for i_local in range(n1):
            for j_local in range(m1):
                full_row = i_local * m + j_local
                U[full_row, col_offset:col_offset+r11] = U11[i_local*m1+j_local, :]
        
        # V11 has shape (m1*p1, r11), rows indexed by j_local*p1+k_local
        # In full V, row = j*p + k where j=j_local, k=k_local
        for j_local in range(m1):
            for k_local in range(p1):
                full_row = j_local * p + k_local
                V[full_row, col_offset:col_offset+r11] = V11[j_local*p1+k_local, :]
        
        # W11 has shape (p1*n1, r11), rows indexed by k_local*n1+i_local
        # In full W, row = k*n + i where k=k_local, i=i_local
        for k_local in range(p1):
            for i_local in range(n1):
                full_row = k_local * n + i_local
                W[full_row, col_offset:col_offset+r11] = W11[k_local*n1+i_local, :]
        
        col_offset += r11
        
        # Sub (1,2): n1 x m2 x p2
        # U12 has shape (n1*m2, r12), rows indexed by i_local*m2+j_local
        # In full U, row = i*m + (m1+j_local) where i=i_local
        for i_local in range(n1):
            for j_local in range(m2):
                full_row = i_local * m + (m1 + j_local)
                U[full_row, col_offset:col_offset+r12] = U12[i_local*m2+j_local, :]
        
        # V12 has shape (m2*p2, r12), rows indexed by j_local*p2+k_local
        # In full V, row = (m1+j_local)*p + k_local
        for j_local in range(m2):
            for k_local in range(p2):
                full_row = (m1 + j_local) * p + k_local
                V[full_row, col_offset:col_offset+r12] = V12[j_local*p2+k_local, :]
        
        # W12 has shape (p2*n1, r12), rows indexed by k_local*n1+i_local
        # In full W, row = (p1+k_local)*n + i_local
        for k_local in range(p2):
            for i_local in range(n1):
                full_row = (p1 + k_local) * n + i_local
                W[full_row, col_offset:col_offset+r12] = W12[k_local*n1+i_local, :]
        
        col_offset += r12
        
        # Sub (2,1): n2 x m1 x p1
        # U21 has shape (n2*m1, r21), rows indexed by i_local*m1+j_local
        # In full U, row = (n1+i_local)*m + j_local
        for i_local in range(n2):
            for j_local in range(m1):
                full_row = (n1 + i_local) * m + j_local
                U[full_row, col_offset:col_offset+r21] = U21[i_local*m1+j_local, :]
        
        # V21 has shape (m1*p1, r21), rows indexed by j_local*p1+k_local
        # In full V, row = j_local*p + k_local (same as sub 1,1)
        for j_local in range(m1):
            for k_local in range(p1):
                full_row = j_local * p + k_local
                V[full_row, col_offset:col_offset+r21] = V21[j_local*p1+k_local, :]
        
        # W21 has shape (p1*n2, r21), rows indexed by k_local*n2+i_local
        # In full W, row = k_local*n + (n1+i_local)
        for k_local in range(p1):
            for i_local in range(n2):
                full_row = k_local * n + (n1 + i_local)
                W[full_row, col_offset:col_offset+r21] = W21[k_local*n2+i_local, :]
        
        col_offset += r21
        
        # Sub (2,2): n2 x m2 x p2
        # U22 has shape (n2*m2, r22), rows indexed by i_local*m2+j_local
        # In full U, row = (n1+i_local)*m + (m1+j_local)
        for i_local in range(n2):
            for j_local in range(m2):
                full_row = (n1 + i_local) * m + (m1 + j_local)
                U[full_row, col_offset:col_offset+r22] = U22[i_local*m2+j_local, :]
        
        # V22 has shape (m2*p2, r22), rows indexed by j_local*p2+k_local
        # In full V, row = (m1+j_local)*p + (p1+k_local)
        for j_local in range(m2):
            for k_local in range(p2):
                full_row = (m1 + j_local) * p + (p1 + k_local)
                V[full_row, col_offset:col_offset+r22] = V22[j_local*p2+k_local, :]
        
        # W22 has shape (p2*n2, r22), rows indexed by k_local*n2+i_local
        # In full W, row = (p1+k_local)*n + (n1+i_local)
        for k_local in range(p2):
            for i_local in range(n2):
                full_row = (p1 + k_local) * n + (n1 + i_local)
                W[full_row, col_offset:col_offset+r22] = W22[k_local*n2+i_local, :]
        
        # Verify the decomposition
        # Check that U @ diag @ V^T reconstructs the tensor correctly
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :]) * 0  # This isn't right
                    # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_rank = total_rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if time.time() > deadline:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
