import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    t0 = time.time()
    deadline = t0 + 110  # safety margin before 120s limit

    def get_schoolbook(n_dim, m_dim, p_dim):
        """
        Returns U, V, W for schoolbook decomposition of M_{n,m,p}.
        Rank is n*m*p.
        """
        nm = n_dim * m_dim
        mp = m_dim * p_dim
        pn = p_dim * n_dim
        R = nm * mp // n_dim # Wait, R = n*m*p
        
        R = n_dim * m_dim * p_dim
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        
        for i in range(n_dim):
            for j in range(m_dim):
                for k in range(p_dim):
                    r = i * (m_dim * p_dim) + j * p_dim + k
                    u_idx = i * m_dim + j
                    v_idx = j * p_dim + k
                    w_idx = k * n_dim + i
                    U[u_idx, r] = 1.0
                    V[v_idx, r] = 1.0
                    W[w_idx, r] = 1.0
        return U, V, W

    # M_{4,4,6} = M_{2,2,3} (outer) x M_{2,2,2} (inner)
    # A = A1 (2x2) x A2 (2x2)
    # B = B1 (2x3) x B2 (2x2)
    # C = C1 (2x3) x C2 (2x2)
    
    n1, m1, p1 = 2, 2, 3
    n2, m2, p2 = 2, 2, 2
    
    U1, V1, W1 = get_schoolbook(n1, m1, p1)
    R1 = U1.shape[1] # 12
    
    U2, V2, W2 = get_schoolbook(n2, m2, p2)
    R2 = U2.shape[1] # 8
    
    R_total = R1 * R2
    
    # Construct full U, V, W
    # Dimensions for M_{4,4,6}
    n_total, m_total, p_total = 4, 4, 6
    nm_total = n_total * m_total # 16
    mp_total = m_total * p_total # 24
    pn_total = p_total * n_total # 24
    
    U = np.zeros((nm_total, R_total))
    V = np.zeros((mp_total, R_total))
    W = np.zeros((pn_total, R_total))
    
    for r1 in range(R1):
        for r2 in range(R2):
            col = r1 * R2 + r2
            
            # Fill U
            # i = i1*2 + i2, j = j1*2 + j2
            # U_idx = i*4 + j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
            # U1_idx = i1*2 + j1
            # U2_idx = i2*2 + j2
            for i1 in range(2):
                for j1 in range(2):
                    u1_val = U1[i1*2 + j1, r1]
                    if u1_val == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2_val = U2[i2*2 + j2, r2]
                            if u2_val == 0: continue
                            u_idx = i1*8 + i2*4 + j1*2 + j2
                            U[u_idx, col] = u1_val * u2_val
            
            # Fill V
            # j = j1*2 + j2, k = k1*3 + k2
            # V_idx = j*6 + k = (j1*2+j2)*6 + (k1*3+k2) = j1*12 + j2*6 + k1*3 + k2
            # V1_idx = j1*3 + k1
            # V2_idx = j2*2 + k2
            for j1 in range(2):
                for k1 in range(3):
                    v1_val = V1[j1*3 + k1, r1]
                    if v1_val == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v2_val = V2[j2*2 + k2, r2]
                            if v2_val == 0: continue
                            v_idx = j1*12 + j2*6 + k1*3 + k2
                            V[v_idx, col] = v1_val * v2_val
            
            # Fill W
            # k = k1*3 + k2, i = i1*2 + i2
            # W_idx = k*4 + i = (k1*3+k2)*4 + (i1*2+i2) = k1*12 + k2*4 + i1*2 + i2
            # W1_idx = k1*2 + i1
            # W2_idx = k2*2 + i2
            for k1 in range(3):
                for i1 in range(2):
                    w1_val = W1[k1*2 + i1, r1]
                    if w1_val == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w2_val = W2[k2*2 + i2, r2]
                            if w2_val == 0: continue
                            w_idx = k1*12 + k2*4 + i1*2 + i2
                            W[w_idx, col] = w1_val * w2_val

    # Verification
    max_err = 0
    for i in range(n_total):
        for j in range(m_total):
            for k in range(p_total):
                u_idx = i * m_total + j
                v_idx = j * p_total + k
                w_idx = k * n_total + i
                val = np.dot(U[u_idx, :] * V[v_idx, :], W[w_idx, :])
                err = abs(val - 1)
                if err > max_err:
                    max_err = err

    if max_err > 1e-10:
        # Should not happen with exact schoolbook
        print(f"Warning: Verification error {max_err}")

    def to_list(mat):
        return [[round(x, 10) for x in row] for row in mat]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
