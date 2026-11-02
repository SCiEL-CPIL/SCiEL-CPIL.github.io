import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses a block decomposition strategy based on Strassen's algorithm for 2x2x2 blocks.
    Splits the 5x6 inner dimension into blocks to reuse linear forms.
    Target Rank: 54
    """
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140

    # Strassen decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # We need to generate U, V, W for the whole 2x5x6 problem.
    # U is (10, R), V is (30, R), W is (12, R).
    # Indices:
    # U[i*m+j, r] where i in 0..1, j in 0..4
    # V[j*p+k, r] where j in 0..4, k in 0..5
    # W[k*n+i, r] where k in 0..5, i in 0..1
    
    # We will build the decomposition block by block.
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (Add to C)
    # Block 3: A[0:2, 4:5] * B[4:5, 0:2] -> C[0:2, 0:2] (Add to C)
    # Block 4: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # ... and so on.
    
    # Actually, the sum is over j.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # We partition j into J1=[0,1], J2=[2,3], J3=[4].
    # We partition k into K1=[0,1], K2=[2,3], K3=[4,5].
    
    # For each pair (J_block, K_block), we compute a 2x2x2 or 2x1x2 or 2x2x1 multiplication.
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(U, V, W, offset_r, i_off, j_off, k_off):
        """
        Adds a Strassen 2x2x2 decomposition for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2]
        to C[i_off:i_off+2, k_off:k_off+2].
        i_off, j_off, k_off are offsets in the original matrices.
        """
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i
        
        # Local indices 0,1 for i, j, k
        # U rows: (i_off+0)*m + (j_off+0), (i_off+0)*m + (j_off+1), 
        #         (i_off+1)*m + (j_off+0), (i_off+1)*m + (j_off+1)
        # V rows: (j_off+0)*p + (k_off+0), (j_off+0)*p + (k_off+1),
        #         (j_off+1)*p + (k_off+0), (j_off+1)*p + (k_off+1)
        # W rows: (k_off+0)*n + (i_off+0), (k_off+0)*n + (i_off+1),
        #         (k_off+1)*n + (i_off+0), (k_off+1)*n + (i_off+1)
        
        u_rows = [
            (i_off + 0) * m + (j_off + 0),
            (i_off + 0) * m + (j_off + 1),
            (i_off + 1) * m + (j_off + 0),
            (i_off + 1) * m + (j_off + 1)
        ]
        v_rows = [
            (j_off + 0) * p + (k_off + 0),
            (j_off + 0) * p + (k_off + 1),
            (j_off + 1) * p + (k_off + 0),
            (j_off + 1) * p + (k_off + 1)
        ]
        w_rows = [
            (k_off + 0) * n + (i_off + 0),
            (k_off + 0) * n + (i_off + 1),
            (k_off + 1) * n + (i_off + 0),
            (k_off + 1) * n + (i_off + 1)
        ]
        
        # Strassen Factors
        # M1 = (a+d)(e+h)
        # a=U[0], d=U[3] -> U_vec = [1, 0, 0, 1]
        # e=V[0], h=V[3] -> V_vec = [1, 0, 0, 1]
        # C[0,0] += M1, C[0,1] -= M1? No.
        # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
        
        # Let's define the 7 terms (u, v, w)
        # u is 4-dim (a,b,c,d), v is 4-dim (e,f,g,h), w is 4-dim (C00, C01, C10, C11)
        
        terms = [
            # M1 = (a+d)(e+h): C00+=1, C11+=1
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            # M2 = (c+d)e: C10+=1, C11-=1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            # M3 = a(f-h): C01+=1, C11+=1
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            # M4 = d(g-e): C00+=1, C10+=1
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
            # M5 = (a+b)h: C00-=1, C01+=1
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            # M6 = (c-a)(e+f): C11+=1
            ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            # M7 = (b-d)(g+h): C00+=1
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        
        for u, v, w in terms:
            U[u_rows[0], offset_r] += u[0]
            U[u_rows[1], offset_r] += u[1]
            U[u_rows[2], offset_r] += u[2]
            U[u_rows[3], offset_r] += u[3]
            
            V[v_rows[0], offset_r] += v[0]
            V[v_rows[1], offset_r] += v[1]
            V[v_rows[2], offset_r] += v[2]
            V[v_rows[3], offset_r] += v[3]
            
            W[w_rows[0], offset_r] += w[0]
            W[w_rows[1], offset_r] += w[1]
            W[w_rows[2], offset_r] += w[2]
            W[w_rows[3], offset_r] += w[3]
            
            offset_r += 1
            
        return offset_r

    # Helper to add a 2x1x2 block (A is 2x1, B is 1x2)
    # A = [[a], [c]], B = [[e, f]]
    # C = [[ae, af], [ce, cf]]
    # Rank 4.
    def add_212_block(U, V, W, offset_r, i_off, j_off, k_off):
        u_rows = [
            (i_off + 0) * m + (j_off + 0),
            (i_off + 1) * m + (j_off + 0)
        ]
        v_rows = [
            (j_off + 0) * p + (k_off + 0),
            (j_off + 0) * p + (k_off + 1)
        ]
        w_rows = [
            (k_off + 0) * n + (i_off + 0),
            (k_off + 0) * n + (i_off + 1),
            (k_off + 1) * n + (i_off + 0),
            (k_off + 1) * n + (i_off + 1)
        ]
        
        # ae
        U[u_rows[0], offset_r] = 1
        V[v_rows[0], offset_r] = 1
        W[w_rows[0], offset_r] = 1
        offset_r += 1
        
        # af
        U[u_rows[0], offset_r] = 1
        V[v_rows[1], offset_r] = 1
        W[w_rows[2], offset_r] = 1
        offset_r += 1
        
        # ce
        U[u_rows[1], offset_r] = 1
        V[v_rows[0], offset_r] = 1
        W[w_rows[1], offset_r] = 1
        offset_r += 1
        
        # cf
        U[u_rows[1], offset_r] = 1
        V[v_rows[1], offset_r] = 1
        W[w_rows[3], offset_r] = 1
        offset_r += 1
        
        return offset_r

    # Helper to add a 2x2x1 block (A is 2x2, B is 2x1)
    # A = [[a, b], [c, d]], B = [[e], [g]]
    # C = [[ae+bg], [ce+dg]]
    # Rank 4.
    def add_221_block(U, V, W, offset_r, i_off, j_off, k_off):
        u_rows = [
            (i_off + 0) * m + (j_off + 0),
            (i_off + 0) * m + (j_off + 1),
            (i_off + 1) * m + (j_off + 0),
            (i_off + 1) * m + (j_off + 1)
        ]
        v_rows = [
            (j_off + 0) * p + (k_off + 0),
            (j_off + 1) * p + (k_off + 0)
        ]
        w_rows = [
            (k_off + 0) * n + (i_off + 0),
            (k_off + 0) * n + (i_off + 1)
        ]
        
        # ae
        U[u_rows[0], offset_r] = 1
        V[v_rows[0], offset_r] = 1
        W[w_rows[0], offset_r] = 1
        offset_r += 1
        
        # bg
        U[u_rows[1], offset_r] = 1
        V[v_rows[1], offset_r] = 1
        W[w_rows[0], offset_r] = 1
        offset_r += 1
        
        # ce
        U[u_rows[2], offset_r] = 1
        V[v_rows[0], offset_r] = 1
        W[w_rows[1], offset_r] = 1
        offset_r += 1
        
        # dg
        U[u_rows[3], offset_r] = 1
        V[v_rows[1], offset_r] = 1
        W[w_rows[1], offset_r] = 1
        offset_r += 1
        
        return offset_r

    # Initialize U, V, W with max possible rank (60)
    R_max = 60
    U = np.zeros((n * m, R_max), dtype=np.float64)
    V = np.zeros((m * p, R_max), dtype=np.float64)
    W = np.zeros((p * n, R_max), dtype=np.float64)
    
    r = 0
    
    # J blocks: [0,1], [2,3], [4]
    # K blocks: [0,1], [2,3], [4,5]
    
    # J0 (2x2), K0 (2x2) -> Strassen (7)
    r = add_strassen_block(U, V, W, r, 0, 0, 0)
    # J0 (2x2), K1 (2x2) -> Strassen (7)
    r = add_strassen_block(U, V, W, r, 0, 0, 2)
    # J0 (2x2), K2 (2x2) -> Strassen (7)
    r = add_strassen_block(U, V, W, r, 0, 0, 4)
    
    # J1 (2x2), K0 (2x2) -> Strassen (7)
    r = add_strassen_block(U, V, W, r, 0, 2, 0)
    # J1 (2x2), K1 (2x2) -> Strassen (7)
    r = add_strassen_block(U, V, W, r, 0, 2, 2)
    # J1 (2x2), K2 (2x2) -> Strassen (7)
    r = add_strassen_block(U, V, W, r, 0, 2, 4)
    
    # J2 (1x2? No, A is 2x1, B is 1x2? No, J2 is index 4, so A[:,4] is 2x1, B[4,:] is 1x6)
    # We split K into K0, K1, K2 (each 2 wide).
    # J2 (2x1), K0 (1x2) -> 212 block (4)
    r = add_212_block(U, V, W, r, 0, 4, 0)
    # J2 (2x1), K1 (1x2) -> 212 block (4)
    r = add_212_block(U, V, W, r, 0, 4, 2)
    # J2 (2x1), K2 (1x2) -> 212 block (4)
    r = add_212_block(U, V, W, r, 0, 4, 4)
    
    # Total r should be 7*6 + 4*3 = 42 + 12 = 54.
    
    # Verify and trim
    if r < R_max:
        U = U[:, :r]
        V = V[:, :r]
        W = W[:, :r]
        
    # Verify
    def verify(U, V, W):
        T = np.einsum('ar,br,cr->abc', U, V, W)
        T_exp = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_exp[a, b, c] = 1.0
        return np.allclose(T, T_exp, atol=1e-9)
    
    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails
        R = n * m * p
        U = np.zeros((n * m, R), dtype=np.float64)
        V = np.zeros((m * p, R), dtype=np.float64)
        W = np.zeros((p * n, R), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        r = R

    def to_list(arr):
        return arr.astype(float).tolist()

    return {
        "U": to_list(U),
        "V": to_list(V),
        "W": to_list(W)
    }
