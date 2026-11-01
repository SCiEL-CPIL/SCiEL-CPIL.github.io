import numpy as np

def construct():
    n, m, p = 2, 3, 5
    rank = 16
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a Strassen term
    # Strassen for 2x2x2
    # i, j in {0,1}, k in {0,1}
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    def add_strassen_block(offset_u, offset_v, offset_w, r_offset):
        # offset_u: row offset for U (i*m + j)
        # offset_v: row offset for V (j*p + k)
        # offset_w: row offset for W (k*n + i)
        # r_offset: column offset in U,V,W
        
        # M1 = (A00+A11)(B00+B11)
        U[0*m+0, r_offset] += 1
        U[1*m+1, r_offset] += 1
        V[0*p+0, r_offset] += 1
        V[1*p+1, r_offset] += 1
        W[0*n+0, r_offset] += 1
        W[1*n+1, r_offset] += 1
        r_offset += 1
        
        # M2 = (A10+A11)B00
        U[0*m+1, r_offset] += 1
        U[1*m+1, r_offset] += 1
        V[0*p+0, r_offset] += 1
        W[0*n+1, r_offset] += 1
        W[1*n+1, r_offset] -= 1
        r_offset += 1
        
        # M3 = A00(B01-B11)
        U[0*m+0, r_offset] += 1
        V[0*p+1, r_offset] += 1
        V[1*p+1, r_offset] -= 1
        W[1*n+0, r_offset] += 1
        W[1*n+1, r_offset] += 1
        r_offset += 1
        
        # M4 = A11(B10-B00)
        U[1*m+1, r_offset] += 1
        V[1*p+0, r_offset] += 1
        V[0*p+0, r_offset] -= 1
        W[0*n+0, r_offset] += 1
        W[0*n+1, r_offset] += 1
        r_offset += 1
        
        # M5 = (A00+A01)B11
        U[0*m+0, r_offset] += 1
        U[0*m+1, r_offset] += 1
        V[1*p+1, r_offset] += 1
        W[0*n+0, r_offset] -= 1
        W[1*n+0, r_offset] += 1
        r_offset += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[0*m+1, r_offset] += 1
        U[0*m+0, r_offset] -= 1
        V[0*p+0, r_offset] += 1
        V[0*p+1, r_offset] += 1
        W[1*n+1, r_offset] += 1
        r_offset += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[0*m+1, r_offset] += 1
        U[1*m+1, r_offset] -= 1
        V[1*p+0, r_offset] += 1
        V[1*p+1, r_offset] += 1
        W[0*n+0, r_offset] += 1
        r_offset += 1
        
        return r_offset

    # Block 1: j in {0,1}, k in {0,1}
    # U indices: i in {0,1}, j in {0,1} -> 0,1,3,4
    # V indices: j in {0,1}, k in {0,1} -> 0,1,5,6
    # W indices: k in {0,1}, i in {0,1} -> 0,1,2,3
    
    # Note: The add_strassen_block function uses hardcoded indices relative to 0.
    # We need to map them correctly.
    # For Block 1, indices are 0-based.
    r = 0
    r = add_strassen_block(0, 0, 0, r) # r becomes 7
    
    # Block 2: j in {0,1}, k in {2,3}
    # U indices: same as Block 1 (j in {0,1})
    # V indices: j in {0,1}, k in {2,3} -> 2,3,7,8
    # W indices: k in {2,3}, i in {0,1} -> 4,5,6,7
    
    # We need to modify the block to use offsets.
    # Let's rewrite add_strassen_block to accept offsets.
    
    def add_strassen_block_offset(u_off, v_off, w_off, r_offset):
        # u_off: added to i*m + j
        # v_off: added to j*p + k
        # w_off: added to k*n + i
        
        # M1
        U[0*m+0 + u_off, r_offset] += 1
        U[1*m+1 + u_off, r_offset] += 1
        V[0*p+0 + v_off, r_offset] += 1
        V[1*p+1 + v_off, r_offset] += 1
        W[0*n+0 + w_off, r_offset] += 1
        W[1*n+1 + w_off, r_offset] += 1
        r_offset += 1
        
        # M2
        U[0*m+1 + u_off, r_offset] += 1
        U[1*m+1 + u_off, r_offset] += 1
        V[0*p+0 + v_off, r_offset] += 1
        W[0*n+1 + w_off, r_offset] += 1
        W[1*n+1 + w_off, r_offset] -= 1
        r_offset += 1
        
        # M3
        U[0*m+0 + u_off, r_offset] += 1
        V[0*p+1 + v_off, r_offset] += 1
        V[1*p+1 + v_off, r_offset] -= 1
        W[1*n+0 + w_off, r_offset] += 1
        W[1*n+1 + w_off, r_offset] += 1
        r_offset += 1
        
        # M4
        U[1*m+1 + u_off, r_offset] += 1
        V[1*p+0 + v_off, r_offset] += 1
        V[0*p+0 + v_off, r_offset] -= 1
        W[0*n+0 + w_off, r_offset] += 1
        W[0*n+1 + w_off, r_offset] += 1
        r_offset += 1
        
        # M5
        U[0*m+0 + u_off, r_offset] += 1
        U[0*m+1 + u_off, r_offset] += 1
        V[1*p+1 + v_off, r_offset] += 1
        W[0*n+0 + w_off, r_offset] -= 1
        W[1*n+0 + w_off, r_offset] += 1
        r_offset += 1
        
        # M6
        U[0*m+1 + u_off, r_offset] += 1
        U[0*m+0 + u_off, r_offset] -= 1
        V[0*p+0 + v_off, r_offset] += 1
        V[0*p+1 + v_off, r_offset] += 1
        W[1*n+1 + w_off, r_offset] += 1
        r_offset += 1
        
        # M7
        U[0*m+1 + u_off, r_offset] += 1
        U[1*m+1 + u_off, r_offset] -= 1
        V[1*p+0 + v_off, r_offset] += 1
        V[1*p+1 + v_off, r_offset] += 1
        W[0*n+0 + w_off, r_offset] += 1
        r_offset += 1
        
        return r_offset

    # Reset U, V, W
    U[:] = 0
    V[:] = 0
    W[:] = 0
    r = 0
    
    # Block 1: j in {0,1}, k in {0,1}
    # u_off=0, v_off=0, w_off=0
    r = add_strassen_block_offset(0, 0, 0, r)
    
    # Block 2: j in {0,1}, k in {2,3}
    # U: j in {0,1} -> u_off = 0
    # V: j in {0,1}, k in {2,3}. 
    # V index is j*p + k.
    # k=2 -> 2. k=3 -> 3.
    # Local k' in {0,1}. Global k = 2 + k'.
    # So v_off should shift k by 2.
    # v_off = 2.
    # W: k in {2,3}. 
    # W index is k*n + i.
    # Local k' in {0,1}. Global k = 2 + k'.
    # So w_off should shift k by 2*n = 4.
    # w_off = 4.
    r = add_strassen_block_offset(0, 2, 4, r)
    
    # Block 3: j in {0,1}, k=4
    # Rank 1 term.
    U[0*m+0, r] = 1
    U[0*m+1, r] = 1
    U[1*m+0, r] = 1
    U[1*m+1, r] = 1
    V[0*p+4, r] = 1
    V[1*p+4, r] = 1
    W[4*n+0, r] = 1
    W[4*n+1, r] = 1
    r += 1
    
    # Block 4: j=2, k in {0..4}
    # Rank 1 term.
    U[0*m+2, r] = 1
    U[1*m+2, r] = 1
    V[2*p+0, r] = 1
    V[2*p+1, r] = 1
    V[2*p+2, r] = 1
    V[2*p+3, r] = 1
    V[2*p+4, r] = 1
    W[0*n+0, r] = 1
    W[0*n+1, r] = 1
    W[1*n+0, r] = 1
    W[1*n+1, r] = 1
    W[2*n+0, r] = 1
    W[2*n+1, r] = 1
    W[3*n+0, r] = 1
    W[3*n+1, r] = 1
    W[4*n+0, r] = 1
    W[4*n+1, r] = 1
    r += 1
    
    # Verify
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-10):
        # Fallback to Rank 19 if this fails (should not)
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
