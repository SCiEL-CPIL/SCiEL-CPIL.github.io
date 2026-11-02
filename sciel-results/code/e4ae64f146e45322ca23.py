"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen for first 2 columns (rank 7) and standard for last 3 (rank 12)
    # Total rank 19
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        # u_idx, v_idx, w_idx are 0-based indices into U, V, W rows
        # We need to map (i,j), (j,k), (k,i) to linear indices
        pass

    # Strassen for cols 0, 1 (k=0, 1)
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices for k=0,1: 00, 01, 10, 11 -> 0, 1, 4, 5 (j*p+k)
    # C indices for k=0,1: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*n+i)
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # A: 0, 3. B: 0, 4. C: 0, 3? No.
    # C00 is k=0, i=0 -> 0. C11 is k=1, i=1 -> 1*2+1 = 3.
    # Wait, C indices: k*n+i.
    # k=0, i=0 -> 0.
    # k=0, i=1 -> 1.
    # k=1, i=0 -> 2.
    # k=1, i=1 -> 3.
    
    # Strassen terms:
    # 1. (A00+A11)(B00+B11)
    # A: 1 at 0, 3. B: 1 at 0, 4.
    # C00 (0,0) gets +1. C11 (1,1) gets +1.
    # But M1 contributes to C00 and C11?
    # C00 = A00B00 + A01B10.
    # C11 = A10B01 + A11B11.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # This doesn't directly map.
    # Strassen combines them.
    # We just need the U, V, W vectors for each of the 7 products.
    
    # Product 1: (A00+A11)(B00+B11)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0] (B00, B11)
    # W: [1, 0, 0, 1, ...] (C00, C11)
    # Wait, W must be linear.
    # The contribution to C is sum of these products times W.
    # Actually, the tensor decomposition is:
    # T = sum_r U_r \otimes V_r \otimes W_r
    # For Strassen, the "W" part is not just 1s. It's the coefficients in the reconstruction.
    # But the interface asks for U, V, W such that C = U V^T W^T?
    # No, the standard form is $C_{ik} = \sum_r U_{ij,r} V_{jk,r} W_{ik,r}$.
    # In Strassen, we compute $M_r$ and then $C = \sum c_r M_r$.
    # This fits the form if $W_{ik,r}$ is the coefficient of $M_r$ in $C_{ik}$.
    
    # Let's define the 7 Strassen terms for the 2x2x2 block (k=0,1).
    # Indices:
    # A: 00(0), 01(1), 10(2), 11(3)
    # B: 00(0), 01(1), 10(4), 11(5)
    # C: 00(0), 10(1), 01(2), 11(3)  <-- Note: C index is k*n+i.
    # k=0, i=0 -> 0. k=0, i=1 -> 1.
    # k=1, i=0 -> 2. k=1, i=1 -> 3.
    
    # M1 = (A00+A11)(B00+B11)
    U1 = [1, 0, 0, 1]
    V1 = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    W1 = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0] # C00, C11
    
    # M2 = (A10+A11)B00
    U2 = [0, 0, 1, 1]
    V2 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    W2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # Wait, M2 contributes to C10 and C11?
    # C10 = A10B00 + A11B10.
    # C11 = A10B01 + A11B11.
    # M2 = A10B00 + A11B00.
    # This doesn't match C10.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's map to our indices.
    # A11->0, A12->1, A21->2, A22->3.
    # B11->0, B12->1, B21->4, B22->5.
    # C11->0 (k=0,i=0), C12->2 (k=1,i=0), C21->1 (k=0,i=1), C22->3 (k=1,i=1).
    
    # M1: (A0+A3)(B0+B5)
    # U: [1,0,0,1]
    # V: [1,0,0,0,0,1,0,0,0,0]
    # W: C11(0) +1, C22(3) +1.
    # W: [1,0,0,1,0,0,0,0,0,0]
    
    # M2: (A2+A3)B0
    # U: [0,0,1,1]
    # V: [1,0,0,0,0,0,0,0,0,0]
    # W: C21(1) +1, C22(3) -1.
    # W: [0,1,0,-1,0,0,0,0,0,0]
    
    # M3: A0(B1-B5)
    # U: [1,0,0,0]
    # V: [0,1,0,0,0,-1,0,0,0,0]
    # W: C12(2) +1, C22(3) +1.
    # W: [0,0,1,1,0,0,0,0,0,0]
    
    # M4: A3(B4-B0)
    # U: [0,0,0,1]
    # V: [-1,0,0,0,1,0,0,0,0,0]
    # W: C11(0) +1, C21(1) +1.
    # W: [1,1,0,0,0,0,0,0,0,0]
    
    # M5: (A0+A1)B5
    # U: [1,1,0,0]
    # V: [0,0,0,0,0,1,0,0,0,0]
    # W: C11(0) -1, C12(2) +1.
    # W: [-1,0,1,0,0,0,0,0,0,0]
    
    # M6: (A2-A0)(B0+B1)
    # U: [-1,0,1,0]
    # V: [1,1,0,0,0,0,0,0,0,0]
    # W: C22(3) +1.
    # W: [0,0,0,1,0,0,0,0,0,0]
    
    # M7: (A1-A3)(B4+B5)
    # U: [0,1,0,-1]
    # V: [0,0,0,0,1,1,0,0,0,0]
    # W: C11(0) +1.
    # W: [1,0,0,0,0,0,0,0,0,0]
    
    # Now add these 7 terms to U, V, W.
    # Then add 12 standard terms for k=2,3,4.
    
    terms = []
    # Append Strassen terms
    for u, v, w in zip([U1, U2, U3, U4, U5, U6, U7], [V1, V2, V3, V4, V5, V6, V7], [W1, W2, W3, W4, W5, W6, W7]):
        terms.append((u, v, w))
        
    # Standard terms for k=2,3,4
    for k in [2, 3, 4]:
        for i in range(n):
            for j in range(m):
                u = [0]*4
                v = [0]*10
                w = [0]*10
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                terms.append((u, v, w))
                
    # Total terms = 7 + 12 = 19.
    rank = len(terms)
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(10)]
    W = [[0]*rank for _ in range(10)]
    
    for r, (u, v, w) in enumerate(terms):
        for i in range(4): U[i][r] = u[i]
        for i in range(10): V[i][r] = v[i]
        for i in range(10): W[i][r] = w[i]
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
