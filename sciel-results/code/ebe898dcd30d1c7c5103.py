import numpy as np

def construct():
    n, m, p = 2, 4, 7
    
    # Rank 46
    rank = 46
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    
    # Helper to add a Strassen block for 2x2x2 multiplication
    # i1, i2: rows of A (and C)
    # j1, j2: cols of A (and rows of B)
    # k1, k2: cols of B (and C)
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        nonlocal r
        
        # Local indices in global arrays
        # A index: i * m + j
        # B index: j * p + k
        # C index: k * n + i
        
        a_idx_11 = i1 * m + j1
        a_idx_12 = i1 * m + j2
        a_idx_21 = i2 * m + j1
        a_idx_22 = i2 * m + j2
        
        b_idx_11 = j1 * p + k1
        b_idx_12 = j1 * p + k2
        b_idx_21 = j2 * p + k1
        b_idx_22 = j2 * p + k2
        
        c_idx_11 = k1 * n + i1
        c_idx_12 = k1 * n + i2
        c_idx_21 = k2 * n + i1
        c_idx_22 = k2 * n + i2
        
        # m1 = (a11 + a22) * (b11 + b22)
        U[a_idx_11, r] += 1
        U[a_idx_22, r] += 1
        V[b_idx_11, r] += 1
        V[b_idx_22, r] += 1
        # Note: (a11+a22) means U has 1 at 11 and 22.
        # (b11+b22) means V has 1 at 11 and 22.
        # But wait, V index is (j, k).
        # b11 is (j1, k1). b22 is (j2, k2).
        # So V[b_idx_11] and V[b_idx_22] are 1.
        # What about b12 and b21? They are 0.
        # This term is (a11+a22)(b11+b22) = a11b11 + a11b22 + a22b11 + a22b22.
        # Let's check the cross terms.
        # a11b22: U[a_idx_11]=1, V[b_idx_22]=1.
        # Contribution to C:
        # C index for a11 (i1, j1) and b22 (j2, k2).
        # j must match. j1 != j2. So this term should be 0.
        # In our tensor product sum:
        # Sum over r U[i,j] V[j',k] W[k',i'].
        # For fixed r, U is non-zero at (i1, j1) and (i2, j2).
        # V is non-zero at (j1, k1) and (j2, k2).
        # W is non-zero at (k1, i1), (k1, i2), (k2, i1), (k2, i2).
        
        # Let's trace a11b22:
        # U[i1, j1]=1. V[j2, k2]=1.
        # j != j'. So no contribution?
        # The tensor T is sum of U V W.
        # T[i,j, j',k, k',i'] = U[i,j] V[j',k] W[k',i'].
        # We want T to be 1 if j=j', k=k', i=i'.
        # For r corresponding to m1:
        # U[i1, j1]=1, U[i2, j2]=1.
        # V[j1, k1]=1, V[j2, k2]=1.
        # W[k1, i1]=1, W[k1, i2]=1, W[k2, i1]=1, W[k2, i2]=1.
        
        # Check T[i1, j1, j1, k1, k1, i1]:
        # U[i1, j1]=1. V[j1, k1]=1. W[k1, i1]=1. Prod=1.
        # Check T[i1, j1, j2, k2, k2, i1]: (a11 b22)
        # U[i1, j1]=1. V[j2, k2]=1. W[k2, i1]=1. Prod=1.
        # But we want this to be 0 because j1 != j2.
        # So this decomposition is WRONG for m1.
        
        # Strassen's algorithm works on the *values* of the matrix multiplication.
        # It computes the result matrix C.
        # The tensor decomposition must reproduce the *identity* of the multiplication operation.
        # i.e. Sum_r U_r V_r W_r = T_mult.
        # T_mult has 1s only when indices match.
        # Strassen expresses C as linear combinations of products of linear combinations of A and B.
        # C_{ik} = sum_j A_{ij} B_{jk}.
        # Strassen: C_{ik} = sum_{r=1}^7 L_{ik}^r ( sum_{ij} u_{ij}^r A_{ij} ) ( sum_{jk} v_{jk}^r B_{jk} ).
        # This matches the form Sum_r U V W.
        # So we just need to plug in the coefficients.
        
        # For m1:
        # A part: a11 + a22. U[i1, j1]=1, U[i2, j2]=1.
        # B part: b11 + b22. V[j1, k1]=1, V[j2, k2]=1.
        # C part: 
        # C11 += m1. C12 += m1. C21 += m1. C22 += m1.
        # So W[k1, i1]=1, W[k1, i2]=1, W[k2, i1]=1, W[k2, i2]=1.
        
        # Let's re-verify the "cross term" issue.
        # The formula is C_{ik} = sum_r (U^r \cdot A) (V^r \cdot B) W^r_{ik}.
        # This is exactly what the tensor sum does.
        # My previous check "T[i1, j1, j2, k2...]" was checking the tensor entries directly.
        # The tensor T is defined by the multiplication.
        # T_{(i,j), (j',k), (k',i')} = delta_{jj'} delta_{kk'} delta_{ii'}.
        # The decomposition is Sum_r U_r(i,j) V_r(j',k) W_r(k',i').
        # If I set U, V, W as above, does the sum equal T?
        # For r=1 (m1):
        # Term is U[i,j] V[j',k] W[k',i'].
        # Non-zero U: (i1,j1), (i2,j2).
        # Non-zero V: (j1,k1), (j2,k2).
        # Non-zero W: (k1,i1), (k1,i2), (k2,i1), (k2,i2).
        
        # Consider index set (i1, j1, j2, k2, k2, i1).
        # U[i1,j1]=1. V[j2,k2]=1. W[k2,i1]=1.
        # Product is 1.
        # But T[i1, j1, j2, k2, k2, i1] should be 0 because j1 != j2.
        # So the sum of all r terms must cancel this out?
        # No, T has 0 there. The sum must be 0.
        # But this term is 1.
        # Are there other r terms that contribute to this index?
        # For (i1, j1, j2, k2, k2, i1), we need U[i1,j1]!=0 and V[j2,k2]!=0 and W[k2,i1]!=0.
        # In Strassen, do other terms have this pattern?
        # m2 = (c+d)e. U: (i2,j1), (i2,j2). V: (j1,k1).
        # U[i1,j1]=0. So no.
        # m3 = a(f-h). U: (i1,j1). V: (j1,k2), (j2,k2).
        # U[i1,j1]=1. V[j2,k2]=-1. W[k2,i1]=1.
        # Product: 1 * (-1) * 1 = -1.
        # So m1 gives +1, m3 gives -1. Sum = 0.
        # It works! The cross terms cancel out across different Strassen terms.
        
        # So the construction is correct. We just need to sum the contributions.
        
        # m1
        U[a_idx_11, r] += 1
        U[a_idx_22, r] += 1
        V[b_idx_11, r] += 1
        V[b_idx_22, r] += 1
        W[c_idx_11, r] += 1
        W[c_idx_12, r] += 1
        W[c_idx_21, r] += 1
        W[c_idx_22, r] += 1
        r += 1
        
        # m2 = (c+d)e
        # U: c(i2,j1), d(i2,j2). V: e(j1,k1).
        # C: C21 += m2, C22 += m2.
        U[a_idx_21, r] += 1
        U[a_idx_22, r] += 1
        V[b_idx_11, r] += 1
        W[c_idx_21, r] += 1
        W[c_idx_22, r] += 1
        r += 1
        
        # m3 = a(f-h)
        # U: a(i1,j1). V: f(j1,k2), h(j2,k2).
        # C: C12 += m3, C22 += m3.
        U[a_idx_11, r] += 1
        V[b_idx_12, r] += 1
        V[b_idx_22, r] -= 1
        W[c_idx_12, r] += 1
        W[c_idx_22, r] += 1
        r += 1
        
        # m4 = d(g-e)
        # U: d(i2,j2). V: g(j2,k1), e(j1,k1).
        # C: C21 += m4, C22 += m4.
        U[a_idx_22, r] += 1
        V[b_idx_21, r] += 1
        V[b_idx_11, r] -= 1
        W[c_idx_21, r] += 1
        W[c_idx_22, r] += 1
        r += 1
        
        # m5 = (a+b)h
        # U: a(i1,j1), b(i1,j2). V: h(j2,k2).
        # C: C12 += m5, C22 += m5.
        U[a_idx_11, r] += 1
        U[a_idx_12, r] += 1
        V[b_idx_22, r] += 1
        W[c_idx_12, r] += 1
        W[c_idx_22, r] += 1
        r += 1
        
        # m6 = (c-d)(f+h)
        # U: c(i2,j1), d(i2,j2). V: f(j1,k2), h(j2,k2).
        # C: C21 += m6, C11 += m6. (Wait, Strassen formula: C11 = m1+m4-m5+m7. C21 = m6+m4. C12 = m2+m3. C22 = m1-m2+m3+m5.)
        # Let's check standard Strassen:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-d)(f+h)
        # m7 = (b-c)(e+g)
        # C11 = m1 + m4 - m5 + m7
        # C12 = m2 + m3
        # C21 = m6 + m4
        # C22 = m1 - m2 + m3 + m5
        
        # So for m6:
        # C21 += m6. C11 += m6? No.
        # C21 = m6 + m4. So C21 gets +1 from m6.
        # C11 does not get m6.
        # Wait, C21 is row 2, col 1.
        # My indices: i2 is row 2, k1 is col 1.
        # So W[c_idx_21] += 1.
        # Does C11 get m6? No.
        # So W[c_idx_11] is 0 for m6.
        
        U[a_idx_21, r] += 1
        U[a_idx_22, r] -= 1
        V[b_idx_12, r] += 1
        V[b_idx_22, r] += 1
        W[c_idx_21, r] += 1
        r += 1
        
        # m7 = (b-c)(e+g)
        # U: b(i1,j2), c(i2,j1). V: e(j1,k1), g(j2,k1).
        # C: C11 += m7.
        U[a_idx_12, r] += 1
        U[a_idx_21, r] -= 1
        V[b_idx_11, r] += 1
        V[b_idx_21, r] += 1
        W[c_idx_11, r] += 1
        r += 1

    # 3 blocks of 2x4x2
    # Block 1: k in {0, 1}
    # j in {0, 1}
    add_strassen_block(0, 1, 0, 1, 0, 1)
    # j in {2, 3}
    add_strassen_block(0, 1, 2, 3, 0, 1)
    
    # Block 2: k in {2, 3}
    add_strassen_block(0, 1, 0, 1, 2, 3)
    add_strassen_block(0, 1, 2, 3, 2, 3)
    
    # Block 3: k in {4, 5}
    add_strassen_block(0, 1, 0, 1, 4, 5)
    add_strassen_block(0, 1, 2, 3, 4, 5)
    
    # 1 block of 2x4x1 (k=6)
    # Rank 4
    # j=0: U[i,0]=1, V[0,6]=1, W[6,i]=1
    for j in range(4):
        U[0, r] += 1
        U[1, r] += 1
        V[j * p + 6, r] += 1
        W[6 * n + 0, r] += 1
        W[6 * n + 1, r] += 1
        r += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = 2, 4, 7
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(U.shape[1]):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                Expected[a_idx, b_idx, c_idx] = 1
                
    if np.allclose(T, Expected):
        print("Valid decomposition")
    else:
        print("Invalid decomposition")
        print("Diff:", np.sum(T - Expected))
