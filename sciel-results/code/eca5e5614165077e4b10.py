"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    
    # Base Case 1: 2x2x2 Strassen Decomposition (Rank 7)
    # Dimensions: n1=2, m1=2, p1=2
    n1, m1, p1 = 2, 2, 2
    rank1 = 7
    
    U1 = [[0]*rank1 for _ in range(n1*m1)]
    V1 = [[0]*rank1 for _ in range(m1*p1)]
    W1 = [[0]*rank1 for _ in range(p1*n1)]
    
    # Strassen factors for 2x2x2
    # (U_entries, V_entries, W_entries)
    # U: (i, j, coeff), V: (j, k, coeff), W: (k, i, coeff)
    strassen_factors = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    for s in range(rank1):
        u_entries, v_entries, w_entries = strassen_factors[s]
        for (i, j, coeff) in u_entries:
            U1[i*m1 + j][s] = coeff
        for (j, k, coeff) in v_entries:
            V1[j*p1 + k][s] = coeff
        for (k, i, coeff) in w_entries:
            W1[k*n1 + i][s] = coeff
            
    # Base Case 2: 1x1x8 Identity Decomposition (Rank 8)
    # Dimensions: n2=1, m2=1, p2=8
    n2, m2, p2 = 1, 1, 8
    rank2 = p2
    
    U2 = [[0]*rank2 for _ in range(n2*m2)]
    V2 = [[0]*rank2 for _ in range(m2*p2)]
    W2 = [[0]*rank2 for _ in range(p2*n2)]
    
    for t in range(rank2):
        U2[0][t] = 1
        V2[t][t] = 1 # Row index in V is j*p2 + k. j=0, k=t. So V2[t][t]=1
        W2[t][t] = 1 # Row index in W is k*n2 + i. k=t, i=0. So W2[t][0]=1? 
                     # Wait. W shape is (p*n, rank).
                     # For 1x1x8: A(1x1) * B(1x8) = C(8x1).
                     # C[k, 0] = A[0,0] * B[0,k].
                     # Tensor: T[k, 0, 0, k] = 1? No.
                     # T[A_idx, B_idx, C_idx] = delta.
                     # A has 1 element. B has 8 elements. C has 8 elements.
                     # C_k = A_0 B_k.
                     # Decomposition: sum_r U[A,r] V[B,r] W[C,r]
                     # A is (1x1). B is (1x8). C is (8x1).
                     # U shape (1, 8). V shape (8, 8). W shape (8, 8).
                     # U[0, r] = 1 for all r.
                     # V[j, r] = delta(j, r). (j is 0..7, r is 0..7)
                     # W[k, r] = delta(k, r). (k is 0..7, r is 0..7)
                     # Check: sum_r 1 * delta(j,r) * delta(k,r) = delta(j,k).
                     # C_k = sum_r U[0,r] V[0,r] W[k,r]? No.
                     # C[k, i] = sum_r U[i, j, r] ... 
                     # Standard form: C_{k, i} = sum_{j, r} A_{i, j} U_{i, j, r} V_{j, k, r} W_{k, i, r}
                     # Here i=0, j=0.
                     # C_{k, 0} = sum_r A_{0, 0} U_{0, 0, r} V_{0, k, r} W_{k, 0, r}
                     # We want C_{k, 0} = A_{0, 0} B_{0, k}.
                     # So sum_r U_{0, 0, r} V_{0, k, r} W_{k, 0, r} = B_{0, k}.
                     # This requires V to carry the B index?
                     # In the previous block code:
                     # V row = j*p + k. Here j=0, p=8. Row = k.
                     # W row = k*n + i. Here n=1, i=0. Row = k.
                     # So V[0, k, r] and W[k, 0, r].
                     # If U[0, 0, r] = 1.
                     # V[0, k, r] = delta(k, r).
                     # W[k, 0, r] = 1.
                     # Sum_r 1 * delta(k, r) * 1 = 1 if k=r else 0.
                     # This gives C_{k, 0} = B_{0, k} * 1? No.
                     # The term is A * U * V * W.
                     # If V[0, k, r] = delta(k, r), then V selects the component of B?
                     # No, V multiplies B.
                     # The decomposition is:
                     # C = sum_r (A * U_r) * (V_r * B) * W_r ... no.
                     # It's a bilinear form.
                     # C_{k, i} = sum_r (sum_j A_{i, j} U_{i, j, r}) (sum_l B_{j, l} V_{j, l, r}) W_{l, i, r}?
                     # The indices in the provided interface:
                     # U: (n*m, rank). Index (i*m+j, r).
                     # V: (m*p, rank). Index (j*p+k, r).
                     # W: (p*n, rank). Index (k*n+i, r).
                     # Formula: C[k, i] = sum_{j, r} A[i, j] U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                     
                     # For 1x1x8:
                     # n=1, m=1, p=8.
                     # i=0, j=0.
                     # C[k, 0] = sum_r A[0, 0] U[0, r] V[k, r] W[k, r]
                     # We want C[k, 0] = A[0, 0] B[0, k].
                     # This implies the sum over r must produce B[0, k].
                     # But B is an input. The formula sums A and B elements separately?
                     # No, the formula above is wrong for matrix mult.
                     # Correct tensor decomposition:
                     # C_{i, k} = sum_{j, r} A_{i, j} B_{j, k} T_{i, j, k, r}
                     # The interface U, V, W is a rank decomposition of the tensor T.
                     # T_{i, j, k} = sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r}
                     # Wait, the W index in the interface is (k*n+i).
                     # So W_{k, i, r} is stored at row k*n+i.
                     # So T_{i, j, k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
                     
                     # Let's check 1x1x8 with this:
                     # i=0, j=0.
                     # T_{0, 0, k} = sum_r U[0, r] V[k, r] W[k, r].
                     # We want T_{0, 0, k} = delta(0,0) delta(0,k)? No.
                     # Matrix mult C = A B.
                     # C_{i, k} = sum_j A_{i, j} B_{j, k}.
                     # So the tensor T_{i, j, k} is 1 if the indices are consistent with the multiplication?
                     # No, the tensor *is* the multiplication map.
                     # T_{i, j, k} = 1 if we are computing C_{i, k} from A_{i, j} and B_{j, k}.
                     # For 1x1x8:
                     # A is 1x1. B is 1x8. C is 8x1.
                     # C_{k, 0} = A_{0, 0} B_{0, k}.
                     # So T_{0, 0, k} = 1 for all k?
                     # No, T_{i, j, k} corresponds to the coefficient of A_{i, j} B_{j, k} in C_{i', k'}.
                     # Actually, the standard definition is:
                     # C_{i, k} = sum_{j} A_{i, j} B_{j, k}.
                     # This can be written as C_{i, k} = sum_{j, r} A_{i, j} B_{j, k} \delta_{i, i'} \delta_{k, k'}?
                     # The tensor T has indices (i, j, k).
                     # T_{i, j, k} = 1 if this term contributes to C_{i, k}.
                     # For 1x1x8:
                     # i=0, j=0.
                     # C_{k, 0} gets contribution from A_{0, 0} B_{0, k}.
                     # So T_{0, 0, k} = 1 for all k=0..7.
                     # Let's check the decomposition:
                     # T_{0, 0, k} = sum_r U[0, r] V[k, r] W[k, r] (using row indices)
                     # U[0, r] = 1.
                     # V[k, r] = ?
                     # W[k, r] = ?
                     # If we set V[k, r] = delta(k, r) and W[k, r] = 1.
                     # Sum_r 1 * delta(k, r) * 1 = 1.
                     # This works!
                     # So:
                     # U2[0, r] = 1 for r=0..7.
                     # V2[k, r] = 1 if k==r else 0. (Row k in V is j*p+k = 0*8+k = k).
                     # W2[k, r] = 1. (Row k in W is k*n+i = k*1+0 = k).
                     
    for r in range(rank2):
        U2[0][r] = 1
        V2[r][r] = 1
        W2[r][r] = 1 # Wait, W2 row is k. Col is r. W2[k, r].
                     # In loop above: W2[t][t] = 1. t is row.
                     # So W2[k, k] = 1.
                     # This matches W[k, r] = delta(k, r)?
                     # My previous derivation: W[k, r] = 1.
                     # If W[k, r] = 1, then W2[k, r] = 1 for all r.
                     # If W[k, r] = delta(k, r), then W2[k, r] = 1 if k==r.
                     # Let's re-evaluate.
                     # T_{0, 0, k} = 1.
                     # Sum_r U[0, r] V[k, r] W[k, r] = 1.
                     # If U=1, V=I, W=I -> Sum_r delta(k, r) delta(k, r) = 1.
                     # If U=1, V=I, W=J (all ones) -> Sum_r delta(k, r) * 1 = 1.
                     # Both work. The "Identity" tensor usually implies diagonal.
                     # Let's use V=I, W=I.
                     # V2[k, r] = delta(k, r).
                     # W2[k, r] = delta(k, r).
                     
    # Correction to W2 filling:
    for r in range(rank2):
        W2[r][r] = 1 # This sets W2[k, r] = 1 if k=r.
        
    # Combine using Tensor Product (Kronecker)
    # Rank = rank1 * rank2 = 56
    total_rank = rank1 * rank2
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    for r1 in range(rank1):
        for r2 in range(rank2):
            r = r1 * rank2 + r2
            
            # U: (n1*m1) x (n2*m2) -> (n1*n2) x (m1*m2) ?
            # n = n1*n2 = 2*1 = 2.
            # m = m1*m2 = 2*1 = 2.
            # U_total[i1*m1 + i2*m2? No.
            # Indexing in U is i*m + j.
            # i = i1*n2 + i2?
            # In Kronecker product of matrices A (n1 x m1) and B (n2 x m2):
            # (A \otimes B)_{i1*n2+i2, j1*m2+j2} = A_{i1, j1} B_{i2, j2}.
            # Here U1 is (n1*m1) x rank1. Reshape to (n1, m1, rank1).
            # U2 is (n2*m2) x rank2. Reshape to (n2, m2, rank2).
            # U_total is (n, m, rank).
            # U_total[i1, j1, i2, j2, r1, r2] = U1[i1, j1, r1] * U2[i2, j2, r2].
            # Flatten i = i1*n2 + i2, j = j1*m2 + j2.
            # Row = i*m + j = (i1*n2 + i2) * (m1*m2) + j1*m2 + j2.
            # Since n2=1, m2=1:
            # Row = (i1*1 + i2) * (2*1) + j1*1 + j2 = (i1 + 0) * 2 + j1 + 0 = 2*i1 + j1.
            # This matches i*m + j for n=2, m=2.
            
            for i1 in range(n1):
                for j1 in range(m1):
                    val1 = U1[i1*m1 + j1][r1]
                    if val1 == 0: continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            val2 = U2[i2*m2 + j2][r2]
                            if val2 == 0: continue
                            row = (i1 * n2 + i2) * m + (j1 * m2 + j2)
                            U[row][r] += val1 * val2
                            
            # V: (m1*p1) x (m2*p2)
            # Index j*p + k.
            # j = j1*m2 + j2? No.
            # m = m1*m2. p = p1*p2.
            # V1 index: j1*p1 + k1.
            # V2 index: j2*p2 + k2.
            # V_total index: j*p + k = (j1*m2 + j2) * (p1*p2) + k1*p2 + k2.
            # m2=1, p2=8.
            # j = j1*1 + j2 = j1. (j2 is 0).
            # k = k1*8 + k2.
            # Row = j1 * 16 + k1*8 + k2.
            
            for j1 in range(m1):
                for k1 in range(p1):
                    val1 = V1[j1*p1 + k1][r1]
                    if val1 == 0: continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            val2 = V2[j2*p2 + k2][r2]
                            if val2 == 0: continue
                            row = (j1 * m2 + j2) * p + (k1 * p2 + k2)
                            V[row][r] += val1 * val2
                            
            # W: (p1*n1) x (p2*n2)
            # Index k*n + i.
            # W1 index: k1*n1 + i1.
            # W2 index: k2*n2 + i2.
            # W_total index: k*n + i = (k1*p2 + k2) * (n1*n2) + i1*n2 + i2.
            # n2=1.
            # k = k1*8 + k2.
            # i = i1*1 + i2 = i1.
            # Row = (k1*8 + k2) * 2 + i1.
            
            for k1 in range(p1):
                for i1 in range(n1):
                    val1 = W1[k1*n1 + i1][r1]
                    if val1 == 0: continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            val2 = W2[k2*n2 + i2][r2]
                            if val2 == 0: continue
                            row = (k1 * p2 + k2) * n + (i1 * n2 + i2)
                            W[row][r] += val1 * val2
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
