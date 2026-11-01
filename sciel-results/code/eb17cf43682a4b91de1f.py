"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START

def construct():
    # The optimal rank for 2x2 matrix multiplication is 7 (Strassen).
    # Any attempt to find a rank < 7 is mathematically impossible.
    # We return the standard Strassen decomposition directly.
    
    n, m, p = DIMENSIONS
    rank = 7
    
    # U: (n*m, rank) = (4, 7)
    # V: (m*p, rank) = (4, 7)
    # W: (p*n, rank) = (4, 7)
    
    # Indices for U (A matrix):
    # 0: A[0,0]
    # 1: A[0,1]
    # 2: A[1,0]
    # 3: A[1,1]
    
    # Indices for V (B matrix):
    # 0: B[0,0]
    # 1: B[0,1]
    # 2: B[1,0]
    # 3: B[1,1]
    
    # Indices for W (C matrix):
    # Convention: W[k*n+i, r] corresponds to C[i, k] ? 
    # Let's check the standard trace convention mentioned: W[k*n+i, r].
    # Usually C = A * B. C[i, j] = sum_k A[i, k] B[k, j].
    # The tensor is T_{i, j, k} = A_{i, k} B_{k, j}.
    # Decomposition: T = sum_r U_{i, k, r} V_{k, j, r} W_{j, i, r} ?
    # The interface says:
    # U: (n*m, rank) -> indexed by (i, k) or similar? 
    # V: (m*p, rank) -> indexed by (k, j) or similar?
    # W: (p*n, rank) -> indexed by (j, i) or similar?
    
    # Let's look at the current code's indexing to ensure we match the evaluator's expectation.
    # Current code:
    # U[0] = A00, U[1] = A01, U[2] = A10, U[3] = A11. (Row-major A)
    # V[0] = B00, V[1] = B01, V[2] = B10, V[3] = B11. (Row-major B)
    # W[0] = C00, W[1] = C10, W[2] = C01, W[3] = C11.
    # Wait, W[k*n+i] with n=2.
    # k=0, i=0 -> W[0]
    # k=0, i=1 -> W[1]
    # k=1, i=0 -> W[2]
    # k=1, i=1 -> W[3]
    # So W[0] corresponds to index (0,0) in (k,i).
    # If C is output, usually C[i,j].
    # If the tensor is defined as T_{i,k,j} = A_{i,k} B_{k,j}, and we contract to get C_{i,j}.
    # The "official trace convention" W[k*n+i] suggests the output index is mapped to a vector of length p*n.
    # If p=2, n=2.
    # Let's assume the standard Strassen mapping works with the current code's verified score.
    # The current code scored 1.1429, which is the optimal score.
    # Therefore, the mapping in the current code is correct for the evaluator.
    
    # Strassen's 7 products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's construct the matrices U, V, W such that:
    # C_{i,j} = sum_r U_{(i,k), r} V_{(k,j), r} W_{(j,i), r} ? 
    # Actually, the decomposition is usually:
    # T_{i,k,j} = sum_r U_{i,k,r} V_{k,j,r} W_{j,i,r} is not quite right for matrix mult tensor.
    # The matrix multiplication tensor is T_{i,j,k} = A_{i,k} B_{k,j}.
    # We want T_{i,j,k} = sum_r U_{i,k,r} V_{k,j,r} W_{j,i,r} ?? No.
    # Standard tensor decomposition for matmul:
    # T_{i,j,k} = sum_r U_{i,k,r} V_{k,j,r} W_{j,i,r} is not standard.
    # It is T_{i,j,k} = sum_r A_{i,k}^{(r)} B_{k,j}^{(r)} C_{j,i}^{(r)}?
    # No, the output C is not part of the input tensor T.
    # The tensor T defines the bilinear map A, B -> C.
    # T_{i,k,j} is the coefficient of A_{i,k} B_{k,j} in C_{i,j}.
    # So T_{i,j,k} (if we view C_{i,j} as output index) is 1 if we multiply A_{i,k} and B_{k,j}.
    # Wait, C_{i,j} = sum_k A_{i,k} B_{k,j}.
    # So the tensor is T_{i,j,k} where the first two indices are the "input" indices for A and B?
    # No, A has indices (i,k), B has indices (k,j).
    # The tensor is 3-mode: A-index, B-index, C-index.
    # U is (n*m, rank) -> (i,k) pairs.
    # V is (m*p, rank) -> (k,j) pairs.
    # W is (p*n, rank) -> (j,i) pairs? Or (i,j)?
    # The prompt says W[k*n+i]. If p=2, n=2.
    # If W index is (j,i), then j is from B's row (k? no, B is m x p, so rows k, cols j).
    # A is n x m (rows i, cols k).
    # B is m x p (rows k, cols j).
    # C is n x p (rows i, cols j).
    # So C index is (i,j).
    # W[k*n+i] -> if k=j and i=i?
    # If W is indexed by (j,i), then W[j*n+i].
    # j ranges 0..p-1, i ranges 0..n-1.
    # This matches W shape (p*n, rank).
    # So W_{j,i,r} contributes to C_{i,j}.
    
    # Let's verify the current code's W indices against this.
    # Current code:
    # W[0] = C00 -> i=0, j=0. W[0*n+0] = W[0]. Matches.
    # W[1] = C10 -> i=0? No, C10 is i=1, j=0.
    # If W[k*n+i] with k=j, i=i.
    # C10: i=1, j=0. W[0*2+1] = W[1].
    # Current code sets W[1] for M2, M4.
    # Let's check M2: (A10+A11)B00.
    # A10: i=1, k=0. A11: i=1, k=1. B00: k=0, j=0.
    # This product contributes to C_{1,0}.
    # So W[1] should have non-zero entries for terms contributing to C_{1,0}.
    # Current code: M2 has W {1: 1, 3: -1}.
    # Wait, M2 is (A10+A11)*B00.
    # In Strassen:
    # C10 = M2 + M4.
    # M2 = (A10+A11)B00.
    # M4 = A11(B10-B00).
    # So C10 = (A10+A11)B00 + A11(B10-B00) = A10 B00 + A11 B10. Correct.
    # So W[1] (C10) should get +1 from M2 and +1 from M4.
    # Current code M2: W {1: 1, 3: -1}.
    # Current code M4: W {0: 1, 1: 1}.
    # So W[1] gets 1 from M2 and 1 from M4. Total 2?
    # But C10 = A10 B00 + A11 B10.
    # M2 = A10 B00 + A11 B00.
    # M4 = A11 B10 - A11 B00.
    # M2 + M4 = A10 B00 + A11 B10.
    # So yes, W[1] should be 1 (from M2) + 1 (from M4) = 2?
    # No, the decomposition is sum_r U_r V_r W_r.
    # If W[1] for M2 is 1 and W[1] for M4 is 1, the total contribution to C10 is 1*U2V2 + 1*U4V4.
    # This is correct.
    
    # The current code is correct.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen = [
        # M1 = (A00 + A11) * (B00 + B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10 + A11) * B00
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
        # M3 = A00 * (B01 - B11)
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
        # M4 = A11 * (B10 - B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
        # M5 = (A00 + A01) * B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        # M6 = (A10 - A00) * (B00 + B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01 - A11) * (B10 + B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    for r, (u_c, v_c, w_c) in enumerate(strassen):
        for idx, val in u_c.items():
            U[idx][r] = val
        for idx, val in v_c.items():
            V[idx][r] = val
        for idx, val in w_c.items():
            W[idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
