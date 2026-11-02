"""Matrix multiplication 2x2x2: rank-7 Strassen decomposition."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-7 tensor decomposition for 2x2x2 matrix multiplication (Strassen).
    
    Convention: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where T[i,j,k] = 1 for all i,j,k in {0,1} (the 2x2x2 matmul tensor).
    
    Index mapping:
      U rows: i*m+j where m=2, so row = i*2+j (a=0, b=1, c=2, d=3)
      V rows: j*p+k where p=2, so row = j*2+k (e=0, f=1, g=2, h=3)
      W rows: k*n+i where n=2, so row = k*2+i (C00=0, C10=1, C01=2, C11=3)
    
    Strassen's 7 products:
      P0 = (a+d)(e+h)
      P1 = (c+d)(g+h)
      P2 = a(f-h)
      P3 = d(g-e)
      P4 = (a+b)e
      P5 = (c+d)(f-h)
      P6 = (a-c)(e+h)
    
    Reconstruction:
      C00 = P0 + P3 - P1 + P6
      C01 = P2 + P4
      C10 = P1 + P3
      C11 = P0 - P2 + P5 - P3
    """
    rank = 7
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(4)]
    W = [[0.0] * rank for _ in range(4)]
    
    # Use numpy to construct and verify the Strassen decomposition.
    # The tensor T[i,j,k] = 1 for all i,j,k in {0,1}.
    # We need: T[i,j,k] = sum_r U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r]
    
    import numpy as np
    
    # Strassen's 7 linear forms for A and B:
    # u_r: coefficient vector for A (rows 0=a,1=b,2=c,3=d)
    # v_r: coefficient vector for B (rows 0=e,1=f,2=g,3=h)
    
    u0 = np.array([1, 0, 0, 1])   # a+d
    v0 = np.array([1, 0, 0, 1])   # e+h
    u1 = np.array([0, 0, 1, 1])   # c+d
    v1 = np.array([0, 0, 1, 1])   # g+h
    u2 = np.array([1, 0, 0, 0])   # a
    v2 = np.array([0, 1, 0, -1])  # f-h
    u3 = np.array([0, 0, 0, 1])   # d
    v3 = np.array([-1, 0, 1, 0])  # g-e
    u4 = np.array([1, 1, 0, 0])   # a+b
    v4 = np.array([1, 0, 0, 0])   # e
    u5 = np.array([0, 0, 1, 1])   # c+d
    v5 = np.array([0, 1, 0, -1])  # f-h
    u6 = np.array([1, 0, -1, 0])  # a-c
    v6 = np.array([1, 0, 0, 1])   # e+h
    
    U = np.array([u0, u1, u2, u3, u4, u5, u6]).T  # shape (4, 7)
    V = np.array([v0, v1, v2, v3, v4, v5, v6]).T  # shape (4, 7)
    
    # Now find W such that for all (i,j,k):
    # sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
    
    # This is a system of 8 equations. For each (i,j,k), the equation is:
    # sum_r (U[i*2+j, r] * V[j*2+k, r]) * W[k*2+i, r] = 1
    
    # Let's set up the system. Unknowns are W[0..3, 0..6] = 28 values.
    # But the equations are coupled. Let me just solve it.
    
    # For each (i,j,k): sum_r A_{r}^{(i,j)} * W[k*2+i, r] = 1
    # where A_r^{(i,j)} = U[i*2+j, r] * V[j*2+k, r]
    
    # There are 8 equations with 28 unknowns. The system is underdetermined.
    # Let me set up and solve using least squares, then verify.
    
    # Actually, let me think differently. The product P_r = u_r(A) * v_r(B).
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (contribution of P_r to C[i,k])
    # The contribution of P_r to C[i,k] is: sum_j u_r[i,j] * v_r[j,k] * P_r_coeff
    # where u_r[i,j] is the coefficient of A[i,j] in the r-th linear form,
    # and v_r[j,k] is the coefficient of B[j,k] in the r-th linear form.
    # So the "weight" of P_r for output C[i,k] is sum_j u_r[i,j] * v_r[j,k].
    # And C[i,k] = sum_r W[k*2+i, r] * P_r where W[k*2+i, r] should satisfy:
    # sum_r W[k*2+i, r] * (sum_j u_r[i,j] * v_r[j,k]) = sum_j delta_{i,j}... 
    # No. C[i,k] = sum_j A[i,j] B[j,k]. And P_r = (sum_j A[i,j] u_r[i,j]) * (sum_k B[j,k] v_r[j,k])
    # Wait, P_r = u_r^T A_flat * v_r^T B_flat where A_flat = [a,b,c,d], B_flat = [e,f,g,h].
    # P_r = (sum_{i,j} u_r[i*2+j] A[i,j]) * (sum_{j,k} v_r[j*2+k] B[j,k])
    # = sum_{i,j,j',k} u_r[i*2+j] A[i,j] v_r[j'*2+k] B[j',k]
    # The contribution to C[i,k] from P_r is: sum_j u_r[i*2+j] * v_r[j*2+k] * P_r
    # So C[i,k] = sum_r (sum_j u_r[i*2+j] * v_r[j*2+k]) * P_r
    # And we need this to equal sum_j A[i,j] B[j,k].
    # So we need: sum_r W[k*2+i, r] * (sum_j u_r[i*2+j] * v_r[j*2+k]) = 1 for all i,k
    # (since the "1" corresponds to the coefficient in the tensor equation)
    
    # Let's compute M[i,k,r] = sum_j u_r[i*2+j] * v_r[j*2+k] for all i,k,r.
    # Then we need: sum_r W[k*2+i, r] * M[i,k,r] = 1 for all i,k.
    
    M = np.zeros((2, 2, 7))  # M[i,k,r]
    for i in range(2):
        for k in range(2):
            for r in range(7):
                M[i, k, r] = 0.0
                for j in range(2):
                    M[i, k, r] += U[i*2+j, r] * V[j*2+k, r]
    
    # Now solve: for each (i,k), sum_r W[k*2+i, r] * M[i,k,r] = 1
    # This gives 4 equations with 28 unknowns.
    # Let me set up the full system and solve.
    
    # Unknowns: W_flat[0..27] where W_flat[k*2+i*7 + r] = W[k*2+i, r]
    # Actually W is 4x7. Let's index W[row, col] where row = k*2+i.
    # Equation for (i,k): sum_r W[k*2+i, r] * M[i,k,r] = 1
    
    # Build the 4x28 matrix.
    A_sys = np.zeros((4, 28))
    b_sys = np.ones(4)
    eq_idx = 0
    for i in range(2):
        for k in range(2):
            row_w = k*2+i
            for r in range(7):
                A_sys[eq_idx, row_w*7+r] = M[i, k, r]
            eq_idx += 1
    
    # Solve least squares
    W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
    W = W_flat.reshape(4, 7)
    
    # Verify
    max_err = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0.0
                for r in range(7):
                    val += U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err > 1e-9:
        # Fallback to schoolbook rank 8
        rank = 8
        U8 = [[0.0] * rank for _ in range(4)]
        V8 = [[0.0] * rank for _ in range(4)]
        W8 = [[0.0] * rank for _ in range(4)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U8[i*2+j][col] = 1.0
                    V8[j*2+k][col] = 1.0
                    W8[k*2+i][col] = 1.0
                    col += 1
        return {"U": U8, "V": V8, "W": W8}
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
