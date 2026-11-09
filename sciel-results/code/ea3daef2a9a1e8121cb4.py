"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    # Reconstruct tensor: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Check against schoolbook: T[i,j,k] should be 1 for all i,j,k (it's the identity-like tensor)
    # Actually, the tensor for matmul is: for each (i,j,k), coefficient is 1
    # So U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] summed over c should equal 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > 1e-10:
                    return False, (i, j, k, s)
    return True, None

def _strassen_222():
    """Return the 7 rank-1 terms for 2x2 @ 2x2 Strassen decomposition.
    Each term is (u, v, w) where u is 2x2 (flattened row-major), v is 2x2, w is 2x2.
    U[i,j], V[j,k], W[k,i] with indices 0,1.
    """
    # Strassen formulas:
    # M1 = (a+d)(e+h): u=[1,0,0,1], v=[1,1,0,0], w=[1,0,0,1]  -> but w needs to map to output
    # Actually let me use the standard:
    # M1 = (a+d)(e+h), contributes to: C11 += M1, C12 -= M5, C21 += M4, C22 += M1
    # Let me just enumerate the 7 terms carefully.
    # 
    # X = [[a,b],[c,d]], Y = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For each Mc, we need u (2x2), v (2x2), w (2x2) such that:
    # Mc = sum_{i,j} u[i,j]*X[i,j] * sum_{j,k} v[j,k]*Y[j,k]
    # and contribution to C[i,k] = Mc * w[k,i] (note: W[k*n+i] convention means W index is k*n+i)
    #
    # Wait, the convention is: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So for 2x2x2: U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    #
    # M1 = (a+d)(e+h) = (X[0,0]+X[1,1])(Y[0,0]+Y[0,1])
    #   u: X[0,0]=1, X[1,1]=1 -> u = [1,0,0,1] (row-major: [0,0],[0,1])
    #   v: Y[0,0]=1, Y[0,1]=1 -> v = [1,1,0,0] (row-major: [0,0],[0,1])
    #   w: contributes +1 to C[0,0] and +1 to C[1,1] -> W[k*2+i]: C[0,0]=W[0*2+0]=W[0], C[1,1]=W[1*2+1]=W[3]
    #   So w[0]=1, w[3]=1, others 0 -> w = [1,0,0,1]
    terms = []
    # M1
    terms.append(([1,0,0,1], [1,1,0,0], [1,0,0,1]))
    # M2 = (c+d)e = (X[1,0]+X[1,1])*Y[0,0]
    #   u: X[1,0]=1, X[1,1]=1 -> u = [0,0,1,1]
    #   v: Y[0,0]=1 -> v = [1,0,0,0]
    #   w: contributes +1 to C[1,0] and +1 to C[1,1] -> W[0*2+1]=W[1], W[1*2+1]=W[3]
    #   w = [0,1,0,1]
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,1]))
    # M3 = a(f-h) = X[0,0]*(Y[0,1]-Y[1,1])
    #   u: X[0,0]=1 -> u = [1,0,0,0]
    #   v: Y[0,1]=1, Y[1,1]=-1 -> v = [0,1,0,-1]
    #   w: contributes +1 to C[0,1] and +1 to C[1,1] -> W[1*2+0]=W[2], W[1*2+1]=W[3]
    #   w = [0,0,1,1]
    terms.append(([1,0,0,0], [0,1,0,-1], [0,0,1,1]))
    # M4 = d(g-e) = X[1,1]*(Y[1,0]-Y[0,0])
    #   u: X[1,1]=1 -> u = [0,0,0,1]
    #   v: Y[1,0]=1, Y[0,0]=-1 -> v = [-1,0,1,0]
    #   w: contributes +1 to C[1,0] and +1 to C[1,1] -> W[0*2+1]=W[1], W[1*2+1]=W[3]
    #   w = [0,1,0,1]
    terms.append(([0,0,0,1], [-1,0,1,0], [0,1,0,1]))
    # M5 = (a+b)h = (X[0,0]+X[0,1])*Y[1,1]
    #   u: X[0,0]=1, X[0,1]=1 -> u = [1,1,0,0]
    #   v: Y[1,1]=1 -> v = [0,0,0,1]
    #   w: contributes -1 to C[0,0] and +1 to C[0,1] -> W[0*2+0]=W[0] (coeff -1), W[1*2+0]=W[2] (coeff +1)
    #   w = [-1,0,1,0]
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
    # M6 = (c-a)(e+f) = (X[1,0]-X[0,0])*(Y[0,0]+Y[0,1])
    #   u: X[1,0]=1, X[0,0]=-1 -> u = [-1,0,1,0]
    #   v: Y[0,0]=1, Y[0,1]=1 -> v = [1,1,0,0]
    #   w: contributes +1 to C[1,1] -> W[1*2+1]=W[3]
    #   w = [0,0,0,1]
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # M7 = (b-d)(g+h) = (X[0,1]-X[1,1])*(Y[1,0]+Y[1,1])
    #   u: X[0,1]=1, X[1,1]=-1 -> u = [0,1,0,-1]
    #   v: Y[1,0]=1, Y[1,1]=1 -> v = [0,0,1,1]
    #   w: contributes +1 to C[0,0] -> W[0*2+0]=W[0]
    #   w = [1,0,0,0]
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    return terms

def _construct_rank_16():
    """Construct rank-16 decomposition for 2x3 @ 3x4 -> 2x4.
    Split m=3 as 2+1:
      C = A[:,:2] @ B[:2,:] + A[:,2] @ B[2,:]
    First part (2x2 @ 2x4): split p=4 as 2+2, use Strassen for each 2x2@2x2.
    Second part (2x1 @ 1x4): rank 2.
    Total: 7 + 7 + 2 = 16.
    """
    n, m, p = 2, 3, 4
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (12, rank)
    # W has shape (p*n, rank) = (8, rank)
    
    strassen_terms = _strassen_222()
    
    # We'll build columns for U, V, W
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Part 1: A[:,:2] @ B[:2,:] -> this is (2x2) @ (2x4)
    # B[:2,:] is 2x4, split into B[:2,:2] and B[:2,2:] (two 2x2 blocks)
    # For block 1 (columns 0,1 of B, i.e., k=0,1):
    #   Strassen on X=A[:,:2] (2x2), Y=B[:2,:2] (2x2)
    #   U indices: i*3+j for i in 0,1 and j in 0,1 -> rows 0,1,3,4
    #   V indices: j*4+k for j in 0,1 and k in 0,1 -> rows 0,1,4,5
    #   W indices: k*2+i for k in 0,1 and i in 0,1 -> rows 0,1,2,3
    for (u, v, w) in strassen_terms:
        U_row = [0]*6
        V_row = [0]*12
        W_row = [0]*8
        # u is 2x2 row-major: u[0]=X[0,0], u[1]=X[0,1], u[2]=X[1,0], u[3]=X[1,1]
        # Map to U[i*3+j]: X[i,j] -> U[i*3+j]
        U_row[0*3+0] = u[0]  # i=0,j=0
        U_row[0*3+1] = u[1]  # i=0,j=1
        U_row[1*3+0] = u[2]  # i=1,j=0
        U_row[1*3+1] = u[3]  # i=1,j=1
        # v is 2x2 row-major: v[0]=Y[0,0], v[1]=Y[0,1], v[2]=Y[1,0], v[3]=Y[1,1]
        # Map to V[j*4+k]: Y[j,k] -> V[j*4+k]
        V_row[0*4+0] = v[0]  # j=0,k=0
        V_row[0*4+1] = v[1]  # j=0,k=1
        V_row[1*4+0] = v[2]  # j=1,k=0
        V_row[1*4+1] = v[3]  # j=1,k=1
        # w is 2x2 row-major: w[0]=W_out[0,0], w[1]=W_out[0,1], w[2]=W_out[1,0], w[3]=W_out[1,1]
        # Map to W[k*2+i]: C[k,i]... wait, C[i,k] with W[k*n+i] = W[k*2+i]
        # w[0] corresponds to C[0,0] -> W[0*2+0] = W[0]
        # w[1] corresponds to C[0,1] -> W[1*2+0] = W[2]
        # w[2] corresponds to C[1,0] -> W[0*2+1] = W[1]
        # w[3] corresponds to C[1,1] -> W[1*2+1] = W[3]
        W_row[0*2+0] = w[0]  # k=0,i=0 -> C[0,0]
        W_row[1*2+0] = w[1]  # k=1,i=0 -> C[0,1]
        W_row[0*2+1] = w[2]  # k=0,i=1 -> C[1,0]
        W_row[1*2+1] = w[3]  # k=1,i=1 -> C[1,1]
        U_cols.append(U_row)
        V_cols.append(V_row)
        W_cols.append(W_row)
    
    # For block 2 (columns 2,3 of B, i.e., k=2,3):
    #   Strassen on X=A[:,:2] (2x2), Y=B[:2,2:4] (2x2)
    #   U indices: same as before
    #   V indices: j*4+k for j in 0,1 and k in 2,3 -> rows 2,3,6,7
    #   W indices: k*2+i for k in 2,3 and i in 0,1 -> rows 4,5,6,7
    for (u, v, w) in strassen_terms:
        U_row = [0]*6
        V_row = [0]*12
        W_row = [0]*8
        U_row[0*3+0] = u[0]
        U_row[0*3+1] = u[1]
        U_row[1*3+0] = u[2]
        U_row[1*3+1] = u[3]
        # Y[j,k] where j in 0,1, k in 2,3 -> V[j*4+k]
        V_row[0*4+2] = v[0]  # j=0,k=2
        V_row[0*4+3] = v[1]  # j=0,k=3
        V_row[1*4+2] = v[2]  # j=1,k=2
        V_row[1*4+3] = v[3]  # j=1,k=3
        # C[i,k] where k in 2,3 -> W[k*2+i]
        W_row[2*2+0] = w[0]  # k=2,i=0
        W_row[3*2+0] = w[1]  # k=3,i=0
        W_row[2*2+1] = w[2]  # k=2,i=1
        W_row[3*2+1] = w[3]  # k=3,i=1
        U_cols.append(U_row)
        V_cols.append(V_row)
        W_cols.append(W_row)
    
    # Part 2: A[:,2] @ B[2,:] -> (2x1) @ (1x4) -> 2x4
    # This is: C[i,k] = A[i,2] * B[2,k] for i in 0,1, k in 0,1,2,3
    # Rank 2: for each i, C[i,:] = A[i,2] * B[2,:]
    # Term 1: i=0: U[0*3+2]=1, V[2*4+k]=B[2,k] for all k, W[k*2+0]=1 for all k
    # Actually, we can do it in 2 terms (one per row of output):
    # Term for i=0: U[2]=1, V[8..11]=1 (B[2,0..3]), W[0,2,4,6]=1 (k*2+0 for k=0,1,2,3)
    # Term for i=1: U[5]=1, V[8..11]=1, W[1,3,5,7]=1 (k*2+1 for k=0,1,2,3)
    U_row = [0]*6
    V_row = [0]*12
    W_row = [0]*8
    U_row[0*3+2] = 1  # A[0,2]
    for k in range(4):
        V_row[2*4+k] = 1  # B[2,k]
        W_row[k*2+0] = 1  # C[0,k]
    U_cols.append(U_row)
    V_cols.append(V_row)
    W_cols.append(W_row)
    
    U_row = [0]*6
    V_row = [0]*12
    W_row = [0]*8
    U_row[1*3+2] = 1  # A[1,2]
    for k in range(4):
        V_row[2*4+k] = 1  # B[2,k]
        W_row[k*2+1] = 1  # C[1,k]
    U_cols.append(U_row)
    V_cols.append(V_row)
    W_cols.append(W_row)
    
    rank = len(U_cols)
    U = U_cols
    V = V_cols
    W = W_cols
    return {"U": U, "V": V, "W": W}

def _construct_rank_24():
    """Schoolbook decomposition of rank 24."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    # Try rank-16 first
    try:
        result = _construct_rank_16()
        valid, info = _verify(result["U"], result["V"], result["W"], n, m, p)
        if valid:
            return result
        else:
            print(f"Rank-16 verification failed: {info}")
    except Exception as e:
        print(f"Rank-16 construction failed: {e}")
    # Fall back to rank-24
    return _construct_rank_24()
# EVOLVE-BLOCK-END
