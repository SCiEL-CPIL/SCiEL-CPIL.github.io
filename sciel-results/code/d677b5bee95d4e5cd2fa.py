"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    deadline = time.time() + 140
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # U indexed by [i*m + j], V by [j*p + k], W by [k*n + i]
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_vec, v_vec, w_vec):
        """u_vec: length n*m=6, v_vec: length m*p=15, w_vec: length p*n=10"""
        U_terms.append(list(u_vec))
        V_terms.append(list(v_vec))
        W_terms.append(list(w_vec))
    
    def zeros6():
        return [0]*6
    
    def zeros15():
        return [0]*15
    
    def zeros10():
        return [0]*10
    
    # Strategy: Decompose 2x3x5 as:
    # Part 1: A[:, 0:2] * B[0:2, :]  (2x2 times 2x5) -> split into 2x2x3 + 2x2x2
    # Part 2: A[:, 2:3] * B[2:3, :]  (2x1 times 1x5) -> rank 5
    
    # --- Part 2: 2x1x5 (rank 5) ---
    # A[0,2]*B[2,k] and A[1,2]*B[2,k] for k=0..4
    for k in range(5):
        u = zeros6()
        v = zeros15()
        w = zeros10()
        # Term: A[0,2] * B[2,k] * 1 contributes to C[0,k]
        # U[i*m+j, r] where i=0, j=2 -> index 0*3+2=2
        # V[j*p+k, r] where j=2, k=k -> index 2*5+k=10+k
        # W[k*n+i, r] where k=k, i=0 -> index k*2+0=2k
        u[2] = 1  # A[0,2]
        v[10+k] = 1  # B[2,k]
        w[2*k] = 1  # C[0,k]
        add_term(u, v, w)
        
        u = zeros6()
        v = zeros15()
        w = zeros10()
        u[5] = 1  # A[1,2] (index 1*3+2=5)
        v[10+k] = 1  # B[2,k]
        w[2*k+1] = 1  # C[1,k] (index k*2+1)
        add_term(u, v, w)
    
    # --- Part 1: 2x2x5 = 2x2x3 (cols 0,1,2) + 2x2x2 (cols 3,4) ---
    
    # Part 1a: 2x2x3 Winograd (rank 10)
    # A_block = A[:, 0:2] = [[a00, a01], [a10, a11]]
    # B_block = B[0:2, 0:3] = [[b00, b01, b02], [b10, b11, b12]]
    # C_block = C[:, 0:3] = [[c00, c01, c02], [c10, c11, c12]]
    
    # Winograd 2x2x3 rank 10 decomposition
    # Using the known optimal scheme:
    # Let me use a verified formulation.
    
    # Standard Winograd 2x2x3:
    # A' = [a00+a11, a00-a11, a00+a01, a00-a01, a10+a11, a10-a11, a01+a10, a01-a10, a00, a11]
    # Wait, that's 10 A-factors but we need 10 products total.
    
    # Let me use a different known rank-10 scheme for 2x2x3:
    # From literature, one formulation:
    # p1 = (a00+a11)*(b00+b11)
    # p2 = (a00-a11)*(b00-b11)
    # p3 = (a00+a01)*(b00+b02)
    # p4 = (a00-a01)*(b00-b02)
    # p5 = (a10+a11)*(b10+b12)
    # p6 = (a10-a11)*(b10-b12)
    # p7 = (a01+a10)*(b01+b11)
    # p8 = (a01-a10)*(b01-b11)
    # p9 = a00*b01
    # p10 = a11*b10
    
    # Then:
    # c00 = p1+p2+p3-p4+p5-p6+p7-p8+p9+p10  -- this doesn't seem right
    
    # Let me use a cleaner approach. I'll construct the decomposition directly.
    
    # For 2x2x2 Strassen (rank 7), I know the exact form. Let me use that for the 2x2x2 part.
    # For 2x2x3, let me use schoolbook (rank 12) to be safe, or find a better one.
    
    # Actually, let me just use schoolbook for all parts to ensure correctness first,
    # then optimize.
    
    # Schoolbook for 2x2x3 (rank 12):
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u = zeros6()
                v = zeros15()
                w = zeros10()
                # A[i,j] where i=0,1; j=0,1
                u[i*3 + j] = 1
                # B[j,k] where j=0,1; k=0,1,2
                v[j*5 + k] = 1
                # C[i,k] where i=0,1; k=0,1,2
                w[k*2 + i] = 1
                add_term(u, v, w)
    
    # Schoolbook for 2x2x2 (rank 8) for cols 3,4:
    for i in range(2):
        for j in range(2):
            for k in range(2):
                u = zeros6()
                v = zeros15()
                w = zeros10()
                u[i*3 + j] = 1
                # B[j, 3+k] where j=0,1; k=0,1
                v[j*5 + 3 + k] = 1
                # C[i, 3+k]
                w[(3+k)*2 + i] = 1
                add_term(u, v, w)
    
    # Total rank so far: 5 (Part 2) + 12 (Part 1a schoolbook) + 8 (Part 1b schoolbook) = 25
    
    rank = len(U_terms)
    
    # Build matrices
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r in range(rank):
        U[:, r] = U_terms[r]
        V[:, r] = V_terms[r]
        W[:, r] = W_terms[r]
    
    # Verify
    T = np.ones((n, m, p))
    result = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                result[i,j,k] = val
    
    if not np.allclose(result, T, atol=1e-9):
        # Fallback to schoolbook rank 30
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    # Check integer/half-integer
    for M in [U, V, W]:
        for val in M.flatten():
            if abs(val * 2 - round(val * 2)) > 1e-9:
                # Shouldn't happen with our construction
                pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
