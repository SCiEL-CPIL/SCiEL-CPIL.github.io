"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W):
    """
    Verify that the decomposition is exact.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    
    The condition is that for all i, j, k:
    T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Here T[i, j, k] is 1 if we are looking at C[i, k] = A[i, :] * B[:, k].
    In standard matrix mult, C[i, k] = sum_j A[i, j] B[j, k].
    So the tensor T_{ijk} = 1 for all i, j, k? No, the tensor entries are 1 where the
    bilinear form picks out A[i,j]B[j,k].
    Specifically, the multiplication tensor T is defined such that 
    C_{ik} = sum_{j} T_{ijk} A_{ij} B_{jk}.
    Thus T_{ijk} = 1 for all i, j, k.
    
    Wait, let's check the dimensions.
    i in 0..n-1, j in 0..m-1, k in 0..p-1.
    U is indexed by (i, j) -> i*m + j.
    V is indexed by (j, k) -> j*p + k.
    W is indexed by (k, i) -> k*n + i.
    
    We check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k.
    """
    n, m, p = 2, 3, 10
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    R = U.shape[1]
    
    # Reshape
    U3 = U.T.reshape(n, m, R) # [n, m, r]
    V3 = V.T.reshape(m, p, R) # [m, p, r]
    W3 = W.T.reshape(p, n, R) # [p, n, r]
    
    # Compute T_{ijk} = sum_r U_{ijr} V_{jkr} W_{kir}
    # Note: W is indexed by k, i. So W3[k, i, r].
    
    # We can compute this using einsum or explicit loops for verification.
    # Since n=2, m=3, p=10, the loop is small (60 iterations).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(R):
                    val += U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _strassen_block(U, V, W, r, i, j, k):
    """
    Helper to add Strassen terms for a 2x2 block.
    This is not strictly necessary if we hardcode, but good for clarity.
    Actually, the current code hardcodes it. I will stick to the hardcoded approach
    but maybe structure it slightly better or just copy the working logic.
    """
    pass

def construct():
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Decomposition of 2x3x10 into 5 blocks of 2x3x2
    # Each 2x3x2 block is decomposed into rank 11
    # Structure:
    # A = [A_2x2 | A_2x1]
    # B = [B_2x2; B_1x2]
    # C = A_2x2 B_2x2 + A_2x1 B_1x2
    
    for block in range(5):
        k_base = 2 * block
        # Indices for A (2x3)
        # A_2x2: rows 0,1; cols 0,1
        # A_2x1: rows 0,1; col 2
        # Indices for B (3x2) relative to block
        # B_2x2: rows 0,1; cols k_base, k_base+1
        # B_1x2: row 2; cols k_base, k_base+1
        # Indices for C (2x2) relative to block
        # C: rows 0,1; cols k_base, k_base+1
        
        # --- Part 1: A_2x2 * B_2x2 (Rank 7 Strassen) ---
        # Let A_2x2 = [[a00, a01], [a10, a11]]
        # Let B_2x2 = [[b00, b01], [b10, b11]]
        
        # U indices for A_2x2
        u_a00 = 0 * m + 0
        u_a01 = 0 * m + 1
        u_a10 = 1 * m + 0
        u_a11 = 1 * m + 1
        
        # V indices for B_2x2
        v_b00 = 0 * p + k_base
        v_b01 = 0 * p + k_base + 1
        v_b10 = 1 * p + k_base
        v_b11 = 1 * p + k_base + 1
        
        # W indices for C_2x2
        w_c00 = k_base * n + 0
        w_c01 = (k_base + 1) * n + 0
        w_c10 = k_base * n + 1
        w_c11 = (k_base + 1) * n + 1
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[u_a00][r] += 1; U[u_a11][r] += 1
        V[v_b00][r] += 1; V[v_b11][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # M2 = (a10 + a11) * b00
        U[u_a10][r] += 1; U[u_a11][r] += 1
        V[v_b00][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] -= 1
        r += 1
        
        # M3 = a00 * (b01 - b11)
        U[u_a00][r] += 1
        V[v_b01][r] += 1; V[v_b11][r] -= 1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # M4 = a11 * (b10 - b00)
        U[u_a11][r] += 1
        V[v_b10][r] += 1; V[v_b00][r] -= 1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1
        
        # M5 = (a00 + a01) * b11
        U[u_a00][r] += 1; U[u_a01][r] += 1
        V[v_b11][r] += 1
        W[w_c00][r] -= 1; W[w_c01][r] += 1
        r += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[u_a10][r] += 1; U[u_a00][r] -= 1
        V[v_b00][r] += 1; V[v_b01][r] += 1
        W[w_c11][r] += 1
        r += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[u_a01][r] += 1; U[u_a11][r] -= 1
        V[v_b10][r] += 1; V[v_b11][r] += 1
        W[w_c00][r] += 1
        r += 1
        
        # --- Part 2: A_2x1 * B_1x2 (Rank 4 Schoolbook) ---
        # A_2x1 = [[a02], [a12]]
        # B_1x2 = [b20, b21]
        
        u_a02 = 0 * m + 2
        u_a12 = 1 * m + 2
        v_b20 = 2 * p + k_base
        v_b21 = 2 * p + k_base + 1
        
        # C_2x2 additions from this part
        # c00 += a02 * b20
        U[u_a02][r] += 1
        V[v_b20][r] += 1
        W[w_c00][r] += 1
        r += 1
        
        # c01 += a02 * b21
        U[u_a02][r] += 1
        V[v_b21][r] += 1
        W[w_c01][r] += 1
        r += 1
        
        # c10 += a12 * b20
        U[u_a12][r] += 1
        V[v_b20][r] += 1
        W[w_c10][r] += 1
        r += 1
        
        # c11 += a12 * b21
        U[u_a12][r] += 1
        V[v_b21][r] += 1
        W[w_c11][r] += 1
        r += 1
        
    # Verification is optional in production for speed, but good for debugging.
    # The evaluator expects the result directly.
    # We return the dictionary as required.
    return {'U': U, 'V': V, 'W': W}

# EVOLVE-BLOCK-END
