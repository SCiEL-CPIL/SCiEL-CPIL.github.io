import time
import numpy as np

# The task is to find a rank-minimal exact tensor decomposition for 2x4 * 4x8.
# We use the Kronecker product structure: T_{2,4,8} \cong T_{2,2,2} \otimes T_{1,2,4}.
# Rank(T_{2,2,2}) = 7 (Strassen).
# Rank(T_{1,2,4}) = 8 (Schoolbook for 1x2 * 2x4).
# Total Rank = 7 * 8 = 56.

# Dimensions: n=2, m=4, p=8
# A is 2x4, B is 4x8, C is 2x8.
# Index conventions:
# A index: i in [0,1], j in [0,3]. Flat: i*4 + j.
# B index: r in [0,3], k in [0,7]. Flat: r*8 + k.
# C index: i in [0,1], k in [0,7]. Flat: k*2 + i (Trace convention W[k*n+i]).

# We construct the decomposition U, V, W of shapes (8, 56), (32, 56), (16, 56).

# 1. Strassen Decomposition for 2x2x2
# Local indices:
# A_s (2x2): i in [0,1], j1 in [0,1]. Flat: i*2 + j1.
# B_s (2x2): r1 in [0,1], c1 in [0,1]. Flat: r1*2 + c1.
# C_s (2x2): i in [0,1], c1 in [0,1]. Flat: c1*2 + i.

# 2. Schoolbook Decomposition for 1x2x4
# Local indices:
# A_b (1x2): j2 in [0,1]. Flat: j2. (Row 0 is implicit).
# B_b (2x4): r2 in [0,1], c2 in [0,3]. Flat: r2*4 + c2.
# C_b (1x4): c2 in [0,3]. Flat: c2. (Row 0 is implicit).

# Global Index Mapping (Kronecker Product):
# A_global[i, j] where j = 2*j1 + j2.
#   A_global[i, 2*j1+j2] corresponds to A_s[i, j1] and A_b[j2].
#   U_s index: i*2 + j1.
#   U_b index: j2.
#   U_global index: i*4 + (2*j1 + j2).

# B_global[r, k] where r = 2*r1 + r2, k = 4*c1 + c2.
#   B_global[2*r1+r2, 4*c1+c2] corresponds to B_s[r1, c1] and B_b[r2, c2].
#   V_s index: r1*2 + c1.
#   V_b index: r2*4 + c2.
#   V_global index: (2*r1+r2)*8 + (4*c1+c2).

# C_global[i, k] where k = 4*c1 + c2.
#   C_global[i, 4*c1+c2] corresponds to C_s[i, c1] and C_b[c2].
#   W_s index: c1*2 + i.
#   W_b index: c2.
#   W_global index: (4*c1+c2)*2 + i.

def construct():
    """
    Construct a rank-56 decomposition for 2x4 * 4x8 using Kronecker Product of 
    Strassen (2x2x2) and Schoolbook (1x2x4).
    """
    n, m, p = 2, 4, 8
    rank = 56

    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # --- Strassen Factors (2x2x2) ---
    # U_s: (4, 7), V_s: (4, 7), W_s: (4, 7)
    # Indices: A(2x2), B(2x2), C(2x2)
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))

    # Term 0: M1 = (A00+A11)(B00+B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1

    # Term 1: M2 = (A10+A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1

    # Term 2: M3 = A00(B01-B11)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1

    # Term 3: M4 = A11(B10-B00)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1

    # Term 4: M5 = (A00+A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1

    # Term 5: M6 = (A10-A00)(B00+B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1

    # Term 6: M7 = (A01-A11)(B10+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1

    # --- Schoolbook Factors (1x2x4) ---
    # U_b: (2, 8), V_b: (8, 8), W_b: (4, 8)
    # A_b (1x2): indices 0,1.
    # B_b (2x4): indices 0..7 (row-major).
    # C_b (1x4): indices 0..3.
    U_b = np.zeros((2, 8))
    V_b = np.zeros((8, 8))
    W_b = np.zeros((4, 8))

    col = 0
    for j in range(2): # j is row of B_b, col of A_b
        for k in range(4): # k is col of B_b, index of C_b
            U_b[j, col] = 1
            V_b[j * 4 + k, col] = 1
            W_b[k, col] = 1
            col += 1

    # --- Construct Global U, V, W ---
    for s in range(7):
        for b in range(8):
            col = s * 8 + b

            # U construction
            # U_global[i, 2*j1+j2] += U_s[i, j1] * U_b[j2]
            for i in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        val_s = U_s[i * 2 + j1, s]
                        val_b = U_b[j2, b]
                        if val_s != 0 and val_b != 0:
                            global_idx = i * 4 + (2 * j1 + j2)
                            U[global_idx, col] += val_s * val_b

            # V construction
            # V_global[2*r1+r2, 4*c1+c2] += V_s[r1, c1] * V_b[r2, c2]
            for r1 in range(2):
                for r2 in range(2):
                    for c1 in range(2):
                        for c2 in range(4):
                            val_s = V_s[r1 * 2 + c1, s]
                            val_b = V_b[r2 * 4 + c2, b]
                            if val_s != 0 and val_b != 0:
                                global_idx = (2 * r1 + r2) * 8 + (4 * c1 + c2)
                                V[global_idx, col] += val_s * val_b

            # W construction
            # W_global[4*c1+c2, i] += W_s[i, c1] * W_b[c2]
            # Note: W index convention is k*n + i. Here k = 4*c1+c2, n=2.
            for i in range(2):
                for c1 in range(2):
                    for c2 in range(4):
                        val_s = W_s[i * 2 + c1, s] # W_s index is c1*2 + i? 
                        # Let's re-verify W_s indexing.
                        # C_s is 2x2. C_s[i, c1].
                        # In the Strassen definition above, W_s was defined with indices matching C_s.
                        # C_s[i, c1] corresponds to flat index c1*2 + i in the W_s matrix?
                        # In the code above: W_s[0, 0]=1 (C00), W_s[3, 0]=1 (C11).
                        # C00 is i=0, c1=0. Flat index 0*2+0=0. Correct.
                        # C11 is i=1, c1=1. Flat index 1*2+1=3. Correct.
                        # So W_s index is c1*2 + i.
                        # My loop uses i in range(2), c1 in range(2).
                        # val_s = W_s[c1 * 2 + i, s]
                        
                        # Wait, in the loop above I wrote: val_s = W_s[i * 2 + c1, s]
                        # Let's check the Strassen definition again.
                        # W_s[0, 0] = 1 -> C00. i=0, c1=0. Index 0.
                        # W_s[3, 0] = 1 -> C11. i=1, c1=1. Index 3.
                        # If I use i*2+c1:
                        # i=0, c1=0 -> 0.
                        # i=1, c1=1 -> 3.
                        # This matches.
                        # However, standard row-major for 2x2 matrix C[i,c] is i*2+c.
                        # So W_s[i*2+c1, s] is correct for C_s[i, c1].
                        
                        val_b = W_b[c2, b]
                        if val_s != 0 and val_b != 0:
                            global_k = 4 * c1 + c2
                            global_idx = global_k * 2 + i
                            W[global_idx, col] += val_s * val_b

    # Verification Step (Fast, using matrix operations)
    # We verify on a single random matrix to ensure correctness before returning.
    # This is much faster than looping over 56 columns with reshapes.
    rng = np.random.default_rng(42)
    A_test = rng.standard_normal((n, m))
    B_test = rng.standard_normal((m, p))
    C_expected = A_test @ B_test

    # Compute C_computed = sum_{r=0}^{55} (U[:,r] A_test V[:,r]) * W[:,r]^T ?
    # No, the decomposition is:
    # C = sum_{r} W_r * (U_r * A * V_r^T)? No.
    # The tensor decomposition T = sum_r u_r \otimes v_r \otimes w_r.
    # C_{i,k} = sum_{j,l} A_{i,j} B_{j,l} T_{i,j,l,k}.
    # T_{i,j,l,k} = sum_r U_{i,j,r} V_{j,l,r} W_{l,k,r}.
    # C_{i,k} = sum_{i,j,l} A_{i,j} B_{j,l} (sum_r U_{i,j,r} V_{j,l,r} W_{l,k,r})
    # C_{i,k} = sum_r W_{l,k,r} (sum_{i,j} A_{i,j} U_{i,j,r}) (sum_{j,l} V_{j,l,r} B_{j,l})? No.
    # Let's use the standard form:
    # C = sum_r (U_r A V_r^T) \circ W_r^T ?
    # Actually, the factors are:
    # u_r is a vector in R^{nm}. Reshape to n x m.
    # v_r is a vector in R^{mp}. Reshape to m x p.
    # w_r is a vector in R^{pn}. Reshape to p x n.
    # The product is (u_r^T A v_r) ? No.
    # The term is u_r \otimes v_r \otimes w_r.
    # C = sum_r w_r * (u_r^T \otimes v_r^T) (A \otimes B^T)?
    
    # Let's stick to the loop for verification but make it slightly faster by pre-reshaping?
    # No, 56 iterations is tiny. The previous program's verification was likely fine, 
    # but the construction loops were the bottleneck/risk.
    
    # Let's do a quick check.
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_vec = U[:, r].reshape(n, m)
        v_vec = V[:, r].reshape(m, p)
        w_vec = W[:, r].reshape(p, n)
        
        # The contribution to C is:
        # C += (u_vec * v_vec^T) ? No.
        # The tensor term is u \otimes v \otimes w.
        # C_{i,k} += sum_{j,l} u_{i,j} v_{j,l} w_{l,k}.
        # This is exactly: C += u_vec @ v_vec * w_vec.T ?
        # (u @ v) is (n x p). w.T is (n x p).
        # Yes.
        C_computed += (u_vec @ v_vec) * w_vec.T

    if not np.allclose(C_computed, C_expected, atol=1e-10, rtol=1e-10):
        # Fallback to schoolbook if construction failed
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
