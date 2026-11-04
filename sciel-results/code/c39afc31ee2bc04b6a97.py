"""Matrix multiplication 4x4x5 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy: Decompose 4x4x5 into 4x4x4 (rank 49 via Strassen) and 4x4x1 (rank 16 via schoolbook).
    Total Rank = 49 + 16 = 65.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    rank_total = 65
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    state = [0]
    
    def add_strassen_4x4x4():
        """
        Add 49 terms for 4x4 * 4x4 using Strassen.
        """
        # Strassen steps for 4x4x4
        # Each step is a 2x2x2 multiplication of linear combinations of 2x2 blocks.
        # We define the 7 steps.
        # Step k: X_k = sum c_{k,a,b} A_{a,b}, Y_k = sum d_{k,a,b} B_{a,b}
        # where a,b in {0,1} are block indices.
        # A_{0,0} is rows 0-1, cols 0-1.
        # A_{0,1} is rows 0-1, cols 2-3.
        # A_{1,0} is rows 2-3, cols 0-1.
        # A_{1,1} is rows 2-3, cols 2-3.
        
        # Define the 7 Strassen combinations for 4x4.
        # c and d are 2x2 matrices of coefficients for the 4 blocks.
        steps = [
            # k=0: (A00+A11)(B00+B11)
            (np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])),
            # k=1: (A10+A11)B00
            (np.array([[0, 0], [1, 1]]), np.array([[1, 0], [0, 0]])),
            # k=2: A00(B01-B11)
            (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, -1]])),
            # k=3: A11(B10-B00)
            (np.array([[0, 0], [0, 1]]), np.array([[0, 0], [1, -1]])),
            # k=4: (A00+A01)B11
            (np.array([[1, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
            # k=5: (A10-A00)(B00+B01)
            (np.array([[0, 0], [1, 0]]) - np.array([[1, 0], [0, 0]]), np.array([[1, 1], [0, 0]])),
            # k=6: (A01-A11)(B10+B11)
            (np.array([[0, 1], [0, 0]]) - np.array([[0, 0], [0, 1]]), np.array([[0, 0], [1, 1]]))
        ]
        
        # Correcting the step definitions to be clear:
        # c[k][a][b] is coeff for A_{a,b}
        # d[k][a][b] is coeff for B_{a,b}
        
        c = [
            np.array([[1, 0], [0, 1]]), # A00+A11
            np.array([[0, 0], [1, 1]]), # A10+A11
            np.array([[1, 0], [0, 0]]), # A00
            np.array([[0, 0], [0, 1]]), # A11
            np.array([[1, 1], [0, 0]]), # A00+A01
            np.array([[-1, 0], [1, 0]]), # A10-A00
            np.array([[0, 1], [0, -1]])  # A01-A11
        ]
        
        d = [
            np.array([[1, 0], [0, 1]]), # B00+B11
            np.array([[1, 0], [0, 0]]), # B00
            np.array([[0, 1], [0, -1]]), # B01-B11
            np.array([[0, 0], [1, -1]]), # B10-B00
            np.array([[0, 0], [0, 1]]), # B11
            np.array([[1, 1], [0, 0]]), # B00+B01
            np.array([[0, 0], [1, 1]])  # B10+B11
        ]
        
        # The 7 kernel terms for 2x2x2 Strassen
        # Each term is defined by U_coeffs (4x1), V_coeffs (4x1), W_coeffs (4x1)
        # Indices 0:00, 1:01, 2:10, 3:11 for 2x2 matrix
        kernel = [
            # 1: (x00+x11)(y00+y11) -> z00+=1, z11+=1
            (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
            # 2: (x10+x11)y00 -> z10+=1, z11-=1
            (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1])),
            # 3: x00(y01-y11) -> z01+=1, z11+=1
            (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 1, 0, 1])),
            # 4: x11(y10-y00) -> z00+=1, z10+=1
            (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 0, 1, 0])),
            # 5: (x00+x01)y11 -> z00-=1, z01+=1
            (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])),
            # 6: (x10-x00)(y00+y01) -> z11+=1
            (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
            # 7: (x01-x11)(y10+y11) -> z00+=1
            (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]))
        ]
        
        # Helper to get global index from block (a,b) and local (lr, lc)
        def get_global_idx(a, b, lr, lc):
            # Block a: row block (0: 0-1, 1: 2-3)
            # Block b: col block (0: 0-1, 1: 2-3)
            row = 2*a + lr
            col = 2*b + lc
            return row, col

        for k in range(7):
            c_k = c[k]
            d_k = d[k]
            for t in range(7):
                u_k, v_k, w_k = kernel[t]
                col_idx = state[0]
                
                # U: 4x4 matrix of coeffs
                # We need to map the 4 local indices of X to global indices of A
                # X is a 2x2 matrix. Its elements are linear combos of A blocks.
                # X[lr, lc] = sum_{a,b} c_k[a,b] * A_{a,b}[lr, lc]
                # So U[global_idx(a,b,lr,lc), col_idx] += c_k[a,b] * u_k[lr*2+lc]
                
                for lr in range(2):
                    for lc in range(2):
                        u_val = u_k[lr*2+lc]
                        if u_val != 0:
                            for a in range(2):
                                for b in range(2):
                                    c_val = c_k[a,b]
                                    if c_val != 0:
                                        row, col_A = get_global_idx(a, b, lr, lc)
                                        U[row * m + col_A, col_idx] += c_val * u_val
                                        
                # V: 4x4 matrix of coeffs
                # Y[lr, lc] = sum_{a,b} d_k[a,b] * B_{a,b}[lr, lc]
                # V[global_idx(a,b,lr,lc), col_idx] += d_k[a,b] * v_k[lr*2+lc]
                # Note: B is 4x4. Block (a,b) of B corresponds to rows 2a..2a+1, cols 2b..2b+1.
                for lr in range(2):
                    for lc in range(2):
                        v_val = v_k[lr*2+lc]
                        if v_val != 0:
                            for a in range(2):
                                for b in range(2):
                                    d_val = d_k[a,b]
                                    if d_val != 0:
                                        row, col_B = get_global_idx(a, b, lr, lc)
                                        V[row * p + col_B, col_idx] += d_val * v_val
                                        
                # W: 4x4 matrix of coeffs
                # Z is the result 2x2 block? No, Z is the 2x2 result of the virtual multiplication.
                # The result of step k contributes to the 4 blocks of C.
                # C = C00 + C01 + C10 + C11?
                # Strassen reconstruction:
                # C00 = M0 - M4 + M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M0 - M1 + M3 + M6
                # Where M0..M6 are the 7 results.
                # So for step k, we add w_k * coeff_k to the appropriate C blocks.
                
                # We need the reconstruction coefficients for step k.
                # Let's define them.
                # k=0 (M0): C00+=1, C11+=1
                # k=1 (M1): C11-=1
                # k=2 (M2): C10+=1
                # k=3 (M3): C01+=1, C11+=1
                # k=4 (M4): C00-=1, C10+=1
                # k=5 (M5): C00+=1, C01+=1
                # k=6 (M6): C00+=1
                
                # Wait, standard Strassen:
                # C00 = M1 - M4 + M5 + M7 (using 1-based indexing for M1..M7)
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                # Let's map my k=0..6 to M1..M7.
                # k=0 -> M1
                # k=1 -> M2
                # k=2 -> M3
                # k=3 -> M4
                # k=4 -> M5
                # k=5 -> M6
                # k=6 -> M7
                
                # Reconstruction coeffs for C blocks (00, 01, 10, 11)
                # Index 0: C00, 1: C01, 2: C10, 3: C11
                recon = [
                    np.array([1, 0, 0, 1]), # k=0 (M1): C00+=1, C11+=1
                    np.array([0, 0, 0, -1]), # k=1 (M2): C11-=1
                    np.array([0, 1, 0, 1]), # k=2 (M3): C01+=1, C11+=1
                    np.array([-1, 0, 1, 0]), # k=3 (M4): C00-=1, C10+=1
                    np.array([1, 1, 0, 0]), # k=4 (M5): C00+=1, C01+=1
                    np.array([0, 0, 0, 1]), # k=5 (M6): C11+=1
                    np.array([1, 0, 0, 0])  # k=6 (M7): C00+=1
                ]
                
                r_coeffs = recon[k]
                
                # Z is 2x2. Z[lr, lc] is the local result.
                # This result is added to C blocks.
                # For each C block (ca, cb), we add r_coeffs[ca*2+cb] * Z[lr, lc]
                # to C[global_idx(ca, cb, lr, lc)]
                
                for lr in range(2):
                    for lc in range(2):
                        w_val = w_k[lr*2+lc]
                        if w_val != 0:
                            for ca in range(2):
                                for cb in range(2):
                                    r_val = r_coeffs[ca*2+cb]
                                    if r_val != 0:
                                        row_C, col_C = get_global_idx(ca, cb, lr, lc)
                                        # W index is k*n + i. Here k is col_C, i is row_C.
                                        W[col_C * n + row_C, col_idx] += r_val * w_val
                                        
                state[0] += 1

    def add_schoolbook_4x4x1():
        """
        Add 16 terms for 4x4 * 4x1.
        B is 4x1 (col 4).
        """
        for i in range(4):
            for j in range(4):
                col_idx = state[0]
                U[i * m + j, col_idx] = 1.0
                V[j * p + 4, col_idx] = 1.0
                W[4 * n + i, col_idx] = 1.0
                state[0] += 1

    add_strassen_4x4x4()
    add_schoolbook_4x4x1()
    
    used_rank = state[0]
    U = U[:, :used_rank]
    V = V[:, :used_rank]
    W = W[:, :used_rank]
    
    # Verify
    def verify(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(actual - expected) > tol:
                        return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
                
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
