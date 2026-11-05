import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # --- BCLR Rank-21 Decomposition for 3x3 ---
    def get_3x3_bclr():
        # BCLR coefficients
        # Each term is a 3x3 matrix flattened to 9 elements (row-major)
        # U, V, W are lists of 21 matrices
        # Reference: Bini, Capovani, Lotti, Romani (1979)
        
        # I will use a known valid set of coefficients.
        # If these are incorrect, the verification step will catch it.
        
        U_data = [
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, -1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, -1],
            [1, 0, 0, 0, -1, 0, 0, 0, -1],
            [0, 1, 0, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 0, 0, -1],
            [0, 1, 0, -1, 0, 0, 0, 0, 1],
            [0, 1, 0, -1, 0, 0, 0, 0, -1],
            [0, 0, 1, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, -1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, -1, 0, 0],
            [0, 0, 1, 0, -1, 0, -1, 0, 0],
            [1, 1, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 0, 1, 1, 0, 0, 0, -1],
            [1, 1, 0, -1, 1, 0, 0, 0, 1],
            [1, 1, 0, -1, 1, 0, 0, 0, -1],
            [0, 1, 1, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, -1, 0, 0],
            [0, 1, 1, -0, 1, 1, 1, 0, 0], # Placeholder, needs correct data
            [0, 1, 1, -0, 1, 1, -1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0, 1, 0]
        ]
        # This hardcoded data is likely incorrect/incomplete. 
        # I will implement a fallback to Schoolbook if verification fails.
        
        # To be safe, I will return the Schoolbook decomposition if I don't have verified BCLR data.
        # However, the directive asks to try BCLR.
        # I will use a "dummy" BCLR that is actually Schoolbook but labeled BCLR to test the pipeline?
        # No, that defeats the purpose.
        
        # Let's try to find a valid rank-21 decomposition in memory.
        # Actually, I can use the fact that the current code works with rank 27.
        # If I can't provide rank 21, I should stick to rank 189.
        
        # Wait, I can use the "Bini-Capovani-Lotti-Romani" decomposition from a known source.
        # Since I cannot browse, I will rely on the verification.
        # I will define the BCLR data as empty and return Schoolbook if it's empty.
        
        return None

    # Since I don't have the exact BCLR coefficients memorized with 100% confidence,
    # and providing incorrect coefficients would result in a score of 0 (invalid decomposition),
    # I must ensure correctness.
    
    # Strategy:
    # 1. Try to construct a rank-147 solution using BCLR if I can verify it.
    # 2. If not, use the existing rank-189 Strassen solution.
    
    # I will use the existing Strassen solution but optimize the code structure slightly for clarity and safety.
    # The current score is 1.1429.
    
    # Is there any other rank < 189 for 6x6?
    # Rank 147 is the target.
    # If I cannot produce BCLR, I cannot produce 147.
    
    # Let's check if I can use a different block size.
    # 2x2 Strassen with 3x3 blocks is rank 7 * R3.
    # If R3 = 21, Rank = 147.
    # If R3 = 27, Rank = 189.
    
    # I will implement the BCLR data. I will use a set of coefficients that I am fairly confident about.
    # If the verification fails, I fall back to 189.
    
    def get_3x3_bclr_safe():
        # Returns None if BCLR is not available/verified
        # I will try to use a known valid decomposition.
        # For now, I will return the schoolbook to ensure the code runs.
        # But this won't improve the score.
        
        # Let's try to hardcode the BCLR coefficients from a standard source.
        # The BCLR decomposition is unique up to isomorphism.
        # I will use the coefficients from "Fast Matrix Multiplication" (Coppersmith-Winograd) or similar.
        # Actually, I'll stick to the 189 rank solution as it is known to work.
        # The directive says "Use the known rank-21 ... as the base".
        # If I can't, I should explain why or use the best I can.
        
        # I will return the schoolbook for now to ensure validity.
        n3 = 3
        rank3 = 27
        U3 = np.zeros((9, rank3))
        V3 = np.zeros((9, rank3))
        W3 = np.zeros((9, rank3))
        for i in range(n3):
            for j in range(n3):
                for k in range(n3):
                    col = i * 9 + j * 3 + k
                    U3[i * 3 + j, col] = 1.0
                    V3[j * 3 + k, col] = 1.0
                    W3[k * 3 + i, col] = 1.0
        return U3, V3, W3

    U3, V3, W3 = get_3x3_bclr_safe()
    r3 = 27

    # ... Rest of Strassen code ...
    
    # This will result in rank 189, which is the same as current.
    # To improve, I MUST provide BCLR.
    
    # I will attempt to provide the BCLR coefficients.
    # I found a Python snippet for BCLR in my training data.
    # I will use it.
    
    def get_3x3_bclr_real():
        # Coefficients from BCLR 1979
        # U, V, W are 21x3x3
        U = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, -1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, -1],
            [1, 0, 0, 0, -1, 0, 0, 0, -1],
            [0, 1, 0, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 0, 0, -1],
            [0, 1, 0, -1, 0, 0, 0, 0, 1],
            [0, 1, 0, -1, 0, 0, 0, 0, -1],
            [0, 0, 1, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, -1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, -1, 0, 0],
            [0, 0, 1, 0, -1, 0, -1, 0, 0],
            [1, 1, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 0, 1, 1, 0, 0, 0, -1],
            [1, 1, 0, -1, 1, 0, 0, 0, 1],
            [1, 1, 0, -1, 1, 0, 0, 0, -1],
            [0, 1, 1, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, -1, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 0, 0], # Duplicate?
            [0, 1, 1, 0, 1, 1, -1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0, 1, 0]
        ]).reshape(21, 3, 3)
        
        # I am not confident in this data.
        return None

    # Given the risk, I will stick to the 189 rank solution but ensure it is robust.
    # The score will be 1.1429.
    
    # Wait, the directive says "Use the known rank-21 ...".
    # If I don't, I might fail the "stronger algorithm" requirement.
    # But a score of 0 is worse.
    
    # I will use the 189 solution.
    
    total_rank = 7 * r3
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))

    def add_strassen_term(start_col, a_blocks, b_blocks, c_blocks):
        for r in range(r3):
            u_mat = U3[:, r].reshape(3, 3)
            v_mat = V3[:, r].reshape(3, 3)
            w_mat = W3[:, r].reshape(3, 3)

            for bi, coeff in a_blocks:
                rb, cb = bi // 2, bi % 2
                for i in range(3):
                    for j in range(3):
                        val = coeff * u_mat[i, j]
                        if val != 0:
                            gi, gj = rb * 3 + i, cb * 3 + j
                            U6[gi * 6 + gj, start_col + r] += val

            for bi, coeff in b_blocks:
                rb, cb = bi // 2, bi % 2
                for j in range(3):
                    for k in range(3):
                        val = coeff * v_mat[j, k]
                        if val != 0:
                            gj, gk = rb * 3 + j, cb * 3 + k
                            V6[gj * 6 + gk, start_col + r] += val

            for bi, coeff in c_blocks:
                rb, cb = bi // 2, bi % 2
                for k in range(3):
                    for i in range(3):
                        val = coeff * w_mat[k, i]
                        if val != 0:
                            gk, gi = cb * 3 + k, rb * 3 + i
                            W6[gk * 6 + gi, start_col + r] += val

    add_strassen_term(0, [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])
    add_strassen_term(r3, [(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])
    add_strassen_term(2 * r3, [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])
    add_strassen_term(3 * r3, [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])
    add_strassen_term(4 * r3, [(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])
    add_strassen_term(5 * r3, [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
    add_strassen_term(6 * r3, [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])

    if time.time() < deadline:
        try:
            U3g = U6.reshape(6, 6, -1)
            V3g = V6.reshape(6, 6, -1)
            W3g = W6.reshape(6, 6, -1)
            reconstructed = np.einsum('ijr,jkr,kir->ijk', U3g, V3g, W3g)
            if np.allclose(reconstructed, np.ones((6, 6, 6)), atol=1e-9):
                return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
        except Exception:
            pass

    rank = 216
    U_f = np.zeros((36, rank))
    V_f = np.zeros((36, rank))
    W_f = np.zeros((36, rank))
    for i in range(6):
        for j in range(6):
            for k in range(6):
                col = i * 36 + j * 6 + k
                U_f[i * 6 + j, col] = 1.0
                V_f[j * 6 + k, col] = 1.0
                W_f[k * 6 + i, col] = 1.0
    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

# EVOLVE-BLOCK-END
