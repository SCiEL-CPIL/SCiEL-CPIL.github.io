import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 4, 6
    t0 = time.time()
    deadline = t0 + 140.0
    
    # Dimensions
    dim_u = n * m  # 8
    dim_v = m * p  # 24
    dim_w = p * n  # 12
    
    # Target rank to search for. 
    # Schoolbook is 48. 
    # The current program claims 26 (which is likely invalid or I am misinterpreting the tensor structure in the prompt vs standard matmul).
    # Wait, the prompt says "score is the rank". Lower is better.
    # The current program returns rank 26 or 48.
    # Let's look at the "Knowledge" section: "Using Strassen's algorithm on 2x2x2 sub-blocks reduces the rank from the schoolbook 48 to 42".
    # So 42 is a known valid rank.
    # Can we go lower?
    # For 2x4 * 4x6, the tensor rank is at least max(nm, mp, np) = max(8, 24, 12) = 24? No, that's for standard matrix mult rank lower bound which is weak.
    # Actually, for $A \in F^{n \times m}$ and $B \in F^{m \times p}$, the tensor rank is the rank of the matmul tensor.
    # Known bounds: $R(M_{n,m,p}) \le n m p$.
    # Strassen style block decomposition:
    # 2x4 * 4x6 can be seen as two 2x2 * 2x6 blocks? Or 2x4 * 2x2 blocks?
    # If we split M=4 into two M=2 blocks:
    # X = [X1 X2] (2x2, 2x2)
    # Y = [Y1; Y2] (2x6, 2x6)
    # Z = X1 Y1 + X2 Y2.
    # Each 2x2 * 2x6 matmul has rank?
    # 2x2 * 2x2 is rank 6 (Strassen).
    # 2x2 * 2x6 is equivalent to 3 independent 2x2*2x2? No, it's one 2x2*2x6.
    # Rank of 2x2 * 2x6 tensor:
    # It can be decomposed using Strassen for the 2x2 part?
    # Actually, $M_{2,2,6}$ rank is known to be 12? (2 * Rank(2x2*2x2) = 2*6=12).
    # So 2 blocks of 2x2*2x6 would be rank 24?
    # But the current program claims rank 26.
    # Let's try to find a rank 42 decomposition first as it is a safe bet mentioned in knowledge, then try to optimize down.
    # Actually, the prompt says "current program (score 1.0000)". The score is seed/value.
    # If seed is 48 (schoolbook), and value is rank, then score = 48/rank.
    # If rank is 42, score = 48/42 = 1.14.
    # If rank is 26, score = 48/26 = 1.84.
    # The current program returns rank 26 (if valid) or 48 (fallback).
    # The knowledge says "Measured: 1 -> 1.143". This implies the island best is currently 1.143 (rank 42).
    # The "current program" in the prompt has a comment "Rank-30 exact tensor decomposition" but the code builds rank 26.
    # Wait, if the current program in the prompt is the *seed* (score 1.0), it might be the schoolbook one?
    # No, the prompt says "CURRENT PROGRAM (score 1.0000)".
    # If the current program is the one shown, and it returns rank 26, why is its score 1.0?
    # Maybe the seed value is 26? Or maybe the current program shown is NOT the one achieving 1.0, but the one I am evolving FROM?
    # "INSPIRATIONS ... (score 1.0000): Initial program..."
    # It is possible the "Current Program" provided in the prompt is actually a *failed* attempt or a specific baseline that scores 1.0 relative to a seed of 26?
    # Or maybe the seed is 48, and the current program returns 48 (fallback) because the rank 26 construction is invalid?
    # Let's check the validity of the rank 26 construction in the current program.
    # It uses 2x2 Strassen (rank 6) for blocks.
    # It constructs 2 blocks of "2x4 * 4x6"? No.
    # It splits X (2x4) into left (2x2) and right (2x2).
    # It splits Y (4x6) into top (2x6) and bottom (2x6).
    # Z = X_left Y_top + X_right Y_bottom? No, Z = X_left Y_top + X_left Y_bottom + X_right Y_top + X_right Y_bottom?
    # The code does:
    # Columns 0..11: Top 4x6 part?
    # U24 is 2x4. V26 is 4x6.
    # It seems to compute X * Y by splitting Y into Y_top (2x6) and Y_bot (2x6).
    # Z = X Y_top + X Y_bot.
    # X is 2x4. Y_top is 2x6. This is 2x4 * 2x6 matmul.
    # Rank of 2x4 * 2x6?
    # Split X into X1 (2x2) and X2 (2x2).
    # Z = X1 Y1 + X2 Y2 where Y1, Y2 are 2x6.
    # Rank of 2x2 * 2x6 is 12 (2 * Strassen 2x2*2x2).
    # So two such terms would be rank 24?
    # The code uses rank 6 for the 2x2*2x2 core, but applies it to 2x2*2x6?
    # In `_matmul22_6`, it returns U,V,W for 2x2*2x2? No, shapes are 4x6, 4x6, 4x6.
    # 2x2*2x2 has size 4x4x4. Rank 6.
    # The code uses this for 2x2 * 2x2 blocks.
    # For the 2x4 * 4x6 case:
    # It seems to treat the problem as two 2x2 * 2x6 multiplications?
    # If rank(2x2*2x6) = 12, then 2 * 12 = 24.
    # The code builds rank 26.
    # If the code is valid, rank is 26. Score 48/26 = 1.84.
    # Why does it say score 1.0000?
    # Maybe the "seed" is 26?
    # Or maybe the code is INVALID and falls back to 48?
    # If it falls back to 48, and seed is 48, score is 1.0.
    # This is highly likely. The manual construction of rank 26 is likely flawed or the verification `np.allclose` fails, triggering the fallback to 48.
    
    # Strategy: Use numerical optimization to find a valid low-rank decomposition.
    # We will search for rank R.
    # We start with R=42 (known valid via Strassen blocks) and try to optimize to R=40, 38, etc.
    # But optimization is slow.
    # Better: Construct a known valid rank 42 decomposition explicitly, verify it, and return it.
    # Then, if time permits, try to optimize it down to 40 or 38.
    
    def construct_strassen_42():
        """
        Construct a rank-42 decomposition using Strassen on 2x2 blocks.
        2x4 * 4x6.
        Split X (2x4) into X1 (2x2), X2 (2x2).
        Split Y (4x6) into Y1 (2x6), Y2 (2x6).
        Z = X1 Y1 + X2 Y2.
        
        Wait, Y is 4x6. Y1 is rows 0-1, Y2 is rows 2-3.
        X is 2x4. X1 is cols 0-1, X2 is cols 2-3.
        
        We need rank of 2x2 * 2x6.
        2x2 * 2x6 can be viewed as 3 independent 2x2 * 2x2 multiplications?
        No, the output is 2x6.
        Let Y1 = [Y1a Y1b Y1c] where Y1a,b,c are 2x2.
        Then X1 Y1 = X1 Y1a + X1 Y1b + X1 Y1c.
        Each X1 Y1k is 2x2 * 2x2, rank 6.
        Total rank for X1 Y1 is 3 * 6 = 18?
        This would give total rank 18 + 18 = 36?
        Is rank(2x2 * 2x6) = 18?
        Lower bound: max(4, 12, 12) = 12.
        Upper bound: 12 (schoolbook for 2x2*2x6 is 2*2*6=24).
        Strassen reduces 2x2*2x2 to 6.
        So 2x2*2x6 = 3 * (2x2*2x2) = 3 * 6 = 18.
        So Z = X1 Y1 + X2 Y2 would be rank 18 + 18 = 36?
        
        Let's check if we can share computations.
        X1 Y1 = X1 [Y1a Y1b Y1c].
        The 3 Strassen decompositions are independent because the outputs are disjoint (columns 0-1, 2-3, 4-5).
        So yes, rank 18 for 2x2*2x6.
        Total rank 36.
        
        Can we do better?
        Maybe 2x4 * 4x6 can be done with rank < 36?
        
        Let's implement the rank 36 construction.
        If valid, score = 48/36 = 1.33.
        This is better than 42 (1.14).
        """
        n, m, p = 2, 4, 6
        U = np.zeros((n*m, 36))
        V = np.zeros((m*p, 36))
        W = np.zeros((p*n, 36))
        
        # Strassen 2x2 * 2x2 (Rank 6)
        # Inputs: X (2x2), Y (2x2). Output Z (2x2).
        # U_s: 4x6, V_s: 4x6, W_s: 4x6
        U_s = np.array([
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 0]
        ])
        V_s = np.array([
            [1, 1, -1, 0, 0, 1],
            [1, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0],
            [1, 0, 0, -1, 1, 1]
        ])
        W_s = np.array([
            [1, 0, -1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 0]
        ])
        
        # We need to decompose 2x4 * 4x6.
        # X = [X1 X2] (2x2, 2x2)
        # Y = [Y1; Y2] (2x6, 2x6)
        # Z = X1 Y1 + X2 Y2
        
        # Decompose X1 Y1 (2x2 * 2x6) into 3 Strassen blocks (2x2 * 2x2).
        # Y1 = [Y1_0 Y1_1 Y1_2] (2x2, 2x2, 2x2)
        # X1 Y1 = X1 Y1_0 + X1 Y1_1 + X1 Y1_2
        
        # Block 1: X1 * Y1_0 (cols 0-1 of Y1, which are cols 0-1 of Y)
        # X1 is X[:, 0:2]. Y1_0 is Y[0:2, 0:2].
        # Output Z[:, 0:2].
        
        # Block 2: X1 * Y1_1 (cols 2-3 of Y1, which are cols 2-3 of Y)
        # X1 is X[:, 0:2]. Y1_1 is Y[0:2, 2:4].
        # Output Z[:, 2:4].
        
        # Block 3: X1 * Y1_2 (cols 4-5 of Y1, which are cols 4-5 of Y)
        # X1 is X[:, 0:2]. Y1_2 is Y[0:2, 4:6].
        # Output Z[:, 4:6].
        
        # Similarly for X2 * Y2.
        # Y2 = [Y2_0 Y2_1 Y2_2]
        # Y2_0 is Y[2:4, 0:2]. Output Z[:, 0:2].
        # Y2_1 is Y[2:4, 2:4]. Output Z[:, 2:4].
        # Y2_2 is Y[2:4, 4:6]. Output Z[:, 4:6].
        
        # Total 6 Strassen blocks. Rank 6 * 6 = 36.
        
        # Indices:
        # X: 0-1 (row 0), 2-3 (row 1) ... wait.
        # X is 2x4. Flattened: x00, x01, x02, x03, x10, x11, x12, x13.
        # X1 (top-left 2x2): x00, x01, x10, x11. Indices: 0,1,4,5.
        # X2 (top-right 2x2): x02, x03, x12, x13. Indices: 2,3,6,7.
        
        # Y is 4x6. Flattened: y00..y05, y10..y15, y20..y25, y30..y35.
        # Y1 (top 2x6): y00..y05, y10..y15. Indices: 0-11.
        # Y2 (bottom 2x6): y20..y25, y30..y35. Indices: 12-23.
        
        # Y1_0 (cols 0-1 of Y1): y00,y01,y10,y11. Indices: 0,1,6,7.
        # Y1_1 (cols 2-3 of Y1): y02,y03,y12,y13. Indices: 2,3,8,9.
        # Y1_2 (cols 4-5 of Y1): y04,y05,y14,y15. Indices: 4,5,10,11.
        
        # Y2_0 (cols 0-1 of Y2): y20,y21,y30,y31. Indices: 12,13,18,19.
        # Y2_1 (cols 2-3 of Y2): y22,y23,y32,y33. Indices: 14,15,20,21.
        # Y2_2 (cols 4-5 of Y2): y24,y25,y34,y35. Indices: 16,17,22,23.
        
        # Output Z is 2x6. Flattened: z00..z05, z10..z15.
        # W[k*n+i] convention: W[k*2+i].
        # z00 is index 0. z01 is index 1. ... z05 is index 5.
        # z10 is index 6. z11 is index 7. ... z15 is index 11.
        
        # Strassen Block 1: X1 * Y1_0 -> Z[:, 0:2] (z00, z01, z10, z11)
        # Strassen Block 2: X1 * Y1_1 -> Z[:, 2:4] (z02, z03, z12, z13)
        # Strassen Block 3: X1 * Y1_2 -> Z[:, 4:6] (z04, z05, z14, z15)
        # Strassen Block 4: X2 * Y2_0 -> Z[:, 0:2] (z00, z01, z10, z11)
        # Strassen Block 5: X2 * Y2_1 -> Z[:, 2:4] (z02, z03, z12, z13)
        # Strassen Block 6: X2 * Y2_2 -> Z[:, 4:6] (z04, z05, z14, z15)
        
        def add_strassen_block(x_indices, y_indices, z_indices, start_col):
            for c in range(6):
                # U
                U[x_indices[0], start_col+c] = U_s[0, c]
                U[x_indices[1], start_col+c] = U_s[1, c]
                U[x_indices[2], start_col+c] = U_s[2, c]
                U[x_indices[3], start_col+c] = U_s[3, c]
                
                # V
                V[y_indices[0], start_col+c] = V_s[0, c]
                V[y_indices[1], start_col+c] = V_s[1, c]
                V[y_indices[2], start_col+c] = V_s[2, c]
                V[y_indices[3], start_col+c] = V_s[3, c]
                
                # W
                W[z_indices[0], start_col+c] = W_s[0, c]
                W[z_indices[1], start_col+c] = W_s[1, c]
                W[z_indices[2], start_col+c] = W_s[2, c]
                W[z_indices[3], start_col+c] = W_s[3, c]
        
        # X1 indices: 0,1,4,5
        # X2 indices: 2,3,6,7
        
        # Block 1: X1, Y1_0 (0,1,6,7), Z0 (0,1,6,7)
        add_strassen_block([0,1,4,5], [0,1,6,7], [0,1,6,7], 0)
        
        # Block 2: X1, Y1_1 (2,3,8,9), Z1 (2,3,8,9)
        add_strassen_block([0,1,4,5], [2,3,8,9], [2,3,8,9], 6)
        
        # Block 3: X1, Y1_2 (4,5,10,11), Z2 (4,5,10,11)
        add_strassen_block([0,1,4,5], [4,5,10,11], [4,5,10,11], 12)
        
        # Block 4: X2, Y2_0 (12,13,18,19), Z0 (0,1,6,7)
        add_strassen_block([2,3,6,7], [12,13,18,19], [0,1,6,7], 18)
        
        # Block 5: X2, Y2_1 (14,15,20,21), Z1 (2,3,8,9)
        add_strassen_block([2,3,6,7], [14,15,20,21], [2,3,8,9], 24)
        
        # Block 6: X2, Y2_2 (16,17,22,23), Z2 (4,5,10,11)
        add_strassen_block([2,3,6,7], [16,17,22,23], [4,5,10,11], 30)
        
        return U, V, W

    # Generate candidate
    U, V, W = construct_strassen_42()
    
    # Verify
    rng = np.random.RandomState(42)
    X = rng.randn(2, 4)
    Y = rng.randn(4, 6)
    Z_true = X @ Y
    
    x = X.reshape(8)
    y = Y.reshape(24)
    
    Z_decomp = np.zeros(12)
    for c in range(36):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        a = np.einsum('r,r->', uc, x)
        b = np.einsum('r,r->', vc, y)
        Z_decomp += wc * (a * b)
    
    # W[k*n+i] -> Z[i,k]
    Z_decomp = Z_decomp.reshape(6, 2).T
    
    if not np.allclose(Z_decomp, Z_true, atol=1e-9, rtol=1e-9):
        # Fallback to schoolbook 48
        U48 = np.zeros((8, 48))
        V48 = np.zeros((24, 48))
        W48 = np.zeros((12, 48))
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(6):
                    U48[i*4+j, col] = 1.0
                    V48[j*6+k, col] = 1.0
                    W48[k*2+i, col] = 1.0
                    col += 1
        return {"U": U48.tolist(), "V": V48.tolist(), "W": W48.tolist()}
    
    # If valid, try to optimize to lower rank?
    # Given time constraints and the reliability of the explicit construction, we return the rank 36 solution.
    # Score 48/36 = 1.333.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
