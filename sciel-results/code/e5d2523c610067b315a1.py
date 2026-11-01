"""Matrix multiplication 2x4 x 4x4 tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    def build_strassen_28():
        """Build rank-28 decomposition using Strassen on 2x2 blocks."""
        rank = 28
        U = [[0] * rank for _ in range(n * m)]  # 8 x 28
        V = [[0] * rank for _ in range(m * p)]  # 16 x 28
        W = [[0] * rank for _ in range(p * n)]  # 8 x 28
        
        # Partition: A = [A_left A_right] each 2x2
        # B = [[B_tl B_tr], [B_bl B_br]] each 2x2
        # C = [C_left C_right] each 2x2
        #
        # C_left = A_left * B_tl + A_right * B_bl
        # C_right = A_left * B_tr + A_right * B_br
        #
        # Apply Strassen to each 2x2 = 2x2 * 2x2 product.
        #
        # For product X(2x2) * Y(2x2) = Z(2x2):
        # M1 = (X11+X22)(Y11+Y22)
        # M2 = (X21+X22)Y11
        # M3 = X11(Y21-Y22)
        # M4 = X22(Y12-Y11)
        # M5 = (X11+X12)Y22
        # M6 = (X21-X11)(Y11+Y12)
        # M7 = (X12-X22)(Y21+Y22)
        # Z11 = M1+M4-M5+M7
        # Z12 = M3+M5
        # Z21 = M2+M4
        # Z22 = M1-M2+M3+M6
        
        # Strassen coefficients for U (X side), V (Y side), W (Z side)
        # Each entry is a dict mapping (row, col) -> coeff
        strassen_U = {
            0: {(0,0):1, (1,1):1},
            1: {(1,0):1, (1,1):1},
            2: {(0,0):1},
            3: {(1,1):1},
            4: {(0,0):1, (0,1):1},
            5: {(1,0):1, (0,0):-1},
            6: {(0,1):1, (1,1):-1},
        }
        strassen_V = {
            0: {(0,0):1, (1,1):1},
            1: {(0,0):1},
            2: {(1,0):1, (1,1):-1},
            3: {(0,1):1, (0,0):-1},
            4: {(1,1):1},
            5: {(0,0):1, (0,1):1},
            6: {(1,0):1, (1,1):1},
        }
        strassen_W = {
            0: {(0,0):1, (1,1):1},
            1: {(1,0):1, (1,1):-1},
            2: {(0,1):1, (1,1):1},
            3: {(0,0):1, (1,0):1},
            4: {(0,0):-1, (0,1):1},
            5: {(1,1):1},
            6: {(0,0):1},
        }
        
        # We need 4 Strassen decompositions:
        # 1. A_left * B_tl -> contributes to C_left
        # 2. A_right * B_bl -> contributes to C_left
        # 3. A_left * B_tr -> contributes to C_right
        # 4. A_right * B_br -> contributes to C_right
        
        # A_left: rows 0,1 of A, cols 0,1 -> U indices: i*4+j for i in {0,1}, j in {0,1}
        # A_right: rows 0,1 of A, cols 2,3 -> U indices: i*4+j for i in {0,1}, j in {2,3}
        # B_tl: rows 0,1 of B, cols 0,1 -> V indices: j*4+k for j in {0,1}, k in {0,1}
        # B_tr: rows 0,1 of B, cols 2,3 -> V indices: j*4+k for j in {0,1}, k in {2,3}
        # B_bl: rows 2,3 of B, cols 0,1 -> V indices: j*4+k for j in {2,3}, k in {0,1}
        # B_br: rows 2,3 of B, cols 2,3 -> V indices: j*4+k for j in {2,3}, k in {2,3}
        # C_left: rows 0,1 of C, cols 0,1 -> W indices: k*2+i for k in {0,1}, i in {0,1}
        # C_right: rows 0,1 of C, cols 2,3 -> W indices: k*2+i for k in {2,3}, i in {0,1}
        
        col = 0
        # Strassen decomposition 1: A_left * B_tl -> C_left
        for s in range(7):
            c = col + s
            for (r, ccol), coeff in strassen_U[s].items():
                U[r * 4 + ccol][c] += coeff  # A_left: row r, col ccol (0,1)
            for (r, ccol), coeff in strassen_V[s].items():
                V[r * 4 + ccol][c] += coeff  # B_tl: row r, col ccol (0,1)
            for (r, ccol), coeff in strassen_W[s].items():
                W[ccol * 2 + r][c] += coeff  # C_left: col ccol, row r
        col += 7
        
        # Strassen decomposition 2: A_right * B_bl -> C_left
        for s in range(7):
            c = col + s
            for (r, ccol), coeff in strassen_U[s].items():
                U[r * 4 + (ccol + 2)][c] += coeff  # A_right: row r, col ccol+2 (2,3)
            for (r, ccol), coeff in strassen_V[s].items():
                V[(r + 2) * 4 + ccol][c] += coeff  # B_bl: row r+2, col ccol (2,3) x (0,1)
            for (r, ccol), coeff in strassen_W[s].items():
                W[ccol * 2 + r][c] += coeff  # C_left: col ccol, row r
        col += 7
        
        # Strassen decomposition 3: A_left * B_tr -> C_right
        for s in range(7):
            c = col + s
            for (r, ccol), coeff in strassen_U[s].items():
                U[r * 4 + ccol][c] += coeff  # A_left: row r, col ccol (0,1)
            for (r, ccol), coeff in strassen_V[s].items():
                V[r * 4 + (ccol + 2)][c] += coeff  # B_tr: row r, col ccol+2 (0,1) x (2,3)
            for (r, ccol), coeff in strassen_W[s].items():
                W[(ccol + 2) * 2 + r][c] += coeff  # C_right: col ccol+2, row r
        col += 7
        
        # Strassen decomposition 4: A_right * B_br -> C_right
        for s in range(7):
            c = col + s
            for (r, ccol), coeff in strassen_U[s].items():
                U[r * 4 + (ccol + 2)][c] += coeff  # A_right: row r, col ccol+2 (2,3)
            for (r, ccol), coeff in strassen_V[s].items():
                V[(r + 2) * 4 + (ccol + 2)][c] += coeff  # B_br: row r+2, col ccol+2
            for (r, ccol), coeff in strassen_W[s].items():
                W[(ccol + 2) * 2 + r][c] += coeff  # C_right: col ccol+2, row r
        col += 7
        
        return {"U": U, "V": V, "W": W}
    
    def validate(U, V, W, rank):
        """Check if U, V, W form a valid decomposition."""
        # Check diagonal (should be 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        # Check off-diagonal (should be 0)
        for a in range(n*m):
            for b_idx in range(m*p):
                for c_idx in range(p*n):
                    i1 = a // m
                    j1 = a % m
                    j2 = b_idx // p
                    k1 = b_idx % p
                    i2 = c_idx // n
                    k2 = c_idx % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    s = 0.0
                    for c in range(rank):
                        s += U[a][c] * V[b_idx][c] * W[c_idx][c]
                    if abs(s) > 1e-9:
                        return False
        return True
    
    # Build and validate the rank-28 Strassen decomposition
    result_28 = build_strassen_28()
    valid_28 = validate(result_28["U"], result_28["V"], result_28["W"], 28)
    
    if valid_28:
        best_result = result_28
        best_rank = 28
    else:
        # Fallback to schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    # Try to find rank-27 decomposition
    if time.time() < deadline - 15:
        # Strategy: Start from rank-28 and try to merge two columns or find
        # a linear dependency that allows reducing rank
        
        # Approach 1: Try removing one column and adjusting others
        # The rank-28 decomposition has 28 columns. If any column can be
        # expressed as a linear combination of others in a way that preserves
        # the tensor structure, we can reduce to 27.
        
        # Approach 2: Direct random search for rank-27
        # The tensor has 8*16*8 = 1024 entries, but only 32 are non-zero.
        # For rank r, we need to satisfy:
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 if i,j,k "match" else 0
        
        # Let's try a structured random search
        found_27 = False
        max_attempts = 5000
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            random.seed(attempt * 7919 + 27 * 1000003)
            
            # Generate random half-integer matrices with sparsity bias
            # Start from Strassen-28 structure and perturb
            
            U28 = np.array(best_result["U"], dtype=float)
            V28 = np.array(best_result["V"], dtype=float)
            W28 = np.array(best_result["W"], dtype=float)
            
            # Try: take first 27 columns of Strassen and see if valid
            # (unlikely, but cheap to check)
            if attempt < 28:
                U27 = U28[:, :27]
                V27 = V28[:, :27]
                W27 = W28[:, :27]
                if validate(U27.tolist(), V27.tolist(), W27.tolist(), 27):
                    best_result = {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
                    best_rank = 27
                    found_27 = True
                    break
            
            # Try: random rank-27 with half-integer entries
            # Use a smarter initialization: random ±1, ±0.5, 0 with some structure
            def rand_half_int():
                vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
                return random.choice(vals)
            
            U = [[rand_half_int() for _ in range(27)] for _ in range(n*m)]
            V = [[rand_half_int() for _ in range(27)] for _ in range(m*p)]
            W = [[rand_half_int() for _ in range(27)] for _ in range(p*n)]
            
            if validate(U, V, W, 27):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = 27
                found_27 = True
                break
        
        if found_27:
            return best_result
    
    # Try rank 26 if we have time
    if time.time() < deadline - 10 and best_rank > 26:
        found_26 = False
        max_attempts = 2000
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            random.seed(99999 + attempt * 31)
            
            def rand_half_int():
                vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
                return random.choice(vals)
            
            U = [[rand_half_int() for _ in range(26)] for _ in range(n*m)]
            V = [[rand_half_int() for _ in range(26)] for _ in range(m*p)]
            W = [[rand_half_int() for _ in range(26)] for _ in range(p*n)]
            
            if validate(U, V, W, 26):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = 26
                found_26 = True
                break
    
    return best_result

# EVOLVE-BLOCK-END
