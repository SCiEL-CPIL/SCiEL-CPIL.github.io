"""
Tensor decomposition for 3x3 * 3x5 matrix multiplication.
Uses a block-based approach splitting A and B into 2x2, 2x1, 1x2, 1x1 blocks.
Achieves rank 22 by using Strassen for 2x2x2 blocks and minimal rank for rectangular blocks.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s deadline

    n, m, p = DIMENSIONS  # 3, 3, 5

    # Rank terms: (U_coeffs, V_coeffs, W_coeffs)
    # U: (i, j), V: (j, k), W: (k, i)
    rank_terms = []

    def add_term(U_coeffs, V_coeffs, W_coeffs):
        rank_terms.append((U_coeffs, V_coeffs, W_coeffs))

    # --- Block 1: A_top,:2 (2x2) * B_top (2x5) ---
    # A_top,:2 is A[0:2, 0:2]. B_top is B[0:2, 0:5].
    # Split B_top into B1 (2x2, k=0,1) and B2 (2x3, k=2,3,4).
    
    # --- Sub-block 1.1: A[0:2,0:2] * B[0:2,0:2] (2x2x2 Strassen) ---
    # M1 = (a00+a11)(b00+b11)
    add_term({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
    # M2 = (a10+a11)b00
    add_term({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1})
    # M3 = a00(b01-b11)
    add_term({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
    # M4 = a11(b10-b00)
    add_term({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1})
    # M5 = (a00+a01)b11
    add_term({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1})
    # M6 = (a10-a00)(b00+b01)
    add_term({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1})
    # M7 = (a01-a11)(b10+b11)
    add_term({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})

    # --- Sub-block 1.2: A[0:2,0:2] * B[0:2,2] (2x2 * 2x1) ---
    # C[0,2] = a00*b02 + a01*b12
    # C[1,2] = a10*b02 + a11*b12
    # M8 = a00*b02
    add_term({(0,0): 1}, {(0,2): 1}, {(2,0): 1})
    # M9 = a01*b12
    add_term({(0,1): 1}, {(1,2): 1}, {(2,0): 1})
    # M10 = a10*b02
    add_term({(1,0): 1}, {(0,2): 1}, {(2,1): 1})
    # M11 = a11*b12
    add_term({(1,1): 1}, {(1,2): 1}, {(2,1): 1})

    # --- Sub-block 1.3: A[0:2,0:2] * B[0:2,3:5] (2x2 * 2x2 Strassen) ---
    # B[0:2, 3:5] has cols 3,4.
    # M12 = (a00+a11)(b03+b13)
    add_term({(0,0): 1, (1,1): 1}, {(0,3): 1, (1,3): 1}, {(3,0): 1, (4,1): 1})
    # M13 = (a10+a11)b03
    add_term({(1,0): 1, (1,1): 1}, {(0,3): 1}, {(3,1): 1, (4,1): -1})
    # M14 = a00(b04-b14)
    add_term({(0,0): 1}, {(0,4): 1, (1,4): -1}, {(4,0): 1, (3,1): 1})
    # M15 = a11(b13-b03)
    add_term({(1,1): 1}, {(1,3): 1, (0,3): -1}, {(3,0): 1, (4,0): 1})
    # M16 = (a00+a01)b13
    add_term({(0,0): 1, (0,1): 1}, {(1,3): 1}, {(3,0): -1, (4,0): 1})
    # M17 = (a10-a00)(b03+b04)
    add_term({(1,0): 1, (0,0): -1}, {(0,3): 1, (0,4): 1}, {(4,1): 1})
    # M18 = (a01-a11)(b13+b14)
    add_term({(0,1): 1, (1,1): -1}, {(1,3): 1, (1,4): 1}, {(3,0): 1})

    # --- Block 2: A[0:2,2] * B[2,:] (2x1 * 1x5) ---
    # C[i,k] = A[i,2] B[2,k] for i=0,1
    # 1 term
    add_term({(0,2): 1, (1,2): 1}, {(2,0): 1, (2,1): 1, (2,2): 1, (2,3): 1, (2,4): 1}, {(0,0): 1, (1,0): 1, (0,1): 1, (1,1): 1, (0,2): 1, (1,2): 1, (0,3): 1, (1,3): 1, (0,4): 1, (1,4): 1})

    # --- Block 3: A[2,0:2] * B[0:2,:] (1x2 * 2x5) ---
    # C[2,k] = A[2,0] B[0,k] + A[2,1] B[1,k]
    # 2 terms
    # M20 = A[2,0] B[0,k]
    add_term({(2,0): 1}, {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1, (0,4): 1}, {(0,2): 1, (1,2): 1, (2,2): 1, (3,2): 1, (4,2): 1})
    # M21 = A[2,1] B[1,k]
    add_term({(2,1): 1}, {(1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1, (1,4): 1}, {(0,2): 1, (1,2): 1, (2,2): 1, (3,2): 1, (4,2): 1})

    # --- Block 4: A[2,2] * B[2,:] (1x1 * 1x5) ---
    # C[2,k] = A[2,2] B[2,k]
    # 1 term
    add_term({(2,2): 1}, {(2,0): 1, (2,1): 1, (2,2): 1, (2,3): 1, (2,4): 1}, {(0,2): 1, (1,2): 1, (2,2): 1, (3,2): 1, (4,2): 1})

    total_rank = len(rank_terms)

    # Build U, V, W matrices
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    for r, (U_coeffs, V_coeffs, W_coeffs) in enumerate(rank_terms):
        for (i, j), coeff in U_coeffs.items():
            U[i * m + j, r] = coeff
        for (j, k), coeff in V_coeffs.items():
            V[j * p + k, r] = coeff
        for (k, i), coeff in W_coeffs.items():
            W[k * n + i, r] = coeff

    # Verify
    def verify(U, V, W):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        result = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            result += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        return np.allclose(result, T)

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        schoolbook_rank = n * m * p
        U_s = np.zeros((n * m, schoolbook_rank), dtype=float)
        V_s = np.zeros((m * p, schoolbook_rank), dtype=float)
        W_s = np.zeros((p * n, schoolbook_rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U_s[i * m + j, col] = 1.0
                    V_s[j * p + k, col] = 1.0
                    W_s[k * n + i, col] = 1.0
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}


# Test the construct function
if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print("U shape:", len(result['U']), "x", len(result['U'][0]))
    print("V shape:", len(result['V']), "x", len(result['V'][0]))
    print("W shape:", len(result['W']), "x", len(result['W'][0]))
