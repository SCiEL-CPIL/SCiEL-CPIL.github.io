"""Improved program: hierarchical tensor decomposition for 2x5x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the 7 rank-1 terms for 2x2x2 Strassen decomposition.
    Each term is (u_dict, v_dict, w_dict) where keys are (i,j), (j,k), (k,i)
    and values are integer coefficients.
    
    Standard Strassen (0-based):
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    terms = []
    # M1
    terms.append({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    # M2
    terms.append({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1})
    # M3
    terms.append({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1})
    # M4
    terms.append({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1})
    # M5
    terms.append({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1})
    # M6
    terms.append({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})
    # M7
    terms.append({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    return terms

def _strassen_2x3():
    """
    Return rank-1 terms for 2x3x2 multiplication using Strassen-like decomposition.
    A is 2x3, B is 3x2, C is 2x2.
    Known rank: 11.
    
    We split the 3 columns of A into (1, 2) and use Strassen on 2x2x2 plus extra terms.
    Actually, let's use a known decomposition.
    
    A = [[a00, a01, a02], [a10, a11, a12]]
    B = [[b00, b01], [b10, b11], [b20, b21]]
    C = [[a00*b00 + a01*b10 + a02*b20, a00*b01 + a01*b11 + a02*b21],
         [a10*b00 + a11*b10 + a12*b20, a10*b01 + a11*b11 + a12*b21]]
    
    One known rank-11 decomposition:
    Let's define 11 multiplications.
    """
    # I'll construct this carefully.
    # Split A into A_left (2x1) and A_right (2x2).
    # A_left = [[a00],[a10]], A_right = [[a01,a02],[a11,a12]]
    # B_left = [[b00],[b10],[b20]] (3x1), B_right = [[b01],[b11],[b21]] (3x1)
    # This doesn't directly work with Strassen.
    
    # Alternative: use the fact that R(2,3,2) = 11.
    # Let me just hardcode a valid set of 11 terms.
    # Reference: Bini, Capovani, Lotti, Romani (1978) or similar.
    
    # For simplicity and correctness, I'll use schoolbook for 2x3x2 which is rank 12,
    # but the known minimum is 11. Let me try to construct rank 11.
    
    # Actually, let me use a different approach: decompose 2x5x6 directly using
    # Kronecker product structure.
    
    # 2x5x6 can be seen as:
    # n=2, m=5, p=6
    # Write m=5 as 2+3, p=6 as 2*3 or 3*2.
    
    # Let's try: split m into m1=2, m2=3.
    # For each pair (i in {0,1}, k in {0..5}), the sum over j is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{4} A[i,j] * B[j,k]
    
    # First part: 2x2x6 (A is 2x2, B is 2x6, C is 2x6)
    # Second part: 2x3x6 (A is 2x3, B is 3x6, C is 2x6)
    
    # For 2x2x6: split p=6 into three 2s.
    # Each 2x2x2 block has rank 7 (Strassen).
    # 3 blocks * 7 = 21.
    
    # For 2x3x6: split p=6 into three 2s.
    # Each 2x3x2 block has rank 11.
    # 3 blocks * 11 = 33.
    
    # Total: 21 + 33 = 54.
    
    # Let me implement this carefully.
    pass

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # Strategy: split m=5 into m1=2, m2=3.
    # Split p=6 into 3 blocks of size 2.
    
    # For block (m1=2, p_block=2): 2x2x2, rank 7 (Strassen)
    # For block (m2=3, p_block=2): 2x3x2, rank 11
    
    # Total rank = 3*7 + 3*11 = 21 + 33 = 54
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]  # 10 x 54
    V = [[0] * rank for _ in range(m * p)]  # 30 x 54
    W = [[0] * rank for _ in range(p * n)]  # 12 x 54
    
    col = 0
    
    def add_term(u_map, v_map, w_map):
        nonlocal col
        for (i, j), coeff in u_map.items():
            U[i * m + j][col] += coeff
        for (j, k), coeff in v_map.items():
            V[j * p + k][col] += coeff
        for (k, i), coeff in w_map.items():
            W[k * n + i][col] += coeff
        col += 1
    
    # Get Strassen terms for 2x2x2
    strassen_terms = _strassen_2x2()
    
    # 2x3x2 terms - I need to define these
    # Let me construct a valid rank-11 decomposition for 2x3x2
    
    # A is 2x3: [[a00, a01, a02], [a10, a11, a12]]
    # B is 3x2: [[b00, b01], [b10, b11], [b20, b21]]
    # C is 2x2: [[c00, c01], [c10, c11]]
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    
    # Known rank-11 decomposition:
    # M1 = (a00+a12)(b00+b21)
    # M2 = (a02+a10)(b20+b01)
    # M3 = (a00+a10)(b00+b01)
    # M4 = (a01+a11)(b10+b11)
    # M5 = a00*b00
    # M6 = a01*b10
    # M7 = a02*b20
    # M8 = a10*b01
    # M9 = a11*b11
    # M10 = a12*b21
    # M11 = (a00-a10)(b01-b00)
    
    # Let me verify this... Actually, I'm not confident about the exact coefficients.
    # Let me use a safer approach: just use schoolbook for 2x3x2 (rank 12) and 
    # Strassen for 2x2x2 (rank 7).
    # Total: 3*7 + 3*12 = 21 + 36 = 57.
    
    # Or better: let me try to find a valid 2x3x2 rank-11 decomposition.
    
    # From literature, one valid rank-11 decomposition for 2x3x2:
    # Let me just use schoolbook for safety: rank 12.
    
    def schoolbook_2x3x2(i_off, j_off, k_off):
        """Add schoolbook terms for 2x3x2 block."""
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    # Apply offsets
                    u = {((i + i_off) % 2, (j + j_off) % 3): 1}
                    v = {((j + j_off) % 3, (k + k_off) % 2): 1}
                    w = {((k + k_off) % 2, (i + i_off) % 2): 1}
                    add_term(u, v, w)
    
    # Actually, let me just do the full schoolbook for 2x5x6 to be safe.
    # Rank = 60.
    
    # Wait, the current program already does rank 60. I need to improve it.
    # Let me implement the 54-rank decomposition with Strassen for 2x2x2 and 
    # schoolbook for 2x3x2, giving 3*7 + 3*12 = 57.
    
    # That's still an improvement over 60.
    
    # Let me redo:
    rank = 3 * 7 + 3 * 12  # 21 + 36 = 57
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # For each of the 3 p-blocks (size 2 each):
    for p_block in range(3):
        k_off = p_block * 2
        
        # Part 1: m1=2 (j=0,1), 2x2x2 Strassen
        for (u_map, v_map, w_map) in strassen_terms:
            # Apply offsets: j_off=0, k_off=k_off, i_off=0
            u_new = {(i, j): c for (i, j), c in u_map.items()}
            v_new = {(j, k + k_off): c for (j, k), c in v_map.items()}
            w_new = {(k + k_off, i): c for (k, i), c in w_map.items()}
            add_term(u_new, v_new, w_new)
        
        # Part 2: m2=3 (j=2,3,4), 2x3x2 schoolbook (rank 12)
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    u = {(i, j + 2): 1}
                    v = {(j + 2, k + k_off): 1}
                    w = {(k + k_off, i): 1}
                    add_term(u, v, w)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
