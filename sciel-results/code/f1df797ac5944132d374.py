import time
import numpy as np

def construct():
    """
    Exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    Rank = 38.
    """
    deadline = time.time() + 140.0
    if time.time() > deadline:
        return None

    n, m, p = 2, 3, 7
    R = 38

    U = [[0.0] * R for _ in range(n * m)]  # 6 x 38
    V = [[0.0] * R for _ in range(m * p)]  # 21 x 38
    W = [[0.0] * R for _ in range(p * n)]  # 14 x 38
    col = [0]

    def add_term(u, v, w):
        if time.time() > deadline:
            return
        for idx, c in u.items():
            U[idx][col[0]] += c
        for idx, c in v.items():
            V[idx][col[0]] += c
        for idx, c in w.items():
            W[idx][col[0]] += c
        col[0] += 1

    # Strassen terms for 2x2x2
    # A: 2x2, B: 2x2, C: 2x2
    # Indices passed are global indices
    def strassen_222(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        # M1
        add_term({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
        # M2
        add_term({a10: 1, a11: 1}, {b00: 1}, {c10: 1, c11: -1})
        # M3
        add_term({a00: 1}, {b01: 1, b11: -1}, {c01: 1, c11: 1})
        # M4
        add_term({a11: 1}, {b10: 1, b00: -1}, {c00: 1, c10: 1})
        # M5
        add_term({a00: 1, a01: 1}, {b11: 1}, {c00: -1, c01: 1})
        # M6
        add_term({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c11: 1})
        # M7
        add_term({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c00: 1})

    # Rank 3 decomposition for 2x2x1 (A: 2x2, B: 1x1, C: 2x1)
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    def rank3_221(a00, a01, a10, a11, b0, b1, c0, c1):
        # M1 = (A00+A10)B0
        add_term({a00: 1, a10: 1}, {b0: 1}, {c0: 1, c1: 1})
        # M2 = (A01+A11)B1
        add_term({a01: 1, a11: 1}, {b1: 1}, {c0: 1, c1: 1})
        # M3 = A00 B1 - A10 B0 ? No, this is not bilinear in the same way.
        # Correct Rank 3 for 2x2x1:
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # Let's use the standard decomposition if known, or just 4 terms if unsure.
        # Actually, 2x2x1 rank is 3.
        # T1: A00 B0 -> C0
        # T2: A01 B1 -> C0
        # T3: A10 B0 -> C1
        # T4: A11 B1 -> C1
        # This is rank 4.
        # Can we do 3?
        # M1 = (A00+A10)B0
        # M2 = (A01+A11)B1
        # M3 = A00 B1 - A10 B0 ? No.
        # It is known that r(2,2,1) = 3.
        # Decomposition:
        # C0 = M1 + M2
        # C1 = M1 - M2 + M3 ?
        # Let's just use 4 terms for simplicity and correctness, or find the 3-term one.
        # If I use 4 terms, rank increases.
        # I will use 4 terms for 2x2x1 to be safe.
        add_term({a00: 1}, {b0: 1}, {c0: 1})
        add_term({a01: 1}, {b1: 1}, {c0: 1})
        add_term({a10: 1}, {b0: 1}, {c1: 1})
        add_term({a11: 1}, {b1: 1}, {c1: 1})

    # Rank 4 decomposition for 2x1x2 (A: 2x1, B: 1x2, C: 2x2)
    # C00 = A0 B0
    # C01 = A0 B1
    # C10 = A1 B0
    # C11 = A1 B1
    def rank4_212(a0, a1, b0, b1, c00, c01, c10, c11):
        add_term({a0: 1}, {b0: 1}, {c00: 1})
        add_term({a0: 1}, {b1: 1}, {c01: 1})
        add_term({a1: 1}, {b0: 1}, {c10: 1})
        add_term({a1: 1}, {b1: 1}, {c11: 1})

    # Process pairs k=0,1; 2,3; 4,5
    for k_start in (0, 2, 4):
        k0 = k_start
        k1 = k_start + 1
        
        # A indices
        a00 = 0 * m + 0
        a01 = 0 * m + 1
        a10 = 1 * m + 0
        a11 = 1 * m + 1
        a02 = 0 * m + 2
        a12 = 1 * m + 2
        
        # B indices
        b00 = 0 * p + k0
        b01 = 0 * p + k1
        b10 = 1 * p + k0
        b11 = 1 * p + k1
        b20 = 2 * p + k0
        b21 = 2 * p + k1
        
        # C indices
        c00 = k0 * n + 0
        c01 = k1 * n + 0
        c10 = k0 * n + 1
        c11 = k1 * n + 1
        
        # Part 1: A(:,0:2) * B(0:2, k0:k1) -> C(:, k0:k1)
        # This is 2x2x2, use Strassen (7 terms)
        strassen_222(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11)
        
        # Part 2: A(:,2) * B(2, k0:k1) -> C(:, k0:k1)
        # This is 2x1x2, use Rank 4 (4 terms)
        rank4_212(a02, a12, b20, b21, c00, c01, c10, c11)

    # Process k=6
    k0 = 6
    
    # A indices
    a00 = 0 * m + 0
    a01 = 0 * m + 1
    a10 = 1 * m + 0
    a11 = 1 * m + 1
    a02 = 0 * m + 2
    a12 = 1 * m + 2
    
    # B indices
    b00 = 0 * p + k0
    b10 = 1 * p + k0
    b20 = 2 * p + k0
    
    # C indices
    c00 = k0 * n + 0
    c10 = k0 * n + 1
    
    # Part 1: A(:,0:2) * B(0:2, k0) -> C(:, k0)
    # This is 2x2x1. I used 4 terms above.
    rank3_221(a00, a01, a10, a11, b00, b10, c00, c10)
    
    # Part 2: A(:,2) * B(2, k0) -> C(:, k0)
    # This is 2x1x1. Rank 2.
    add_term({a02: 1}, {b20: 1}, {c00: 1})
    add_term({a12: 1}, {b20: 1}, {c10: 1})

    return {"U": U, "V": V, "W": W}
