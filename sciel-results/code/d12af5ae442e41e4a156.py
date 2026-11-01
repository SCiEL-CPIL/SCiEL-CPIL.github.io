"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import time

def _build_strassen():
    """
    Return the verified rank-7 Strassen decomposition.
    U shape: (4, 7) -> rows i*m+k (i=0..1, k=0..1)
    V shape: (4, 7) -> rows k*p+j (k=0..1, j=0..1)
    W shape: (4, 7) -> rows j*n+i (j=0..1, i=0..1)
    
    Index mapping for U (A indices i, k):
    0: A00 (i=0, k=0)
    1: A01 (i=0, k=1)
    2: A10 (i=1, k=0)
    3: A11 (i=1, k=1)
    
    Index mapping for V (B indices k, j):
    0: B00 (k=0, j=0)
    1: B01 (k=0, j=1)
    2: B10 (k=1, j=0)
    3: B11 (k=1, j=1)
    
    Index mapping for W (C indices j, i) based on seed observation:
    0: C00 (j=0, i=0)
    1: C10 (j=0, i=1)
    2: C01 (j=1, i=0)
    3: C11 (j=1, i=1)
    """
    n, m, p = DIMENSIONS
    r = 7
    # Initialize with integers to satisfy "integer/half-integer entries" strictly
    U = [[0] * r for _ in range(4)]
    V = [[0] * r for _ in range(4)]
    W = [[0] * r for _ in range(4)]

    # M1 = (A00+A11)*(B00+B11)
    # U: A00 (0), A11 (3)
    # V: B00 (0), B11 (3)
    # W: C00 (0), C11 (3) -> M1 contributes to C00 and C11
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (A10+A11)*B00
    # U: A10 (2), A11 (3)
    # V: B00 (0)
    # W: C10 (1), C11 (3) -> M2 contributes to C10 and C11 (with -1 for C11 in standard formula)
    # Standard Strassen:
    # C00 = M1 + M4 - M5 - M7
    # C10 = M2 + M3
    # C01 = M1 + M4
    # C11 = M1 + M3 - M2 - M6
    
    # Let's align the signs with the standard formula.
    # M2 contributes +1 to C10, -1 to C11.
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    # M3 = A00*(B01-B11)
    # U: A00 (0)
    # V: B01 (1), B11 (3)
    # W: C10 (1), C11 (3) -> M3 contributes +1 to C10, +1 to C11?
    # Standard: C10 = M2 + M3. C11 = ... + M3 ...
    # So M3 contributes +1 to C10 and +1 to C11.
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4 = A11*(B10-B00)
    # U: A11 (3)
    # V: B10 (2), B00 (0)
    # W: C00 (0), C01 (2) -> M4 contributes +1 to C00, +1 to C01.
    # Standard: C00 = ... + M4 ... C01 = M1 + M4.
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5 = (A00+A01)*B11
    # U: A00 (0), A01 (1)
    # V: B11 (3)
    # W: C00 (0), C01 (2) -> M5 contributes -1 to C00, +1 to C01?
    # Standard: C00 = ... - M5 ... C01 does NOT contain M5 in standard Strassen?
    # Let's re-verify standard Strassen formulas.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C10 = M2 + M3
    # C01 = M1 + M4
    # C11 = M1 + M3 - M2 - M6
    
    # Check M5:
    # C00 has -M5.
    # C01 does NOT have M5.
    # So W for M5 should only affect C00?
    # Wait, M5 = (A00+A01)B11.
    # In the expansion of C = AB:
    # C00 = A00B00 + A01B10
    # C10 = A10B00 + A11B10
    # C01 = A00B01 + A01B11
    # C11 = A10B01 + A11B11
    
    # Let's check the decomposition identity for M5.
    # M5 contributes to C00 with -1.
    # Does it contribute to others?
    # The tensor decomposition is unique up to transformation.
    # Let's check the seed code's M5:
    # U[0][4]=1, U[1][4]=1 (A00, A01)
    # V[3][4]=1 (B11)
    # W[0][4]=-1, W[2][4]=1 (C00, C01)
    # If W[2][4]=1, then M5 contributes to C01.
    # Standard formula says C01 = M1 + M4.
    # Let's expand M1 + M4:
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # M4 = A11B10 - A11B00
    # M1 + M4 = A00B00 + A00B11 + A11B10 + A11B11
    # This is NOT C01 (A00B01 + A01B11).
    # There is a mismatch in my memory of Strassen formulas or the seed code is using a different variant.
    # Let's trust the seed code's verification which passed.
    # The seed code had:
    # M5: W[0]=-1, W[2]=1.
    # M7: W[0]=1.
    # M6: W[3]=1.
    
    # Let's just copy the exact coefficients from the seed code which is known to work.
    
    # Reset U, V, W to be safe
    U = [[0] * r for _ in range(4)]
    V = [[0] * r for _ in range(4)]
    W = [[0] * r for _ in range(4)]

    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return {"U": U, "V": V, "W": W}

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin
    
    # Rank 7 is optimal. Return the hardcoded Strassen decomposition.
    res7 = _build_strassen()
    return res7

# EVOLVE-BLOCK-END
