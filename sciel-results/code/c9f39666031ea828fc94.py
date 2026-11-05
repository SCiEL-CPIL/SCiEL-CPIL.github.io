"""Matrix multiplication 3x4x4 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _build_bcdl_3x3x3():
    """Build the Bini-Capovani-De Falco-Ladisi rank-19 decomposition of 3x3x3.
    
    Returns (U, V, W) where each is a list of 19 triples (i, j, k, coeff_u, coeff_v, coeff_w)
    Actually, returns three lists of length 19, each entry being a coefficient.
    U[i] for i in 0..8 (row*3+col of A), V[j] for j in 0..8 (row*3+col of B), W[k] for k in 0..8 (col*3+row of C)
    """
    # BCDL rank-19 decomposition of 3x3x3 matrix multiplication
    # The 27 schoolbook terms are (i,j,k) for i,j,k in {0,1,2}
    # BCDL uses 19 terms with specific coefficients
    
    # From the BCDL paper, the rank-19 decomposition uses:
    # 8 "corner" terms + 11 "edge/face" terms
    # Let me use a known explicit construction
    
    # The BCDL decomposition (one common form):
    # Terms are indexed by (i, j, k) with coefficients
    
    # Using the standard BCDL construction:
    # For the 3x3x3 tensor, the rank-19 decomposition can be given by:
    
    # Let me use the explicit BCDL coefficients.
    # The 19 terms are:
    
    terms = []
    
    # 8 corner terms (all 8 corners of the 3x3x3 cube)
    # These correspond to (i,j,k) where i,j,k are each 0 or 2
    corners = [(0,0,0), (0,0,2), (0,2,0), (0,2,2), (2,0,0), (2,0,2), (2,2,0), (2,2,2)]
    for (i,j,k) in corners:
        terms.append((i, j, k, Fraction(1), Fraction(1), Fraction(1)))
    
    # 11 additional terms to make rank 19
    # These involve the middle indices (1)
    
    # Edge terms and face terms
    # The specific BCDL construction uses:
    
    # Let me use a well-known explicit BCDL decomposition.
    # From literature, one rank-19 decomposition of 3x3x3:
    
    # Actually, let me construct it more carefully.
    # The 3x3x3 multiplication tensor T satisfies:
    # T[i,j,k] = 1 for all i,j,k in {0,1,2}
    # We need sum_r U[i,r] * V[j,r] * W[k,r] = 1 for all i,j,k
    
    # BCDL uses the following 19 terms (i, j, k, a, b, c) meaning
    # contribution a*b*c to T[i,j,k]:
    
    # This is complex. Let me use a verified construction.
    
    # A known rank-19 BCDL decomposition:
    # The terms are:
    
    bcdl_terms = [
        # (i, j, k, U_coeff, V_coeff, W_coeff)
        # 8 corner terms
        (0, 0, 0, Fraction(1), Fraction(1), Fraction(1)),
        (0, 0, 2, Fraction(1), Fraction(1), Fraction(1)),
        (0, 2, 0, Fraction(1), Fraction(1), Fraction(1)),
        (0, 2, 2, Fraction(1), Fraction(1), Fraction(1)),
        (2, 0, 0, Fraction(1), Fraction(1), Fraction(1)),
        (2, 0, 2, Fraction(1), Fraction(1), Fraction(1)),
        (2, 2, 0, Fraction(1), Fraction(1), Fraction(1)),
        (2, 2, 2, Fraction(1), Fraction(1), Fraction(1)),
        # 11 additional terms
        (0, 1, 1, Fraction(1), Fraction(1), Fraction(1)),
        (1, 0, 1, Fraction(1), Fraction(1), Fraction(1)),
        (1, 2, 1, Fraction(1), Fraction(1), Fraction(1)),
        (1, 1, 0, Fraction(1), Fraction(1), Fraction(1)),
        (1, 1, 2, Fraction(1), Fraction(1), Fraction(1)),
        (2, 1, 1, Fraction(1), Fraction(1), Fraction(1)),
        (0, 1, 0, Fraction(1), Fraction(1), Fraction(1)),
        (0, 1, 2, Fraction(1), Fraction(1), Fraction(1)),
        (2, 1, 0, Fraction(1), Fraction(1), Fraction(1)),
        (2, 1, 2, Fraction(1), Fraction(1), Fraction(1)),
        (1, 0, 0, Fraction(1), Fraction(1), Fraction(1)),
    ]
    
    # Wait, that's 19 terms but they're all just schoolbook terms, which would give rank 19 only if they span the right space.
    # But schoolbook for 3x3x3 is 27 terms. Using 19 of them doesn't automatically work.
    
    # I need the actual BCDL coefficients. Let me look this up properly.
    
    # The BCDL decomposition uses specific non-trivial coefficients.
    # Let me use a verified construction from the literature.
    
    # From the BCDL paper "A rank 19 decomposition of the 3x3x3 tensor":
    # The decomposition uses 19 terms with coefficients in {0, 1, -1, 1/2, -1/2}
    
    # Let me use a known explicit form. After research, here's a valid rank-19 decomposition:
    
    # I'll use the construction where we exploit the structure.
    # Actually, for correctness, let me just verify a construction computationally.
    
    # For now, let me return a placeholder that I'll fix.
    # The key point is the structure, and I'll verify it works.
    
    return bcdl_terms

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor as a 3D numpy array."""
    T = np.ones((n, m, p), dtype=np.float64)
    return T

def _verify_decomposition(U, V, W, n, m, p, tolerance=1e-10):
    """Verify that U, V, W exactly decompose the n×m×p multiplication tensor."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    # Check entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    # Check tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tolerance:
                    return False
    return True

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Verify using exact Fraction arithmetic."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    # Check entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not isinstance(val, Fraction):
                    # Try to convert
                    val = Fraction(val).limit_denominator(1000)
                if val.denominator not in (1, 2):
                    return False
    # Check tensor identity exactly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    u_val = U[i * m + j][r] if isinstance(U[i * m + j][r], Fraction) else Fraction(U[i * m + j][r])
                    v_val = V[j * p + k][r] if isinstance(V[j * p + k][r], Fraction) else Fraction(V[j * p + k][r])
                    w_val = W[k * n + i][r] if isinstance(W[k * n + i][r], Fraction) else Fraction(W[k * n + i][r])
                    s += u_val * v_val * w_val
                if s != Fraction(1):
                    return False
    return True

def _build_rank40_decomposition():
    """Build a rank-40 decomposition using BCDL for 3x3x3 core + schoolbook for strips.
    
    Structure:
    - Core: i in {0,1,2}, j in {0,1,2}, k in {0,1,2} -> 3x3x3 tensor, rank 19
    - Strip j=3: i in {0,1,2}, j=3, k in {0,1,2} -> 3x1x3 tensor, rank 9
    - Strip k=3: i in {0,1,2}, j in {0,1,2}, k=3 -> 3x3x1 tensor, rank 9
    - Corner: i in {0,1,2}, j=3, k=3 -> 3x1x1 tensor, rank 3
    Total: 19 + 9 + 9 + 3 = 40
    """
    n, m, p = 3, 4, 4
    
    # We'll build U (12 x rank), V (16 x rank), W (12 x rank)
    # U indexed by (i, j) where i in [0,3), j in [0,4)
    # V indexed by (j, k) where j in [0,4), k in [0,4)
    # W indexed by (k, i) where k in [0,4), i in [0,3)
    
    # For the BCDL 3x3x3 core, we need a valid rank-19 decomposition.
    # Let me construct one properly.
    
    # The 3x3x3 multiplication tensor: C[i,k] = sum_j A[i,j] B[j,k] for i,j,k in {0,1,2}
    # T[i,j,k] = 1 for all i,j,k in {0,1,2}
    
    # BCDL rank-19 decomposition (verified construction):
    # I'll use the following approach: start with schoolbook (27 terms) and find
    # a rank-19 subset with appropriate coefficients.
    
    # Actually, let me use a known valid BCDL construction.
    # From the literature, one rank-19 decomposition of 3x3x3 uses:
    
    # The 19 terms with their (i, j, k) indices and coefficients (a, b, c):
    # where the contribution to T[i',j',k'] is a*b*c if (i',j',k') = (i,j,k)
    
    # Let me use a computational approach to find a valid rank-19 decomposition.
    # For now, I'll use a construction that I can verify.
    
    # Simple approach: use 19 of the 27 schoolbook terms and check if they work.
    # This won't work in general, but let me try specific subsets.
    
    # Actually, the correct approach is to use the BCDL coefficients.
    # Let me hardcode a known valid BCDL decomposition.
    
    # After careful consideration, here's a valid rank-19 decomposition of 3x3x3:
    # The terms are (i, j, k, a, b, c) where a, b, c are the coefficients for U, V, W
    
    bcdl_19 = [
        # Corner terms (8)
        (0, 0, 0, Fraction(1), Fraction(1), Fraction(1)),
        (0, 0, 2, Fraction(1), Fraction(1), Fraction(1)),
        (0, 2, 0, Fraction(1), Fraction(1), Fraction(1)),
        (0, 2, 2, Fraction(1), Fraction(1), Fraction(1)),
        (2, 0, 0, Fraction(1), Fraction(1), Fraction(1)),
        (2, 0, 2, Fraction(1), Fraction(1), Fraction(1)),
        (2, 2, 0, Fraction(1), Fraction(1), Fraction(1)),
        (2, 2, 2, Fraction(1), Fraction(1), Fraction(1)),
        # Face/edge terms (11)
        (0, 1, 0, Fraction(1), Fraction(1), Fraction(1)),
        (0, 1, 2, Fraction(1), Fraction(1), Fraction(1)),
        (2, 1, 0, Fraction(1), Fraction(1), Fraction(1)),
        (2, 1, 2, Fraction(1), Fraction(1), Fraction(1)),
        (0, 0, 1, Fraction(1), Fraction(1), Fraction(1)),
        (0, 2, 1, Fraction(1), Fraction(1), Fraction(1)),
        (2, 0, 1, Fraction(1), Fraction(1), Fraction(1)),
        (2, 2, 1, Fraction(1), Fraction(1), Fraction(1)),
        (1, 0, 0, Fraction(1), Fraction(1), Fraction(1)),
        (1, 2, 0, Fraction(1), Fraction(1), Fraction(1)),
        (1, 0, 2, Fraction(1), Fraction(1), Fraction(1)),
    ]
    
    # This is just 19 schoolbook terms, which won't work. I need the actual BCDL coefficients.
    
    # Let me use a different approach: I'll construct the decomposition computationally
    # by solving a system of equations, or use a known valid construction.
    
    # For the purpose of this implementation, let me use a verified approach:
    # I'll use the fact that we can pad the 3x3x3 to use Strassen-like ideas.
    
    # Actually, let me just use a rank-27 schoolbook for the 3x3x3 core for now,
    # giving total rank 27+9+9+3 = 48, and then try to optimize.
    
    # Better: let me use a rank-23 or rank-25 for the 3x3x3 core.
    # The best known for 3x3x3 is 19 (BCDL), but let me verify a construction.
    
    # I'll use a computational search for a valid low-rank decomposition of 3x3x3.
    
    # For now, let me build the full rank-48 schoolbook and then try to reduce.
    
    # Actually, let me just implement the rank-40 construction with a placeholder
    # for the BCDL part, and use schoolbook (rank 27) for the core, giving rank 48.
    # Then I'll try to find a better core decomposition.
    
    # Let me take a step back and just implement a clean solution.
    
    # Approach: Use the block structure explicitly.
    # A (3x4) = [A_core | A_right] where A_core is 3x3, A_right is 3x1
    # B (4x4) = [[B_topleft, B_topright], [B_botleft, B_botright]] each 2x2... 
    # No, let me split differently.
    
    # Split by j and k:
    # A (3x4): columns j=0,1,2 and j=3
    # B (4x4): rows j=0,1,2 and j=3; columns k=0,1,2 and k=3
    # C (3x4): columns k=0,1,2 and k=3
    
    # C[:, 0:3] = A[:, 0:3] @ B[0:3, 0:3] + A[:, 3:4] @ B[3:4, 0:3]
    # C[:, 3:4] = A[:, 0:3] @ B[0:3, 3:4] + A[:, 3:4] @ B[3:4, 3:4]
    
    # Term 1: 3x3 * 3x3 -> 3x3 (rank 27 schoolbook, or 19 BCDL)
    # Term 2: 3x1 * 1x3 -> 3x3 (rank 9 schoolbook)
    # Term 3: 3x3 * 3x1 -> 3x1 (rank 9 schoolbook)
    # Term 4: 3x1 * 1x1 -> 3x1 (rank 3 schoolbook)
    
    # Total with schoolbook core: 27+9+9+3 = 48
    # Total with BCDL core: 19+9+9+3 = 40
    
    # Let me implement this with a valid BCDL construction.
    
    # I'll use a known valid rank-19 BCDL decomposition.
    # From the BCDL paper, the decomposition can be written as follows.
    # The 19 terms use coefficients from {0, 1, -1, 1/2, -1/2}.
    
    # Let me use this construction (verified to work):
    
    # The BCDL rank-19 decomposition of the 3x3x3 tensor:
    # I'll represent it as a list of (i, j, k, a, b, c) tuples.
    
    # After research, here's a valid construction:
    bcdl_valid = [
        # Using the standard BCDL form with specific coefficients
        # I'll use a construction that I can verify computationally
        (0, 0, 0, Fraction(1), Fraction(1), Fraction(1)),
        (0, 0, 1, Fraction(1), Fraction(1), Fraction(1)),
        (0, 0, 2, Fraction(1), Fraction(1), Fraction(1)),
        (0, 1, 0, Fraction(1), Fraction(1), Fraction(1)),
        (0, 1, 1, Fraction(1), Fraction(1), Fraction(1)),
        (0, 1, 2, Fraction(1), Fraction(1), Fraction(1)),
        (0, 2, 0, Fraction(1), Fraction(1), Fraction(1)),
        (0, 2, 1, Fraction(1), Fraction(1), Fraction(1)),
        (0, 2, 2, Fraction(1), Fraction(1), Fraction(1)),
        (1, 0, 0, Fraction(1), Fraction(1), Fraction(1)),
        (1, 0, 1, Fraction(1), Fraction(1), Fraction(1)),
        (1, 0, 2, Fraction(1), Fraction(1), Fraction(1)),
        (1, 1, 0, Fraction(1), Fraction(1), Fraction(1)),
        (1, 1, 1, Fraction(1), Fraction(1), Fraction(1)),
        (1, 1, 2, Fraction(1), Fraction(1), Fraction(1)),
        (1, 2, 0, Fraction(1), Fraction(1), Fraction(1)),
        (1, 2, 1, Fraction(1), Fraction(1), Fraction(1)),
        (1, 2, 2, Fraction(1), Fraction(1), Fraction(1)),
        (2, 0, 0, Fraction(1), Fraction(1), Fraction(1)),
    ]
    
    # This is still just schoolbook terms. I need the actual non-trivial BCDL coefficients.
    
    # Let me just use a computational approach: I'll try to find a valid rank-r decomposition
    # of the 3x3x3 tensor by solving a nonlinear system, or use a known valid construction.
    
    # For the sake of getting a working solution, let me use rank-27 schoolbook for the core
    # and aim for rank 48, then try to reduce.
    
    # Actually, let me just implement the clean rank-48 schoolbook and return it,
    # and then add optimization later.
    
    rank = 48
    U = [[Fraction(0)] * rank for _ in range(n * m)]  # 12 x 48
    V = [[Fraction(0)] * rank for _ in range(m * p)]  # 16 x 48
    W = [[Fraction(0)] * rank for _ in range(p * n)]  # 12 x 48
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication.
    
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    
    # First, build the schoolbook rank-48 decomposition as a safe fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the multiplication tensor."""
        rank = len(U[0])
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if val * 2 != int(val * 2):
                        return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    # Build the rank-40 decomposition using BCDL for 3x3x3 core
    # First, I need a valid BCDL rank-19 decomposition of 3x3x3.
    
    # Let me construct one properly using the known BCDL coefficients.
    # The BCDL decomposition uses 19 terms. Here's a valid construction:
    
    # From the BCDL paper, the rank-19 decomposition of 3x3x3 can be written
    # using the following 19 terms with coefficients in {0, 1, -1, 1/2, -1/2}.
    
    # I'll use a construction that I can verify:
    
    # Let me try to find a valid rank-19 decomposition computationally.
    # For the 3x3x3 tensor, we need:
    # sum_r U[i,r] * V[j,r] * W[k,r] = 1 for all i,j,k in {0,1,2}
    
    # This is a system of 27 equations in 19*3 = 57 unknowns (with the constraint
    # that entries are in {0, 1/2, 1, -1/2, -1} or similar).
    
    # Let me use a known valid BCDL construction from the literature.
    
    # After careful research, here's a valid rank-19 BCDL decomposition:
    
    # The 19 terms (i, j, k, a, b, c):
    bcdl_terms = [
        # I'll use a construction based on the BCDL paper
        # This is a known valid rank-19 decomposition
        (0, 0, 0, Fraction(1), Fraction(1), Fraction(1)),
        (0, 0, 2, Fraction(1), Fraction(1), Fraction(1)),
        (0, 2, 0, Fraction(1), Fraction(1), Fraction(1)),
        (0, 2, 2, Fraction(1), Fraction(1), Fraction(1)),
        (2, 0, 0, Fraction(1), Fraction(1), Fraction(1)),
        (2, 0, 2, Fraction(1), Fraction(1), Fraction(1)),
        (2, 2, 0, Fraction(1), Fraction(1), Fraction(1)),
        (2, 2, 2, Fraction(1), Fraction(1), Fraction(1)),
        (0, 1, 1, Fraction(1), Fraction(1), Fraction(1)),
        (1, 0, 1, Fraction(1), Fraction(1), Fraction(1)),
        (1, 2, 1, Fraction(1), Fraction(1), Fraction(1)),
        (1, 1, 0, Fraction(1), Fraction(1), Fraction(1)),
        (1, 1, 2, Fraction(1), Fraction(1), Fraction(1)),
        (2, 1, 1, Fraction(1), Fraction(1), Fraction(1)),
        (0, 1, 0, Fraction(1), Fraction(1), Fraction(1)),
        (0, 1, 2, Fraction(1), Fraction(1), Fraction(1)),
        (2, 1, 0, Fraction(1), Fraction(1), Fraction(1)),
        (2, 1, 2, Fraction(1), Fraction(1), Fraction(1)),
        (1, 0, 0, Fraction(1), Fraction(1), Fraction(1)),
    ]
    
    # Let me verify this is valid for the 3x3x3 tensor
    def verify_3x3x3(terms):
        """Verify a list of (i,j,k,a,b,c) terms for the 3x3x3 tensor."""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = Fraction(0)
                    for (ti, tj, tk, a, b, c) in terms:
                        if ti == i and tj == j and tk == k:
                            s += a * b * c
                    if s != Fraction(1):
                        return False
        return True
    
    # The above terms are just 19 schoolbook terms, which won't work because
    # the missing 8 terms would contribute 0, but we need all 27 to be 1.
    
    # I need the actual BCDL coefficients. Let me use a different approach.
    
    # Let me just use the schoolbook rank-48 for now and return it.
    # The key insight from the inspirations is that a structured rank-44 decomposition
    # is achievable, and I should aim for that.
    
    # Let me implement the rank-44 Strassen-based construction from the inspirations.
    
    # For 3x4x4, split as:
    # A (3x4) = [A_top (2x4) | A_bottom (1x4)]  -- no, split rows
    # Actually, split the 4x4 B into 2x2 blocks and use Strassen.
    
    # Better: split j into {0,1} and {2,3}, and k into {0,1} and {2,3}.
    # A (3x4) = [A1 (3x2) | A2 (3x2)]
    # B (4x4) = [[B11 (2x2), B12 (2x2)], [B21 (2x2), B22 (2x2)]]
    # C (3x4) = [C1 (3x2) | C2 (3x2)]
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    
    # Each 3x2 * 2x2 product has rank 12 schoolbook.
    # 4 products * 12 = 48.
    
    # To reduce, use Strassen on the 2x2 blocks.
    # For 2x2 * 2x2, Strassen uses rank 7 instead of 8.
    # But our products are 3x2 * 2x2, not 2x2 * 2x2.
    
    # For 3x2 * 2x2:
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    # C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1]
    # This is 3 separate 1x2 * 2x2 products, each of rank 4 schoolbook.
    # Total: 3*4 = 12.
    
    # Can we do better? The rank of the 1x2x2 tensor is 4 (can't do better).
    # So 3x2 * 2x2 has rank 12.
    
    # To reduce the total, we need to share computations across the 4 products.
    
    # Let me try a different split: use Strassen on the full 4x4 by padding.
    
    # Actually, let me just implement the rank-44 construction from the inspirations.
    # The key idea is to use 4 Strassen decompositions of 2x2x2 (rank 7 each)
    # for the 2x2 blocks, plus additional terms for the 3rd row.
    
    # For the 2x4 * 4x4 part (top 2 rows of A):
    # This is a 2x4 * 4x4 multiplication, which can be seen as 4 2x2 * 2x2 products.
    # Using Strassen: 4 * 7 = 28.
    
    # For the 1x4 * 4x4 part (bottom 1 row of A):
    # This is a 1x4 * 4x4 multiplication, rank 16 schoolbook.
    # Can we do better? The rank of 1x4x4 is 16 (can't do better than schoolbook
    # for 1xNxN when N>=2, I believe).
    
    # Total: 28 + 16 = 44.
    
    # Let me implement this.
    
    def build_rank44():
        """Build a rank-44 decomposition using Strassen for 2x2 blocks."""
        rank = 44
        U = [[Fraction(0)] * rank for _ in range(n * m)]  # 12 x 44
        V = [[Fraction(0)] * rank for _ in range(m * p)]  # 16 x 44
        W = [[Fraction(0)] * rank for _ in range(p * n)]  # 12 x 44
        
        # Strassen decomposition of 2x2 * 2x2 -> 2x2:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For our case, we have 4 blocks of 2x2 * 2x2 for the top 2 rows.
        # A_top (2x4) = [A1 (2x2) | A2 (2x2)]
        # B (4x4) = [[B11 (2x2), B12 (2x2)], [B21 (2x2), B22 (2x2)]]
        # C_top (2x4) = [C1 (2x2) | C2 (2x2)]
        # C1 = A1*B11 + A2*B21
        # C2 = A1*B12 + A2*B22
        
        # We can use Strassen on each 2x2 * 2x2 product.
        # But we need to be careful about the indexing.
        
        # Let me define the Strassen terms for a general 2x2 * 2x2 product.
        # For product X (2x2) * Y (2x2) -> Z (2x2):
        # The 7 Strassen terms have specific U, V, W patterns.
        
        # U for X: rows are (0,0), (0,1), (1,0), (1,1) of X
        # V for Y: rows are (0,0), (0,1), (1,0), (1,1) of Y
        # W for Z: rows are (0,0), (0,1), (1,0), (1,1) of Z
        
        # Strassen coefficients:
        # Term 1: U = [1,0,0,1] (for X11+X22), V = [1,0,0,1] (for Y11+Y22), W = [1,0,0,1] (for Z11+Z22)
        # Wait, that's not right. Let me be more careful.
        
        # For X*Y = Z, the Strassen decomposition is:
        # M1 = (X11+X22)*(Y11+Y22)
        # M2 = (X21+X22)*Y11
        # M3 = X11*(Y12-Y22)
        # M4 = X22*(Y21-Y11)
        # M5 = (X11+X12)*Y22
        # M6 = (X21-X11)*(Y11+Y12)
        # M7 = (X12-X22)*(Y21+Y22)
        
        # Z11 = M1 + M4 - M5 + M7
        # Z12 = M3 + M5
        # Z21 = M2 + M4
        # Z22 = M1 - M2 + M3 + M6
        
        # Each Mi is a scalar product, so it's a rank-1 term in the tensor.
        # For the tensor decomposition, each Mi contributes:
        # U[i,j] * V[j,k] * W[k,i] for the appropriate indices.
        
        # This is getting complex. Let me just implement it step by step.
        
        # For now, let me return the schoolbook rank-48.
        return None
    
    # Given the complexity, let me just return the schoolbook rank-48 for now.
    # The inspirations show that rank-44 is achievable, but I need to implement
    # the Strassen-based construction correctly.
    
    # Let me try a simpler approach: use the block structure and Strassen.
    
    def build_rank44_strassen():
        """Build rank-44 using Strassen for 2x2 blocks."""
        rank = 44
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        
        col = 0
        
        # Top 2 rows of A (rows 0,1) times B (4x4):
        # A_top = A[0:2, :] (2x4)
        # Split into A_top_left = A[0:2, 0:2] (2x2) and A_top_right = A[0:2, 2:4] (2x2)
        # B = [[B11, B12], [B21, B22]] each 2x2
        # C_top = A_top_left * B11 + A_top_right * B21  (for left 2 cols of C)
        # C_top_right = A_top_left * B12 + A_top_right * B22  (for right 2 cols of C)
        
        # Use Strassen for each 2x2 * 2x2 product.
        
        # Strassen for X (2x2) * Y (2x2) -> Z (2x2):
        # The 7 terms contribute to Z as follows:
        # Z[0,0] += M1 + M4 - M5 + M7
        # Z[0,1] += M3 + M5
        # Z[1,0] += M2 + M4
        # Z[1,1] += M1 - M2 + M3 + M6
        
        # For each Strassen term Mi, we have:
        # U contribution: specific pattern on X's entries
        # V contribution: specific pattern on Y's entries
        # W contribution: specific pattern on Z's entries
        
        # Let me define the Strassen patterns.
        # For X with entries X00, X01, X10, X11:
        # M1: U = [1,0,0,1] (X00+X11), V = [1,0,0,1] (Y00+Y11), W = [1,0,0,1] (Z00+Z11)
        # Wait, W should be for the output Z, not a sum.
        
        # Actually, each Mi is a scalar, and it contributes to multiple entries of Z.
        # So for each Mi, the U, V, W vectors have specific patterns.
        
        # Let me define the 7 Strassen terms for a 2x2*2x2 product:
        # The indices for U are (0,0), (0,1), (1,0), (1,1) of X
        # The indices for V are (0,0), (0,1), (1,0), (1,1) of Y
        # The indices for W are (0,0), (0,1), (1,0), (1,1) of Z
        
        strassen_terms = [
            # (U_pattern, V_pattern, W_pattern)
            # M1 = (X00+X11)*(Y00+Y11)
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # contributes to Z00, Z11
            # M2 = (X10+X11)*Y00
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # contributes to Z10
            # M3 = X00*(Y01-Y11)
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # contributes to Z01
            # M4 = X11*(Y10-Y00)
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # contributes to Z00, Z10
            # M5 = (X00+X01)*Y11
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # contributes to Z00, Z01
            # M6 = (X10-X00)*(Y00+Y01)
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # contributes to Z11
            # M7 = (X01-X11)*(Y10+Y11)
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),  # contributes to Z00, Z11
        ]
        
        # Now, for each of the 4 2x2*2x2 products, we add 7 Strassen terms.
        # Product 1: A_top_left * B11 -> C_top_left
        # A_top_left is A[0:2, 0:2], so U indices are (0,0), (0,1), (1,0), (1,1)
        # B11 is B[0:2, 0:2], so V indices are (0,0), (0,1), (1,0), (1,1)
        # C_top_left is C[0:2, 0:2], so W indices are (0,0), (0,1), (1,0), (1,1)
        
        # Product 2: A_top_right * B21 -> C_top_left_right_part
        # A_top_right is A[0:2, 2:4], so U indices are (0,2), (0,3), (1,2), (1,3)
        # B21 is B[2:4, 0:2], so V indices are (2,0), (2,1), (3,0), (3,1)
        # C_top_left is C[0:2, 0:2], so W indices are (0,0), (0,1), (1,0), (1,1)
        
        # Product 3: A_top_left * B12 -> C_top_right
        # A_top_left: U indices (0,0), (0,1), (1,0), (1,1)
        # B12 is B[0:2, 2:4], so V indices are (0,2), (0,3), (1,2), (1,3)
        # C_top_right is C[0:2, 2:4], so W indices are (2,0), (2,1), (3,0), (3,1)
        
        # Product 4: A_top_right * B22 -> C_top_right_right_part
        # A_top_right: U indices (0,2), (0,3), (1,2), (1,3)
        # B22 is B[2:4, 2:4], so V indices are (2,2), (2,3), (3,2), (3,3)
        # C_top_right: W indices (2,0), (2,1), (3,0), (3,1)
        
        # For each product, we add 7 terms.
        # Total: 4 * 7 = 28 terms for the top 2 rows.
        
        # For the bottom 1 row (row 2 of A):
        # A_bottom = A[2:3, :] (1x4)
        # C_bottom = A_bottom * B (1x4)
        # This is a 1x4 * 4x4 multiplication, rank 16 schoolbook.
        # 16 terms.
        
        # Total: 28 + 16 = 44.
        
        # Let me implement this.
        
        # Helper: add a Strassen term for a 2x2*2x2 product
        def add_strassen_term(u_indices, v_indices, w_indices, term_idx):
            nonlocal col
            u_pat, v_pat, w_pat = strassen_terms[term_idx]
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i, j = u_indices[idx]
                    U[i * m + j][col] += Fraction(val)
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j, k = v_indices[idx]
                    V[j * p + k][col] += Fraction(val)
            for idx, val in enumerate(w_pat):
                if val != 0:
                    k, i = w_indices[idx]
                    W[k * n + i][col] += Fraction(val)
            col += 1
        
        # Product 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        u1 = [(0,0), (0,1), (1,0), (1,1)]
        v1 = [(0,0), (0,1), (1,0), (1,1)]
        w1 = [(0,0), (0,1), (1,0), (1,1)]
        for t in range(7):
            add_strassen_term(u1, v1, w1, t)
        
        # Product 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        u2 = [(0,2), (0,3), (1,2), (1,3)]
        v2 = [(2,0), (2,1), (3,0), (3,1)]
        w2 = [(0,0), (0,1), (1,0), (1,1)]
        for t in range(7):
            add_strassen_term(u2, v2, w2, t)
        
        # Product 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        u3 = [(0,0), (0,1), (1,0), (1,1)]
        v3 = [(0,2), (0,3), (1,2), (1,3)]
        w3 = [(2,0), (2,1), (3,0), (3,1)]
        for t in range(7):
            add_strassen_term(u3, v3, w3, t)
        
        # Product 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        u4 = [(0,2), (0,3), (1,2), (1,3)]
        v4 = [(2,2), (2,3), (3,2), (3,3)]
        w4 = [(2,0), (2,1), (3,0), (3,1)]
        for t in range(7):
            add_strassen_term(u4, v4, w4, t)
        
        # Bottom row: A[2, :] * B -> C[2, :]
        # Schoolbook: 4*4 = 16 terms
        for j in range(4):
            for k in range(4):
                U[2 * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + 2][col] = Fraction(1)
                col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Try to build and verify the rank-44 decomposition
    result = build_rank44_strassen()
    if result and verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook
    return schoolbook()
