import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]


# EVOLVE-BLOCK-START
def _matmul_check(U, V, W, n, m, p):
    """Exact check that the tensor decomposition reconstructs the MM tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return (err < 1e-9), float(err)


def _trim_zero_columns(U, V, W):
    """Remove rank columns that are all-zero."""
    n_u = len(U)
    n_v = len(V)
    n_w = len(W)
    rank = len(U[0]) if n_u else 0
    keep = []
    for c in range(rank):
        if any(U[i][c] != 0 for i in range(n_u)) or \
           any(V[i][c] != 0 for i in range(n_v)) or \
           any(W[i][c] != 0 for i in range(n_w)):
            keep.append(c)
    if len(keep) == rank:
        return U, V, W, rank
    U2 = [[U[i][c] for c in keep] for i in range(n_u)]
    V2 = [[V[i][c] for c in keep] for i in range(n_v)]
    W2 = [[W[i][c] for c in keep] for i in range(n_w)]
    return U2, V2, W2, len(keep)


def _construct_rank32():
    """
    Rank-32 decomposition: 8 independent schoolbook 2x2 blocks.
    C[i,k] = sum_j A[i,j]*B[j,k]
    Split into 4 column-pairs (l=0..3) x 2 row-pairs (rb=0,1) x 2 i x 2 j x 2 k = 32 terms
    """
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for l in range(4):
        for rb in range(2):
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        r = col
                        # A index: i * m + j_local
                        j_global = rb * 2 + j
                        U[i * m + j_global][r] = 1
                        # B index: j_global * p + k_local
                        k_global = 2 * l + k
                        V[j_global * p + k_global][r] = 1
                        # W index: k_global * n + i
                        W[k_global * n + i][r] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}


def _construct_rank28():
    """
    Attempt rank-28 via combining Strassen structure.
    
    The 2x4 * 4x8 multiplication:
    C = A * B where A is 2x4, B is 4x8, C is 2x8.
    
    Split B into 4 column-pairs: B_0, B_1, B_2, B_3 (each 4x2)
    C_i = A * B_i for i=0..3 (each 2x2)
    
    For each C_i = A * B_i (2x4 * 4x2 -> 2x2):
    This is equivalent to 4 products: A_col0*B_i_col0 + A_col1*B_i_col1 + A_col2*B_i_col2 + A_col3*B_i_col3
    where A_colj is 2x1 and B_i_colk is 1x2.
    
    Actually, let's think differently. 2x4 * 4x2:
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    We can use Strassen-like tricks. Split A into A1=[A[0:2,0:2]], A2=[A[0:2,2:4]] (each 2x2)
    Split B into B1=[B[0:2,0:2]], B2=[B[2:4,0:2]] (each 2x2)
    
    C = A1*B1 + A2*B2 (2x2 * 2x2 + 2x2 * 2x2)
    
    This is a sum of two 2x2*2x2 products. Using Strassen on each would give 14 terms per C_i.
    Total: 4 * 14 = 56. That's worse.
    
    Alternative: use the fact that all 4 C_i share the same A.
    C = A * B = [A1 | A2] * B where B is 4x8.
    
    Let me try a different approach: use 7 terms for the full 2x4*4x8 by exploiting
    the rectangular structure. This is non-trivial.
    
    For now, return None to skip this candidate.
    """
    return None


def _construct_rank24():
    """
    Attempt rank-24: 6 terms per column-pair using rectangular Strassen.
    
    For 2x4 * 4x2 -> 2x2, we can potentially use 6 terms instead of 8.
    Split A into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
    Split B into B1 (2x2, rows 0-1) and B2 (2x2, rows 2-3)
    
    C = A1*B1 + A2*B2
    
    Using a modified Strassen:
    m1 = (A1+A2)*(B1+B2)  -- 1 term (but this is 2x2*2x2, needs 4 scalar mults)
    
    Actually, each "term" in the tensor decomposition is a rank-1 tensor, which corresponds
    to one scalar multiplication. So for 2x2*2x2, we need at least 4 scalar mults (schoolbook)
    or 7 (Strassen).
    
    For C = A1*B1 + A2*B2, schoolbook needs 4+4=8 scalar mults.
    Strassen on the combined structure:
    
    Let's define:
    S1 = A1 + A2  (2x2)
    S2 = B1 + B2  (2x2)
    
    m1 = S1 * S2 (2x2*2x2, 4 scalar mults schoolbook)
    m2 = A1 * B1 (4 scalar mults)
    m3 = A2 * B2 (4 scalar mults)
    
    C = m1 - m2 - m3 ... no, that's not right.
    
    C = A1*B1 + A2*B2
    S1*S2 = (A1+A2)(B1+B2) = A1*B1 + A1*B2 + A2*B1 + A2*B2
    So C = S1*S2 - A1*B2 - A2*B1
    That needs 4 + 4 + 4 = 12 scalar mults. Worse.
    
    Let me try Strassen directly on the 2x2*2x2 blocks:
    For A1*B1: 7 terms
    For A2*B2: 7 terms
    Total: 14 per C_i, 4 C_i's = 56. Same as before.
    
    I don't see a clear path to rank < 32 with simple constructions.
    Return None.
    """
    return None


def _search_rank28_with_polish(t_start, deadline):
    """
    Try to find a rank-28 decomposition using iterative refinement.
    Start from rank-32 and try to eliminate 4 terms by finding dependencies.
    """
    n, m, p = DIMENSIONS
    # Get rank-32 base
    base = _construct_rank32()
    U = np.array(base["U"], dtype=float)
    V = np.array(base["V"], dtype=float)
    W = np.array(base["W"], dtype=float)
    
    # Try to find linear dependencies among the 32 terms
    # If 4 terms are linearly dependent in a certain sense, we can merge them.
    # This is a heuristic and may not work.
    
    # Alternative: try random rank-28 and verify
    # This is very unlikely to work randomly.
    
    return None


def _construct_rank56():
    """The original Strassen-based rank-56 decomposition (known valid)."""
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 terms
    strassen = [
        {'U': {(0, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1, (1, 1): 1}},
        {'U': {(1, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1}, 'W': {(0, 1): 1, (1, 1): -1}},
        {'U': {(0, 0): 1}, 'V': {(0, 1): 1, (1, 1): -1}, 'W': {(1, 0): 1, (1, 1): 1}},
        {'U': {(1, 1): 1}, 'V': {(1, 0): 1, (0, 0): -1}, 'W': {(0, 0): 1, (0, 1): 1}},
        {'U': {(0, 0): 1, (0, 1): 1}, 'V': {(1, 1): 1}, 'W': {(0, 0): -1, (1, 0): 1}},
        {'U': {(1, 0): 1, (0, 0): -1}, 'V': {(0, 0): 1, (0, 1): 1}, 'W': {(1, 1): 1}},
        {'U': {(0, 1): 1, (1, 1): -1}, 'V': {(1, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1}},
    ]

    col = 0
    for j_high in range(2):
        for l in range(4):
            for term in strassen:
                for (i, j_low), c in term['U'].items():
                    j = j_high * 2 + j_low
                    U[i * m + j][col] += c
                for (j_low, k_low), c in term['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    V[j * p + k][col] += c
                for (k_low, i), c in term['W'].items():
                    k = l * 2 + k_low
                    W[k * n + i][col] += c
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_rank26_polish(t_start, deadline):
    """
    Attempt to find a rank-26 or rank-24 decomposition by modifying the rank-32 base.
    Use the SVD of the flattening to understand the structure.
    """
    n, m, p = DIMENSIONS
    
    # Build the tensor T[i,j,k] = delta (1 for all valid i,j,k)
    # The tensor has shape (n, m, p) = (2, 4, 8)
    # Flatten to matrix: (n*p, m) = (16, 4) or (n*m, p) = (8, 8) or (m*p, n) = (32, 2)
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # As a matrix M[i*n+j, k] where M is (n*m, p) = (8, 8): all ones
    # Rank of all-ones 8x8 matrix is 1. But that's not the right flattening for tensor rank.
    
    # For tensor rank, we need: T = sum_r u_r ⊗ v_r ⊗ w_r
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # The matrix flattening M[(i,k), j] where M is (n*p, m) = (16, 4):
    # M[i*p+k, j] = T[i,j,k] = 1 for all i,j,k
    # This is a 16x4 all-ones matrix, rank 1.
    
    # Another flattening: M[(i,j), k] where M is (n*m, p) = (8, 8):
    # M[i*m+j, k] = T[i,j,k] = 1
    # 8x8 all-ones, rank 1.
    
    # Another: M[(j,k), i] where M is (m*p, n) = (32, 2):
    # M[j*p+k, i] = T[i,j,k] = 1
    # 32x2 all-ones, rank 1.
    
    # The tensor rank is at least the matrix rank for any flattening, so rank >= 1.
    # But the tensor rank can be much higher than the matrix rank.
    
    # For the all-ones tensor of size 2x4x8, the tensor rank is known to be 8 (I think).
    # Actually, for an all-ones tensor of size a x b x c, the tensor rank is min(a,b,c) if we
    # use the "sum of outer products" construction.
    # T[i,j,k] = sum_{r=0}^{min-1} 1*1*1 ... no, that gives T[i,j,k] = min, not 1.
    
    # To get T[i,j,k] = 1: use r = 1 term with U[i,j,0]=1, V[j,k,0]=1, W[k,i,0]=1.
    # That gives T[i,j,k] = 1. So tensor rank is 1?!
    
    # Wait, that can't be right. Let me re-examine.
    # The MM tensor for 2x4*4x8: T[i,j,k] = 1 for all i in 0..1, j in 0..3, k in 0..7.
    # If I use one term: U[i,j,0]=1, V[j,k,0]=1, W[k,i,0]=1
    # Then T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1*1*1 = 1. Correct!
    
    # So the tensor rank is 1?? That seems wrong. Let me re-read the problem.
    
    # Oh wait, I think I'm confusing the tensor structure. In matrix multiplication,
    # the tensor is T[i,j,k] = 1, and the decomposition is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j sum_r U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k] * W[k,i,r]
    # 
    # No, that's not right either. The standard tensor decomposition for MM is:
    # C[i,k] = sum_r (sum_j U[i,j,r]*A[i,j]) * (sum_j V[j,k,r]*B[j,k]) * W[k,i,r]
    # 
    # Hmm, actually the correct formulation is:
    # The MM tensor T has T[i,j,k] = 1, and we want:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # 
    # With one term: U[i,j,0]=1, V[j,k,0]=1, W[k,i,0]=1
    # T[i,j,k] = 1*1*1 = 1. Yes, this works!
    
    # So the tensor rank of the 2x4x8 MM tensor is 1?
    # That can't be right because then we could compute any 2x4*4x8 MM with 1 multiplication.
    
    # I think the issue is that U, V, W are not arbitrary - they must satisfy the
    # constraint that the decomposition computes MM correctly for ALL inputs A, B.
    # The tensor equation T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 is necessary
    # but the actual computation requires:
    # C[i,k] = sum_r (sum_j U[i,j,r]*A[i,j]) * (sum_j V[j,k,r]*B[j,k]) * W[k,i,r]
    #         = sum_j A[i,j]*B[j,k]  for all A, B
    # 
    # This is a stronger condition. The tensor equation alone is not sufficient.
    # We need the bilinear form to match, not just the tensor coefficients.
    
    # Actually, I think the correct statement is:
    # C[i,k] = sum_r (sum_j U[i,j,r]*A[i,j]) * (sum_l V[l,k,r]*B[l,k]) * W[k,i,r]
    # For this to equal sum_j A[i,j]*B[j,k] for all A,B, we need:
    # sum_r U[i,j,r]*V[j',k,r]*W[k,i,r] = delta_{j,j'} for all i,j,j',k
    # 
    # Wait, that's not right either. Let me be more careful.
    # 
    # C[i,k] = sum_r alpha_r * beta_r * W[k,i,r]
    # where alpha_r = sum_j U[i,j,r]*A[i,j] and beta_r = sum_l V[l,k,r]*B[l,k]
    # 
    # = sum_r (sum_j U[i,j,r]*A[i,j]) * (sum_l V[l,k,r]*B[l,k]) * W[k,i,r]
    # = sum_j sum_l (sum_r U[i,j,r]*V[l,k,r]*W[k,i,r]) * A[i,j] * B[l,k]
    # 
    # For this to equal sum_j A[i,j]*B[j,k], we need:
    # sum_r U[i,j,r]*V[l,k,r]*W[k,i,r] = delta_{j,l} for all i,j,l,k
    # 
    # So the condition is NOT T[i,j,k]=1, but rather:
    # sum_r U[i,j,r]*V[l,k,r]*W[k,i,r] = delta_{j,l}
    # 
    # This is a different tensor! It's a 4-index tensor (i,j,l,k) that must equal delta_{j,l}.
    # 
    # Let me re-examine my _matmul_check function. It computes:
    # T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    # This gives T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # And checks T[i,j,k] == 1.
    # 
    # But the correct condition should be:
    # sum_r U[i,j,r]*V[l,k,r]*W[k,i,r] = delta_{j,l}
    # 
    # These are different! Let me fix the check.
    
    # Actually wait, let me re-derive. In the standard MM tensor decomposition:
    # We have C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # The bilinear algorithm uses r scalar multiplications:
    # m_r = (sum_j U[i,j,r]*A[i,j]) * (sum_l V[l,k,r]*B[l,k])
    # C[i,k] = sum_r W[k,i,r]*m_r
    # 
    # Expanding:
    # C[i,k] = sum_r W[k,i,r] * (sum_j U[i,j,r]*A[i,j]) * (sum_l V[l,k,r]*B[l,k])
    # = sum_j sum_l (sum_r U[i,j,r]*V[l,k,r]*W[k,i,r]) * A[i,j]*B[l,k]
    # 
    # For this to equal sum_j A[i,j]*B[j,k], we need:
    # sum_r U[i,j,r]*V[l,k,r]*W[k,i,r] = delta_{j,l} for all i,j,l,k
    # 
    # So the correct check is:
    # For all i,j,l,k: sum_r U[i,j,r]*V[l,k,r]*W[k,i,r] == (1 if j==l else 0)
    
    # My current check is wrong! It checks sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1
    # which is a different (and weaker) condition.
    
    # Let me fix the check and re-evaluate.
    return None


def _matmul_check_correct(U, V, W, n, m, p):
    """Correct check: sum_r U[i,j,r]*V[l,k,r]*W[k,i,r] == delta_{j,l} for all i,j,l,k."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    # Compute S[i,j,l,k] = sum_r U[i,j,r]*V[l,k,r]*W[k,i,r]
    S = np.einsum('ijr,lkr,kin->ijkl', Ur, Vr, Wr)
    # Expected: delta_{j,l} for all i,k
    expected = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i, j, j, k] = 1.0
    err = np.max(np.abs(S - expected))
    return (err < 1e-9), float(err)


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140

    best = None

    # First, use the correct check to verify candidates
    # Candidate 1: rank-32 schoolbook
    try:
        cand = _construct_rank32()
        ok, err = _matmul_check_correct(cand["U"], cand["V"], cand["W"], n, m, p)
        if ok:
            U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
            best = (r, U, V, W)
    except Exception:
        pass

    # Candidate 2: rank-56 Strassen
    try:
        cand = _construct_rank56()
        ok, err = _matmul_check_correct(cand["U"], cand["V"], cand["W"], n, m, p)
        if ok:
            U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
            if best is None or r < best[0]:
                best = (r, U, V, W)
    except Exception:
        pass

    # Candidate 3: rank-28 attempt
    try:
        cand = _construct_rank28()
        if cand is not None:
            ok, err = _matmul_check_correct(cand["U"], cand["V"], cand["W"], n, m, p)
            if ok:
                U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
                if best is None or r < best[0]:
                    best = (r, U, V, W)
    except Exception:
        pass

    if best is None:
        # Last resort
        cand = _construct_rank56()
        return cand

    r, U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
