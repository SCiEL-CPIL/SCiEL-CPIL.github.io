import time
from fractions import Fraction

DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 10.0
START_TIME = time.time()


def construct():
    """Return a valid exact tensor decomposition for 3x3 * 3x3 matrix multiplication.

    Strategy: Build a rank-26 decomposition using Strassen's algorithm for the 2x2 core
    and individual terms for the cross-terms and the 1x1 core.
    Returns JSON-serializable integers.
    """
    n, m, p = 3, 3, 3

    def _verify(U, V, W, rank):
        """Verify sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all (i,j,k)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    def _build_rank26():
        """Construct a rank-26 decomposition."""
        rank = 26
        U = [[Fraction(0)] * rank for _ in range(9)]
        V = [[Fraction(0)] * rank for _ in range(9)]
        W = [[Fraction(0)] * rank for _ in range(9)]

        def add_term(u_dict, v_dict, w_dict, r):
            for (i, j), c in u_dict.items():
                U[i * 3 + j][r] += c
            for (j, k), c in v_dict.items():
                V[j * 3 + k][r] += c
            for (i, k), c in w_dict.items():
                W[k * 3 + i][r] += c

        r = 0

        # --- 7 Strassen terms for the 2x2 core (indices 0,1) ---
        # M1 = (A00+A11)(B00+B11) -> C00 + C11
        add_term({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, r); r += 1
        # M2 = (A10+A11)B00 -> C10 + C11
        add_term({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):1}, r); r += 1
        # M3 = A00(B10-B11) -> C00 - C01
        add_term({(0,0):1}, {(1,0):1,(1,1):-1}, {(0,0):1,(0,1):-1}, r); r += 1
        # M4 = A11(B01-B00) -> -C10 + C11
        add_term({(1,1):1}, {(0,1):1,(0,0):-1}, {(1,0):-1,(1,1):1}, r); r += 1
        # M5 = (A00+A01)B11 -> C00 + C01
        add_term({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):1,(0,1):1}, r); r += 1
        # M6 = (A10-A00)(B00+B01) -> C10 + C11
        add_term({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,0):1,(1,1):1}, r); r += 1
        # M7 = (A01-A11)(B10+B11) -> -C00 + C01
        add_term({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):-1,(0,1):1}, r); r += 1

        # --- 1 term for the 1x1 core (index 2) ---
        # A22 * B22 -> C22
        add_term({(2,2):1}, {(2,2):1}, {(2,2):1}, r); r += 1

        # --- 18 cross terms ---
        # We need to cover all (i,j,k) where at least one index is 2, excluding (2,2,2) which is done.
        # Total such triples: 18.
        # We assign one rank-1 term to each triple (i,j,k) with A[i,j]=1, B[j,k]=1, W[k,i]=1.
        
        cross_triples = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i == 2 and j == 2 and k == 2:
                        continue # Already handled
                    if i in (0,1) and j in (0,1) and k in (0,1):
                        continue # Already handled by Strassen
                    cross_triples.append((i,j,k))
        
        # Sort for determinism
        cross_triples.sort()
        
        for (i, j, k) in cross_triples:
            add_term({(i,j):1}, {(j,k):1}, {(i,k):1}, r)
            r += 1
            
        if r != rank:
            return None

        # Verify
        if not _verify(U, V, W, rank):
            return None
            
        # Convert to integers for JSON serialization
        U_int = [[int(U[i][j]) for j in range(rank)] for i in range(9)]
        V_int = [[int(V[i][j]) for j in range(rank)] for i in range(9)]
        W_int = [[int(W[i][j]) for j in range(rank)] for i in range(9)]
        
        return {"U": U_int, "V": V_int, "W": W_int}

    # Check time
    if time.time() - START_TIME > DEADLINE_SECONDS - SAFETY_MARGIN:
        # Fallback to rank-27 if out of time, though construction is fast
        pass

    result = _build_rank26()
    
    if result is not None:
        return result

    # Fallback: Rank-27 Schoolbook (JSON serializable)
    rank = 27
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
