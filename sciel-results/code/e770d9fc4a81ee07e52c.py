"""Combined program: recursive block decomposition with proper Strassen columns."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    U_dim = n * m  # 12
    V_dim = m * p  # 24
    W_dim = p * n  # 18

    # Build target tensor for verification
    target = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the target tensor."""
        M = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
        for r in range(U.shape[1]):
            M += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(M, target, atol=1e-9)

    def to_integer_list(M):
        """Convert numpy array to list of lists with integer values."""
        return [[int(round(x)) for x in row] for row in M]

    # We represent the decomposition as a list of terms.
    # Each term is a single multiplication (column r) with:
    #   a_coeffs: dict {(i,j): coeff}  -> U[i*m+j, r]
    #   b_coeffs: dict {(j,k): coeff}  -> V[j*p+k, r]
    #   c_coeffs: dict {(i,k): coeff}  -> W[k*n+i, r]
    #
    # The tensor identity: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # For a term with a_coeffs, b_coeffs, c_coeffs:
    #   contribution to T[i,j,k] = a_coeffs[(i,j)] * b_coeffs[(j,k)] * c_coeffs[(i,k)]
    #   (0 if any key missing)
    #
    # For schoolbook: a={(i,j):1}, b={(j,k):1}, c={(i,k):1} -> contributes 1 to T[i,j,k].
    # For Strassen: linear combos that collectively reconstruct the product.

    terms = []  # list of (a_dict, b_dict, c_dict)

    def add_term(a_dict, b_dict, c_dict):
        terms.append((a_dict, b_dict, c_dict))

    def strassen_2x2x2(i0, j0, k0):
        """Add 7 Strassen terms for 2x2x2 block at offset (i0, j0, k0).
        A block: rows i0..i0+1, cols j0..j0+1
        B block: rows j0..j0+1, cols k0..k0+1
        C block: rows i0..i0+1, cols k0..k0+1
        """
        # M1 = (A11+A22)(B11+B22)
        add_term(
            {(i0, j0): 1, (i0 + 1, j0 + 1): 1},
            {(j0, k0): 1, (j0 + 1, k0 + 1): 1},
            {(i0, k0): 1, (i0 + 1, k0 + 1): 1}
        )
        # M2 = (A21+A22)B11
        add_term(
            {(i0 + 1, j0): 1, (i0 + 1, j0 + 1): 1},
            {(j0, k0): 1},
            {(i0 + 1, k0): 1, (i0 + 1, k0 + 1): -1}
        )
        # M3 = A11(B12-B22)
        add_term(
            {(i0, j0): 1},
            {(j0, k0 + 1): 1, (j0 + 1, k0 + 1): -1},
            {(i0, k0 + 1): 1, (i0 + 1, k0 + 1): 1}
        )
        # M4 = A22(B21-B11)
        add_term(
            {(i0 + 1, j0 + 1): 1},
            {(j0 + 1, k0): 1, (j0, k0): -1},
            {(i0, k0): 1, (i0 + 1, k0): 1}
        )
        # M5 = (A11+A12)B22
        add_term(
            {(i0, j0): 1, (i0, j0 + 1): 1},
            {(j0 + 1, k0 + 1): 1},
            {(i0, k0): -1, (i0, k0 + 1): 1}
        )
        # M6 = (A21-A11)(B11+B12)
        add_term(
            {(i0 + 1, j0): 1, (i0, j0): -1},
            {(j0, k0): 1, (j0, k0 + 1): 1},
            {(i0 + 1, k0 + 1): 1}
        )
        # M7 = (A12-A22)(B21+B22)
        add_term(
            {(i0, j0 + 1): 1, (i0 + 1, j0 + 1): -1},
            {(j0 + 1, k0): 1, (j0 + 1, k0 + 1): 1},
            {(i0, k0): 1}
        )

    def schoolbook_block(i0, i1, j0, j1, k0, k1):
        """Add schoolbook terms for A[i0:i1, j0:j1] * B[j0:j1, k0:k1] -> C[i0:i1, k0:k1]."""
        for i in range(i0, i1):
            for j in range(j0, j1):
                for k in range(k0, k1):
                    add_term(
                        {(i, j): 1},
                        {(j, k): 1},
                        {(i, k): 1}
                    )

    # Recursive block decomposition for multiplying A[n x m] * B[m x p] -> C[n x p]
    # We split the middle dimension (m) and the output dimensions as needed.
    # Strategy: for a block A[ir x jr] * B[jr x kr] -> C[ir x kr]:
    #   - If all dimensions are 2, use Strassen (7 terms).
    #   - Otherwise, split the largest dimension and recurse.

    def decompose(ir, jr, kr):
        """Decompose A[i0:i0+ir, j0:j0+jr] * B[j0:j0+jr, k0:k0+kr] -> C[i0:i0+ir, k0:k0+kr].
        Here ir, jr, kr are the dimensions (heights/widths) of the blocks.
        We use absolute offsets passed via closure... 
        Actually, let's use explicit offsets to be clean.
        """
        pass

    # Let me use explicit offsets for clarity.
    def decompose(i0, i1, j0, j1, k0, k1):
        """Decompose A[i0:i1, j0:j1] * B[j0:j1, k0:k1] -> C[i0:i1, k0:k1].
        ir = i1-i0, jr = j1-j0, kr = k1-k0.
        """
        ir = i1 - i0
        jr = j1 - j0
        kr = k1 - k0

        # Check time
        if time.time() > deadline:
            return

        # Base case: all dimensions are 2 -> use Strassen
        if ir == 2 and jr == 2 and kr == 2:
            strassen_2x2x2(i0, j0, k0)
            return

        # Base case: any dimension is 1 -> schoolbook (can't do better easily)
        if ir == 1 or jr == 1 or kr == 1:
            schoolbook_block(i0, i1, j0, j1, k0, k1)
            return

        # Recursive case: split the largest dimension
        # We prefer to split the middle dimension (jr) since it couples A and B.
        # But we can also split ir or kr.
        # Let's try splitting jr first if it's the largest, then kr, then ir.
        
        # Determine which dimension to split
        dims = [(jr, 'j'), (kr, 'k'), (ir, 'i')]
        dims.sort(reverse=True)
        split_dim = dims[0][1]

        if split_dim == 'j':
            # Split middle dimension: j0..j1 into j0..jm and jm..j1
            jm = j0 + jr // 2
            decompose(i0, i1, j0, jm, k0, k1)
            decompose(i0, i1, jm, j1, k0, k1)
        elif split_dim == 'k':
            # Split output columns: k0..k1 into k0..km and km..k1
            km = k0 + kr // 2
            decompose(i0, i1, j0, j1, k0, km)
            decompose(i0, i1, j0, j1, km, k1)
        else:  # split_dim == 'i'
            # Split output rows: i0..i1 into i0..im and im..i1
            im = i0 + ir // 2
            decompose(i0, im, j0, j1, k0, k1)
            decompose(im, i1, j0, j1, k0, k1)

    # Start the decomposition for the full 3x4x6 problem
    decompose(0, n, 0, m, 0, p)

    # Build U, V, W from terms
    rank = len(terms)
    U = np.zeros((U_dim, rank))
    V = np.zeros((V_dim, rank))
    W = np.zeros((W_dim, rank))

    for r, (a_dict, b_dict, c_dict) in enumerate(terms):
        for (i, j), coeff in a_dict.items():
            U[i * m + j, r] = coeff
        for (j, k), coeff in b_dict.items():
            V[j * p + k, r] = coeff
        for (i, k), coeff in c_dict.items():
            W[k * n + i, r] = coeff

    # Verify
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((U_dim, rank_school))
        V = np.zeros((V_dim, rank_school))
        W = np.zeros((W_dim, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    return {"U": to_integer_list(U), "V": to_integer_list(V), "W": to_integer_list(W)}
# EVOLVE-BLOCK-END
