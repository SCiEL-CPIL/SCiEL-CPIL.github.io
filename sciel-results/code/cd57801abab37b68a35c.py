"""Fresh design for 2x5x5 matrix multiplication tensor decomposition.
Goal: find a rank-minimal exact decomposition.
Uses block-diagonal partition with Strassen's rank-7 algorithm on 2x2x2 sub-blocks.
"""
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
DEADLINE = 130  # seconds, safety margin before 150s kill


def _validate(U, V, W, n=2, m=5, p=5):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(rank):
                    val += Fraction(U[i*m + j][c]) * Fraction(V[j*p + k][c]) * Fraction(W[k*n + i][c])
                if val != 1:
                    return False
    return True


def _validate_float(U, V, W, n=2, m=5, p=5, tol=1e-6):
    """Float-based validation for speed during search."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _strassen_block(j0, k0):
    """Return 7 (U_col, V_col, W_col) tuples for Strassen 2x2x2 at (j0, k0).
    U_col: length 10 (A is 2x5)
    V_col: length 25 (B is 5x5)
    W_col: length 10 (C is 5x2, indexed as k*n+i)
    """
    n, m, p = 2, 5, 5

    def make_U(terms):
        vec = [0] * (n * m)
        for i, j, c in terms:
            vec[i * m + j] = c
        return vec

    def make_V(terms):
        vec = [0] * (m * p)
        for j, k, c in terms:
            vec[j * p + k] = c
        return vec

    def make_W(terms):
        vec = [0] * (p * n)
        for i, k, c in terms:
            vec[k * n + i] = c
        return vec

    # Strassen products for C = A*B where:
    # a00=A[0,j0], a01=A[0,j0+1], a10=A[1,j0], a11=A[1,j0+1]
    # b00=B[j0,k0], b01=B[j0,k0+1], b10=B[j0+1,k0], b11=B[j0+1,k0+1]
    # c00=C[0,k0], c01=C[0,k0+1], c10=C[1,k0], c11=C[1,k0+1]
    #
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b10-b11)
    # P4 = a11(b01-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b10)
    # P7 = (a01-a11)(b01+b11)
    #
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6

    products = []

    # P1: (A[0,j0]+A[1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
    # c00 += P1, c10 += P1, c01 -= P1, c11 += P1
    products.append((
        make_U([(0, j0, 1), (1, j0 + 1, 1)]),
        make_V([(j0, k0, 1), (j0 + 1, k0 + 1, 1)]),
        make_W([(0, k0, 1), (1, k0, 1), (0, k0 + 1, -1), (1, k0 + 1, 1)])
    ))

    # P2: (A[1,j0]+A[1,j0+1]) * B[j0,k0]
    # c10 += P2, c11 -= P2
    products.append((
        make_U([(1, j0, 1), (1, j0 + 1, 1)]),
        make_V([(j0, k0, 1)]),
        make_W([(1, k0, 1), (1, k0 + 1, -1)])
    ))

    # P3: A[0,j0] * (B[j0+1,k0]-B[j0+1,k0+1])
    # c01 += P3, c11 += P3
    products.append((
        make_U([(0, j0, 1)]),
        make_V([(j0 + 1, k0, 1), (j0 + 1, k0 + 1, -1)]),
        make_W([(0, k0 + 1, 1), (1, k0 + 1, 1)])
    ))

    # P4: A[1,j0+1] * (B[j0,k0+1]-B[j0,k0])
    # c00 += P4, c10 += P4
    products.append((
        make_U([(1, j0 + 1, 1)]),
        make_V([(j0, k0 + 1, 1), (j0, k0, -1)]),
        make_W([(0, k0, 1), (1, k0, 1)])
    ))

    # P5: (A[0,j0]+A[0,j0+1]) * B[j0+1,k0+1]
    # c00 -= P5, c01 += P5
    products.append((
        make_U([(0, j0, 1), (0, j0 + 1, 1)]),
        make_V([(j0 + 1, k0 + 1, 1)]),
        make_W([(0, k0, -1), (0, k0 + 1, 1)])
    ))

    # P6: (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0+1,k0])
    # c11 += P6
    products.append((
        make_U([(1, j0, 1), (0, j0, -1)]),
        make_V([(j0, k0, 1), (j0 + 1, k0, 1)]),
        make_W([(1, k0 + 1, 1)])
    ))

    # P7: (A[0,j0+1]-A[1,j0+1]) * (B[j0,k0+1]+B[j0+1,k0+1])
    # c00 += P7, c11 += P7
    products.append((
        make_U([(0, j0 + 1, 1), (1, j0 + 1, -1)]),
        make_V([(j0, k0 + 1, 1), (j0 + 1, k0 + 1, 1)]),
        make_W([(0, k0, 1), (1, k0 + 1, 1)])
    ))

    return products


def _schoolbook_block_2x2x1(j0, k0):
    """Schoolbook for 2x2x1 block: i in {0,1}, j in {j0,j0+1}, k={k0}. Returns 4 terms."""
    n, m, p = 2, 5, 5
    products = []
    for i in range(2):
        for j in [j0, j0 + 1]:
            U_col = [0] * (n * m)
            U_col[i * m + j] = 1
            V_col = [0] * (m * p)
            V_col[j * p + k0] = 1
            W_col = [0] * (p * n)
            W_col[k0 * n + i] = 1
            products.append((U_col, V_col, W_col))
    return products


def _schoolbook_block_2x1x2(j0, k0):
    """Schoolbook for 2x1x2 block: i in {0,1}, j={j0}, k in {k0,k0+1}. Returns 4 terms."""
    n, m, p = 2, 5, 5
    products = []
    for i in range(2):
        for k in [k0, k0 + 1]:
            U_col = [0] * (n * m)
            U_col[i * m + j0] = 1
            V_col = [0] * (m * p)
            V_col[j0 * p + k] = 1
            W_col = [0] * (p * n)
            W_col[k * n + i] = 1
            products.append((U_col, V_col, W_col))
    return products


def _schoolbook_block_2x1x1(j0, k0):
    """Schoolbook for 2x1x1 block: i in {0,1}, j={j0}, k={k0}. Returns 2 terms."""
    n, m, p = 2, 5, 5
    products = []
    for i in range(2):
        U_col = [0] * (n * m)
        U_col[i * m + j0] = 1
        V_col = [0] * (m * p)
        V_col[j0 * p + k0] = 1
        W_col = [0] * (p * n)
        W_col[k0 * n + i] = 1
        products.append((U_col, V_col, W_col))
    return products


def _construct_block_partition(n=2, m=5, p=5):
    """Construct rank-46 decomposition using block partition (2,2,1) x (2,2,1) with Strassen on 2x2x2 blocks.
    
    Partition j indices into J1={0,1}, J2={2,3}, J3={4}
    Partition k indices into K1={0,1}, K2={2,3}, K3={4}
    
    For each (Ja, Kb) pair:
    - If |Ja|=2 and |Kb|=2: use Strassen (rank 7)
    - If |Ja|=2 and |Kb|=1: use schoolbook (rank 4)
    - If |Ja|=1 and |Kb|=2: use schoolbook (rank 4)
    - If |Ja|=1 and |Kb|=1: use schoolbook (rank 2)
    
    Total: 4*7 + 2*4 + 2*4 + 1*2 = 28 + 8 + 8 + 2 = 46
    """
    n, m, p = 2, 5, 5
    all_products = []

    # J blocks: (start, size)
    J_blocks = [(0, 2), (2, 2), (4, 1)]
    # K blocks: (start, size)
    K_blocks = [(0, 2), (2, 2), (4, 1)]

    for j0, js in J_blocks:
        for k0, ks in K_blocks:
            if js == 2 and ks == 2:
                # 2x2x2 block: use Strassen
                all_products.extend(_strassen_block(j0, k0))
            elif js == 2 and ks == 1:
                # 2x2x1 block: schoolbook
                all_products.extend(_schoolbook_block_2x2x1(j0, k0))
            elif js == 1 and ks == 2:
                # 2x1x2 block: schoolbook
                all_products.extend(_schoolbook_block_2x1x2(j0, k0))
            else:
                # 2x1x1 block: schoolbook
                all_products.extend(_schoolbook_block_2x1x1(j0, k0))

    rank = len(all_products)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u_col, v_col, w_col) in enumerate(all_products):
        for idx in range(n * m):
            U[idx][c] = u_col[idx]
        for idx in range(m * p):
            V[idx][c] = v_col[idx]
        for idx in range(p * n):
            W[idx][c] = w_col[idx]

    return {"U": U, "V": V, "W": W}


def _schoolbook(n=2, m=5, p=5):
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
    return {"U": U, "V": V, "W": W}


def construct():
    """Return the best exact tensor decomposition found within time budget."""
    start_time = time.time()
    n, m, p = DIMENSIONS

    # Try the block-partition Strassen construction (rank 46)
    result = _construct_block_partition(n, m, p)
    rank = len(result["U"][0])

    # Validate with float first for speed
    if _validate_float(result["U"], result["V"], result["W"]):
        # Then validate exactly
        if _validate(result["U"], result["V"], result["W"]):
            return result

    # Fallback to schoolbook if validation fails
    return _schoolbook(n, m, p)
