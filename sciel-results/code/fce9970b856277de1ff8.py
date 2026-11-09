import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition for n x m x p matrix multiplication.
    
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The convention is W[k*n+i, r] for output C[i,k].
    
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    For all other (a,b,c) triples:
        sum_r U[a, r] * V[b, r] * W[c, r] = 0
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :] * V[b, :], W[c, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    
    # Check invalid entries (spot check for efficiency)
    # For 3x3x8, we can check a subset
    for a in range(n * m):
        i1 = a // m
        j1 = a % m
        for b in range(m * p):
            j2 = b // p
            k2 = b % p
            for c in range(p * n):
                k3 = c // n
                i3 = c % n
                if i1 == i3 and j1 == j2 and k2 == k3:
                    continue
                val = np.dot(U[a, :] * V[b, :], W[c, :])
                if abs(val) > 1e-6:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7.
    
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 products for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    # Product 1: (a)(e) -> contributes to C[0,0]
    # U[0,0]=1, V[0,0]=1, W[0,0]=1  (a*e for C[0,0])
    # But Strassen combines these.
    
    # Let me use the standard Strassen formulation:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a-b)(f+h)
    # M4 = (a)(d-e)
    # M5 = (c)(h-g)
    # M6 = (b-a)(e+h)  -- wait, let me be more careful
    # M6 = (b)(g)  -- no
    
    # Standard Strassen:
    # M1 = (a + d) * (e + h)
    # M2 = (c + d) * e
    # M3 = a * (f + h)
    # M4 = d * (e - f)
    # M5 = (a - b) * (e + f)
    # M6 = (c - a) * (g + h)
    # M7 = (b - d) * (g + h)
    
    # C[0,0] = M1 - M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 - M6
    
    # A indices: 0=a, 1=b, 2=c, 3=d
    # B indices: 0=e, 1=f, 2=g, 3=h
    # C indices: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
    
    # M1: (a+d)(e+h)
    # U: a=1, d=1 -> U[0,0]=1, U[3,0]=1
    # V: e=1, h=1 -> V[0,0]=1, V[3,0]=1
    # W: C[0,0]=1, C[1,1]=1 -> W[0,0]=1, W[3,0]=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (c+d)(e)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: a*(f+h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = 1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: d*(e-f)
    U[3, 3] = 1
    V[0, 3] = 1; V[1, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5: (a-b)*(e+f)
    U[0, 4] = 1; U[1, 4] = -1
    V[0, 4] = 1; V[1, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (c-a)*(g+h)
    U[2, 5] = 1; U[0, 5] = -1
    V[2, 5] = 1; V[3, 5] = 1
    W[3, 5] = -1
    
    # M7: (b-d)*(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_3x3x8():
    """
    Construct a decomposition for 3x3x8 matrix multiplication.
    
    Strategy: Use Strassen-like block decomposition.
    Split the 3x3 x 3x8 into blocks.
    
    Approach: Treat the 3x3 matrix as having a 2x2 block in the top-left,
    and handle the remaining parts separately.
    
    A (3x3) = [[A11, A12, A13],
               [A21, A22, A23],
               [A31, A32, A33]]
    
    B (3x8) = [[B11, B12, ..., B18],
               [B21, B22, ..., B28],
               [B31, B32, ..., B38]]
    
    C = A * B (3x8)
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can group the j index: j=0,1 form a 2x2 block with the first two rows/cols of A,
    and j=2 is separate.
    
    For j=0,1: C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    For j=2: C[i,k] += A[i,2]*B[2,k]
    
    The j=0,1 part involves a 3x2 matrix (rows 0,1,2 of A, cols 0,1) times a 2x8 matrix (rows 0,1 of B).
    The j=2 part involves a 3x1 matrix (col 2 of A) times a 1x8 matrix (row 2 of B).
    
    Let me think differently. Let's use the fact that:
    - 3x3x3 has known rank 19
    - We can decompose 3x3x8 as 3x3x3 + 3x3x3 + 3x3x2
    
    But that gives rank 19+19+? which is not optimal.
    
    Better approach: Use Strassen on the 3x3x3 part and handle the extra columns.
    
    Actually, let me try a different decomposition:
    Split the 3x3 input A into a 2x2 block and a 1x3 row:
    A = [[A22, A21],
         [A12, A11]]
    where A22 is 2x2, A21 is 2x1, A12 is 1x2, A11 is 1x1.
    
    Similarly, B (3x8) can be split as:
    B = [[B28],
         [B18]]
    where B28 is 2x8, B18 is 1x8.
    
    Then C = A * B = [[A22, A21], [A12, A11]] * [[B28], [B18]]
    = [[A22*B28 + A21*B18],
       [A12*B28 + A11*B18]]
    
    This gives us:
    - C_top (2x8) = A22*B28 + A21*B18
    - C_bottom (1x8) = A12*B28 + A11*B18
    
    For C_top: A22 is 2x2, B28 is 2x8, A21 is 2x1, B18 is 1x8
    A22*B28 is 2x2 * 2x8 = can use Strassen-like for each column group
    A21*B18 is 2x1 * 1x8 = outer product, rank 2 per column... 
    
    This is getting complex. Let me just use the schoolbook approach but with some optimization.
    
    Actually, the simplest improvement is to use Strassen for the 2x2x2 sub-problems.
    
    Let me try: decompose 3x3x8 by splitting into:
    - 2x3x8 (top 2 rows of A) + 1x3x8 (bottom row of A)
    
    For 2x3x8: A is 2x3, B is 3x8, C is 2x8
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    Split j into {0,1} and {2}:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    The first two terms form a 2x2 * 2x8 product (using rows 0,1 of A and rows 0,1 of B).
    For each column k, we have a 2x2 * 2x1 product, which can use Strassen (rank 7 per column).
    But that's 7*8 = 56 just for that part, which is worse than schoolbook 2*2*8=32.
    
    So Strassen doesn't help for thin matrices. Let me just use schoolbook.
    
    Schoolbook rank for 3x3x8 = 3*3*8 = 72.
    """
    n, m, p = 3, 3, 8
    U, V, W = _schoolbook(n, m, p)
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Try to construct a better decomposition
    try:
        U, V, W = _construct_3x3x8()
        rank = U.shape[1]
        
        # Verify the decomposition
        if _verify_decomposition(U, V, W, n, m, p):
            # Convert to lists for the return format
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Return schoolbook as fallback
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
