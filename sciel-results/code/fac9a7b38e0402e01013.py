"""Matrix multiplication 3,4,7 - block decomposition with Strassen on 2x2x2 blocks."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 7]
DEADLINE = 120.0
START_TIME = time.time()

def check_deadline():
    return time.time() - START_TIME > DEADLINE - 2.0

def strassen_222_terms():
    """Return list of (u, v, w) tuples for Strassen 2x2x2.
    U indexed as [i*2+j], V as [j*2+k], W as [k*2+i]."""
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    terms = []
    # M1: A: [1,0,0,1], B: [1,0,0,1], C: C00=1,C11=1 -> W[0]=1, W[3]=1
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2: A: [0,0,1,1], B: [1,0,0,0], C: C10=1, C11=-1 -> W[1]=1, W[3]=-1
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,-1]))
    # M3: A: [1,0,0,0], B: [0,1,0,-1], C: C01=1, C11=1 -> W[2]=1, W[3]=1
    terms.append(([1,0,0,0], [0,1,0,-1], [0,0,1,1]))
    # M4: A: [0,0,0,1], B: [0,0,1,-1], C: C00=1, C10=1 -> W[0]=1, W[1]=1
    terms.append(([0,0,0,1], [0,0,1,-1], [1,1,0,0]))
    # M5: A: [1,1,0,0], B: [0,0,0,1], C: C00=-1, C01=1 -> W[0]=-1, W[2]=1
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
    # M6: A: [0,-1,1,0], B: [1,1,0,0], C: C11=1 -> W[3]=1
    terms.append(([0,-1,1,0], [1,1,0,0], [0,0,0,1]))
    # M7: A: [0,1,-1,0], B: [0,0,1,1], C: C00=1 -> W[0]=1
    terms.append(([0,1,-1,0], [0,0,1,1], [1,0,0,0]))
    
    return terms

def schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m by m x p. Returns list of (u,v,w) with lengths n*m, m*p, p*n."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0]*(n*m)
                v = [0]*(m*p)
                w = [0]*(p*n)
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                terms.append((u, v, w))
    return terms

def embed_terms(terms, offset_i, offset_j, offset_k, n_total, m_total, p_total, local_n, local_m, local_p):
    """Embed local terms into global space.
    Local indices: i in [0,local_n), j in [0,local_m), k in [0,local_p)
    Global: i_global = offset_i + i, j_global = offset_j + j, k_global = offset_k + k
    """
    embedded = []
    for u, v, w in terms:
        gu = [0]*(n_total*m_total)
        gv = [0]*(m_total*p_total)
        gw = [0]*(p_total*n_total)
        for li in range(local_n):
            for lj in range(local_m):
                gi = offset_i + li
                gj = offset_j + lj
                gu[gi*m_total + gj] = u[li*local_m + lj]
        for lj in range(local_m):
            for lk in range(local_p):
                gj = offset_j + lj
                gk = offset_k + lk
                gv[gj*p_total + gk] = v[lj*local_p + lk]
        for lk in range(local_p):
            for li in range(local_n):
                gk = offset_k + lk
                gi = offset_i + li
                gw[gk*n_total + gi] = w[lk*local_n + li]
        embedded.append((gu, gv, gw))
    return embedded

def validate_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[:,c] outer V[:,c] matches matrix mult tensor with W[:,c]."""
    rank = len(U[0])
    # Build the tensor from decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Compare with identity tensor: T[i,j,k] should equal 1 if this is schoolbook, but actually
    # the decomposition represents: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # should satisfy: for each (i,k), sum_j T[i,j,k] A[i,j]... no wait.
    
    # Actually the correct check: the tensor T of shape (n*m, m*p, p*n) with
    # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # should equal the "matrix multiplication tensor" where T[a,b,c] = 1 if a,b,c correspond to
    # (i,j), (j,k), (k,i) for the same i,j,k, else 0.
    
    # More precisely: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and T = 0 otherwise.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = 0
                for cc in range(rank):
                    val += U[a][cc] * V[b][cc] * W[c][cc]
                if abs(val - 1) > tol:
                    return False, f"Mismatch at ({i},{j},{k}): {val}"
    # Check off-diagonal entries are zero
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                # Skip if this corresponds to a valid (i,j,k) triple
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if j == j2 and k == k2 and i == i2:
                    continue
                val = 0
                for cc in range(rank):
                    val += U[a][cc] * V[b][cc] * W[c][cc]
                if abs(val) > tol:
                    return False, f"Non-zero off-diagonal at ({a},{b},{c}): {val}"
    return True, ""

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Strategy: Split n=3 into 2+1, m=4 into 2+2, p=7 into 4+3
    # Block (0,0): 2x2x4 -> use Strassen on 2x2x2 twice (p=4 split into 2+2)
    # Block (0,1): 2x2x3 -> use Strassen on 2x2x2 once + schoolbook 2x2x1
    # Block (1,0): 1x2x4 -> schoolbook
    # Block (1,1): 1x2x3 -> schoolbook
    
    # Actually let's think about this more carefully.
    # n=3, m=4, p=7
    # Split: n = 2+1, m = 2+2, p = 4+3
    
    # Block (n0,m0,p0) = (2,2,4): Strassen 2x2x2 for p-part 0 (2 cols) + Strassen 2x2x2 for p-part 1 (2 cols)
    #   Rank = 7 + 7 = 14
    # Block (n0,m1,p0) = (2,2,4): same = 14
    # Block (n0,m0,p1) = (2,2,3): Strassen 2x2x2 (2 cols) + schoolbook 2x2x1 (1 col) = 7+4 = 11
    # Block (n0,m1,p1) = (2,2,3): same = 11
    # Block (n1,m0,p0) = (1,2,4): schoolbook = 1*2*4 = 8
    # Block (n1,m1,p0) = (1,2,4): schoolbook = 8
    # Block (n1,m0,p1) = (1,2,3): schoolbook = 6
    # Block (n1,m1,p1) = (1,2,3): schoolbook = 6
    # Total: 14+14+11+11+8+8+6+6 = 78
    
    all_terms = []
    
    strassen = strassen_222_terms()
    
    # Block (0,0,0): n=2, m=2, p=4, offset (0,0,0)
    # p split into 2+2: first 2 cols use Strassen, next 2 cols use Strassen
    terms_b000 = []
    # p-part 0 (k=0,1): Strassen 2x2x2
    terms_b000.extend(strassen)
    # p-part 1 (k=2,3): Strassen 2x2x2
    terms_b000.extend(strassen)
    # Embed with local_p=4
    embedded = embed_terms(terms_b000, 0, 0, 0, n, m, p, 2, 2, 4)
    all_terms.extend(embedded)
    
    # Block (0,1,0): n=2, m=2, p=4, offset (0,2,0)
    terms_b010 = []
    terms_b010.extend(strassen)  # p-part 0
    terms_b010.extend(strassen)  # p-part 1
    embedded = embed_terms(terms_b010, 0, 2, 0, n, m, p, 2, 2, 4)
    all_terms.extend(embedded)
    
    # Block (0,0,1): n=2, m=2, p=3, offset (0,0,4)
    # p split into 2+1: Strassen for first 2, schoolbook for last 1
    terms_b001 = []
    terms_b001.extend(strassen)  # p-part 0 (k=0,1)
    # p-part 1 (k=2): schoolbook 2x2x1
    sb = schoolbook_block(2, 2, 1)
    terms_b001.extend(sb)
    embedded = embed_terms(terms_b001, 0, 0, 4, n, m, p, 2, 2, 3)
    all_terms.extend(embedded)
    
    # Block (0,1,1): n=2, m=2, p=3, offset (0,2,4)
    terms_b011 = []
    terms_b011.extend(strassen)
    sb = schoolbook_block(2, 2, 1)
    terms_b011.extend(sb)
    embedded = embed_terms(terms_b011, 0, 2, 4, n, m, p, 2, 2, 3)
    all_terms.extend(embedded)
    
    # Block (1,0,0): n=1, m=2, p=4, offset (2,0,0)
    sb = schoolbook_block(1, 2, 4)
    embedded = embed_terms(sb, 2, 0, 0, n, m, p, 1, 2, 4)
    all_terms.extend(embedded)
    
    # Block (1,1,0): n=1, m=2, p=4, offset (2,2,0)
    sb = schoolbook_block(1, 2, 4)
    embedded = embed_terms(sb, 2, 2, 0, n, m, p, 1, 2, 4)
    all_terms.extend(embedded)
    
    # Block (1,0,1): n=1, m=2, p=3, offset (2,0,4)
    sb = schoolbook_block(1, 2, 3)
    embedded = embed_terms(sb, 2, 0, 4, n, m, p, 1, 2, 3)
    all_terms.extend(embedded)
    
    # Block (1,1,1): n=1, m=2, p=3, offset (2,2,4)
    sb = schoolbook_block(1, 2, 3)
    embedded = embed_terms(sb, 2, 2, 4, n, m, p, 1, 2, 3)
    all_terms.extend(embedded)
    
    # Now assemble U, V, W
    rank = len(all_terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c, (u, v, w) in enumerate(all_terms):
        for idx in range(n*m):
            U[idx][c] = u[idx]
        for idx in range(m*p):
            V[idx][c] = v[idx]
        for idx in range(p*n):
            W[idx][c] = w[idx]
    
    # Validate
    valid, msg = validate_decomposition(U, V, W, n, m, p)
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
