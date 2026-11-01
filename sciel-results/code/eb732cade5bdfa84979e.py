"""Matrix multiplication 2,2,7 - rank-25 block decomposition."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
import time
_START = time.time()
_DEADLINE = 150.0  # evaluator wall-clock; keep a safety margin below this

def construct():
    # 2 x (2*7) = 2x2 times 2x7. Output: p*n = 7*2 = 14 rows, W[k*n+i, r].
    # Decomposition: three Strassen blocks (2 columns each) + one schoolbook block (1 column).
    n, m, p = DIMENSIONS  # 2,2,7
    # p=7 columns of the second matrix. Group: cols 0,1 (block A), 2,3 (block B), 4,5 (block C), 6 (block D).
    # Each Strassen block: rank 7. Schoolbook block: rank 4. Total = 7*3 + 4 = 25.

    U_rows = n * m  # 4
    V_rows = m * p  # 14
    W_rows = p * n  # 14

    entries = []  # list of (u_row, v_row, w_row, u_val, v_val, w_val)

    def strassen_block(col0, col1):
        # Multiply 2x2 A by the 2-column submatrix [col0, col1] -> 2x2 result for those cols.
        # Result R[i, k] for i in 0..1, k in 0..1 (k local = global col - col0).
        # W index = k_global*n + i = (col0 + k)*2 + i.
        # Strassen rank-7:
        # M1 = (A11 - A21)(B21 + B22)
        # M2 = (A11 + A21)B21
        # M3 = A12(B12 - B22)
        # M4 = (A22 - A12)B12
        # M5 = (A11 + A12)B11
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # R11 = M5 + M4 - M2 - M7
        # R12 = M1 + M3
        # R21 = M2 + M6
        # R22 = M5 + M3 - M1 - M4
        # A index = i*m + j = i*2 + j -> A11=0,A12=1,A21=2,A22=3
        # B (2x2): B11 at col0 row0 -> V index = j*p + k_global; j row of A (0..1), k_global col.
        #   B11 = j=0,k=col0 ; B12 = j=0,k=col1 ; B21 = j=1,k=col0 ; B22 = j=1,k=col1
        A = {(0,0):0,(0,1):1,(1,0):2,(1,1):3}
        def Vidx(j, k): return j*p + k
        def Widx(k, i): return k*n + i
        def Aidx(i, j): return i*m + j
        M = [
            # (name, A_factor, B_factor) where each is list of (row,col,sign)
            ("M1", [(-1,(1,0)), (1,(0,0))], [(1,(1,0)), (1,(1,1))]),
            ("M2", [(1,(0,0)), (1,(1,0))], [(1,(1,0))]),
            ("M3", [(1,(0,1))], [(1,(0,1)), (-1,(1,1))]),
            ("M4", [(1,(1,1)), (-1,(0,1))], [(1,(0,1))]),
            ("M5", [(1,(0,0)), (1,(0,1))], [(1,(0,0))]),
            ("M6", [(1,(1,0)), (-1,(0,0))], [(1,(0,0)), (1,(0,1))]),
            ("M7", [(1,(0,1)), (-1,(1,1))], [(1,(1,0)), (1,(1,1))]),
        ]
        # Combine A factors into a U row (sparse), B factors into a V row.
        from collections import defaultdict
        for (name, A_terms, B_terms) in M:
            u = defaultdict(float); v = defaultdict(float)
            for (sign,(i,j)) in A_terms:
                u[Aidx(i,j)] += sign
            for (sign,(j,k)) in B_terms:
                v[Vidx(j, col0 + (0 if k==0 else 1))] += sign
            # result contributions
            if name == "M1":
                res = [(1,(0,1)), (1,(1,0))]  # R12 += M1, R21 += M1
            elif name == "M2":
                res = [(-1,(0,0)), (1,(1,0))]
            elif name == "M3":
                res = [(1,(0,1)), (1,(1,1))]
            elif name == "M4":
                res = [(-1,(0,0)), (-1,(1,1))]
            elif name == "M5":
                res = [(1,(0,0)), (1,(1,1))]
            elif name == "M6":
                res = [(1,(1,0))]
            elif name == "M7":
                res = [(-1,(0,0)), (-1,(0,1))]
            for (sign,(i,k)) in res:
                w = Widx(col0 + k, i)
                entries.append((dict(u), dict(v), {w: sign}))
    # Hmm, storing dict per entry is heavy; but rank 25 is tiny.

    # Simpler: build U, V, W directly with integer lists.
    U = [[0]*25 for _ in range(U_rows)]
    V = [[0]*25 for _ in range(V_rows)]
    W = [[0]*25 for _ in range(W_rows)]

    def strassen_fill(col0, col1, base):
        n,m,p = 2,2,7
        def Aidx(i,j): return i*2+j
        def Vidx(j,k): return j*7+k
        def Widx(k,i): return k*2+i
        def setU(r,val,c): U[r][c]=val
        def setV(r,val,c): V[r][c]=val
        def setW(r,val,c): W[r][c]=val
        # Define each Strassen matrix's U (A-part) and V (B-part) rows and W contributions.
        # A-part rows (u): for each M, a vector over 4 A entries.
        # B-part rows (v): vector over the 4 B entries of this block.
        blocks = [
            # (u_vec, v_vec, [(w_row, coeff)])
            ([0,-1,1,0],[0,0,1,1],[(Widx(col0+1,0),1),(Widx(col0,1),1)]),          # M1
            ([0,1,1,0],[0,0,1,0],[(Widx(col0,0),-1),(Widx(col0+1,1),1)]),            # M2
            ([0,1,0,0],[0,1,-1,0],[(Widx(col0+1,0),1),(Widx(col0+1,1),1)]),          # M3
            ([0,0,-1,1],[0,1,0,0],[(Widx(col0,0),-1),(Widx(col0+1,1),-1)]),          # M4
            ([0,1,1,0],[1,0,0,0],[(Widx(col0,0),1),(Widx(col0+1,1),1)]),             # M5
            ([0,-1,1,0],[1,1,0,0],[(Widx(col0+1,1),1)]),                              # M6
            ([0,1,-1,0],[0,1,1,0],[(Widx(col0,0),-1),(Widx(col0+1,0),-1)]),          # M7
        ]
        for u_vec, v_vec, wterms in blocks:
            for r,val in enumerate(u_vec):
                if val: U[r][base]=val
            # v_vec indexed by (j,klocal) where klocal 0->col0,1->col1
            for j in range(2):
                for kloc in range(2):
                    val = v_vec[j*2+kloc]
                    if val: V[Vidx(j, col0+kloc)][base]=val
            for (wr, coeff) in wterms:
                W[wr][base]=coeff
            base += 1
        return base

    def schoolbook_fill(col0, base):
        # single column col0: result R[i, col0] = sum_j A[i,j]*B[j,col0]
        # W index = col0*2 + i
        n,m,p=2,2,7
        def Vidx(j,k): return j*7+k
        def Widx(k,i): return k*2+i
        for i in range(2):
            for j in range(2):
                U[i*2+j][base]=1
                V[Vidx(j,col0)][base]=1
                W[Widx(col0,i)][base]=1
                base += 1
        return base

    base = 0
    base = strassen_fill(0,1,base)
    base = strassen_fill(2,3,base)
    base = strassen_fill(4,5,base)
    base = schoolbook_fill(6,base)
    # base should be 25
    rank = base
    U = U[:U_rows]
    V = V[:V_rows]
    W = W[:W_rows]

    # Exact verification using fractions
    try:
        from fractions import Fraction
        def F(x): return Fraction(x)
        # Build tensor check: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == delta_{?}
        # The matmul tensor T[i,j,k] = 1 if ... Actually the multiplication tensor:
        # C[i,k] = sum_j A[i,j] B[j,k]. Tensor indices: A (i,j), B (j,k), C (i,k).
        # T[i,j,k] = 1 if (i,k) is the output index; the decomposition should give
        # sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k (since each output
        # C[i,k] receives A[i,j]*B[j,k] with coefficient 1).
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += F(U[i*m+j][r]) * F(V[j*p+k][r]) * F(W[k*n+i][r])
                    if s != 1:
                        ok = False
                        break
                if not ok: break
            if not ok: break
        if not ok:
            # Fallback to schoolbook rank 28 (always valid)
            return _schoolbook()
    except Exception:
        return _schoolbook()

    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n,m,p = DIMENSIONS
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col]=1
                V[j*p+k][col]=1
                W[k*n+i][col]=1
                col+=1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
