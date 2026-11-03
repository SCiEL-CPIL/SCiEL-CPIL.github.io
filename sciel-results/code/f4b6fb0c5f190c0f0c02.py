import random
import time
import numpy as np

random.seed(42)

DIMENSIONS = [4, 5, 6]
n, m, p = DIMENSIONS
rank_full = n * m * p  # 120
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0


def remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)


def schoolbook():
    U = [[0.0] * rank_full for _ in range(n * m)]
    V = [[0.0] * rank_full for _ in range(m * p)]
    W = [[0.0] * rank_full for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def verify(U, V, W, rank=None):
    if rank is None:
        rank = len(U[0])
    # Exact check: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def verify_fast(U, V, W, rank=None):
    """Faster verification using random matrices."""
    if rank is None:
        rank = len(U[0])
    for _ in range(5):
        if remaining() <= 0:
            return False
        A = np.random.randint(-3, 4, (n, m))
        B = np.random.randint(-3, 4, (m, p))
        C_true = A @ B
        C_comp = np.zeros((n, p))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            # u is (n*m,), v is (m*p,), w is (p*n,)
            # reshape u to (n,m), v to (m,p), w to (p,n) -> w.T is (n,p)
            u_mat = u.reshape(n, m)
            v_mat = v.reshape(m, p)
            w_mat = np.array(w).reshape(p, n).T  # (n,p)
            # C += sum_j u_mat[i,j] * v_mat[j,k] * w_mat[i,k]
            # This is elementwise product of (u_mat @ v_mat) and w_mat? No.
            # C[i,k] += sum_j u_mat[i,j] * v_mat[j,k] * w_mat[i,k]
            # Let M = u_mat @ v_mat. Then C[i,k] += M[i,k] * w_mat[i,k]
            M = u_mat @ v_mat
            C_comp += M * w_mat
        if not np.allclose(C_comp, C_true, atol=1e-9):
            return False
    return True


def winograd_2x2x2_terms():
    """Returns 7 (u, v, w) vectors of length 4 for 2x2x2 matmul."""
    # Standard Winograd F^T(2,2,2) rank 7
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)g
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(h+f)
    # C00 = p1 - p4 + p5 - p7
    # C01 = p3 + p4 + p5 + p7
    # C10 = p2 + p4 - p6 + p7
    # C11 = p1 - p2 + p6 - p7
    
    terms = []
    # p1
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # p2
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    # p3
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    # p4
    terms.append(([0,0,0,1], [-1,0,1,0], [-1,1,1,0]))
    # p5
    terms.append(([1,1,0,0], [0,0,1,0], [1,1,0,0]))
    # p6
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,-1,1]))
    # p7
    terms.append(([0,1,0,-1], [0,1,0,1], [-1,1,1,0]))
    return terms


def build_block_decomposition():
    """
    Build a block-structured decomposition using Winograd 2x2x2 blocks.
    Covers i in {0,1,2,3}, k in {0,1,2,3}, j in {0,1,2,3} with 2x2x2 blocks.
    Remaining indices (j=4 or k in {4,5}) handled by schoolbook.
    """
    terms_U = []
    terms_V = []
    terms_W = []
    
    wg_terms = winograd_2x2x2_terms()
    
    covered = set()
    
    for ib in range(2):  # i block: 2*ib, 2*ib+1
        for kb in range(2):  # k block: 2*kb, 2*kb+1
            for jb in range(2):  # j block: 2*jb, 2*jb+1
                for (u, v, w) in wg_terms:
                    ufull = [0.0] * (n * m)
                    vfull = [0.0] * (m * p)
                    wfull = [0.0] * (p * n)
                    
                    for a in range(2):
                        for b in range(2):
                            ui = (2 * ib + a) * m + (2 * jb + b)
                            ufull[ui] = u[a * 2 + b]
                            covered.add((2*ib+a, 2*jb+b, 2*kb+0))  # placeholder
                            covered.add((2*ib+a, 2*jb+b, 2*kb+1))
                    
                    for b in range(2):
                        for c in range(2):
                            vi = (2 * jb + b) * p + (2 * kb + c)
                            vfull[vi] = v[b * 2 + c]
                    
                    for c in range(2):
                        for a in range(2):
                            wi = (2 * kb + c) * n + (2 * ib + a)
                            wfull[wi] = w[c * 2 + a]
                    
                    terms_U.append(ufull)
                    terms_V.append(vfull)
                    terms_W.append(wfull)
    
    # Recalculate covered set properly
    covered = set()
    for ib in range(2):
        for kb in range(2):
            for jb in range(2):
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            covered.add((2*ib+a, 2*jb+b, 2*kb+c))
    
    # Add schoolbook for uncovered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                ufull = [0.0] * (n * m)
                vfull = [0.0] * (m * p)
                wfull = [0.0] * (p * n)
                ufull[i * m + j] = 1.0
                vfull[j * p + k] = 1.0
                wfull[k * n + i] = 1.0
                terms_U.append(ufull)
                terms_V.append(vfull)
                terms_W.append(wfull)
    
    rank = len(terms_U)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for c, (u, v, w) in enumerate(zip(terms_U, terms_V, terms_W)):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}


def try_column_removal(U, V, W, rank, max_removals=12, timeout=None):
    """
    Try to remove columns from the decomposition while maintaining validity.
    Greedy: try removing each column, check if still valid.
    """
    if timeout is None:
        timeout = remaining() - 10
    deadline = time.time() + timeout
    
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    current_rank = rank
    
    for _ in range(max_removals):
        if time.time() > deadline:
            break
        removed = False
        for col in range(current_rank):
            if time.time() > deadline:
                break
            # Try removing column col
            # Check if column is "redundant" by seeing if we can express it
            # as a linear combination of others, or just try removal and verify.
            # For speed, do a quick check: if all entries in column are 0, remove.
            all_zero_u = all(U[r][col] == 0 for r in range(n*m))
            all_zero_v = all(V[r][col] == 0 for r in range(m*p))
            all_zero_w = all(W[r][col] == 0 for r in range(p*n))
            
            if all_zero_u and all_zero_v and all_zero_w:
                # Remove
                for r in range(n*m):
                    U[r].pop(col)
                for r in range(m*p):
                    V[r].pop(col)
                for r in range(p*n):
                    W[r].pop(col)
                current_rank -= 1
                removed = True
                break
            
            # Try actual removal and fast verify
            # Save state
            U_backup = [row[:] for row in U]
            V_backup = [row[:] for row in V]
            W_backup = [row[:] for row in W]
            
            for r in range(n*m):
                U[r].pop(col)
            for r in range(m*p):
                V[r].pop(col)
            for r in range(p*n):
                W[r].pop(col)
            
            if verify_fast(U, V, W, current_rank - 1):
                current_rank -= 1
                removed = True
                break
            else:
                # Restore
                U = U_backup
                V = V_backup
                W = W_backup
        
        if not removed:
            break
    
    return U, V, W, current_rank


def construct():
    start = time.time()
    
    # Start with block decomposition (rank ~108)
    best = build_block_decomposition()
    rank = len(best["U"][0])
    
    # Verify it's correct
    if not verify_fast(best["U"], best["V"], best["W"], rank):
        # Fallback to schoolbook
        best = schoolbook()
        rank = rank_full
    
    # Try column removal to reduce rank further
    if remaining() > 20:
        U, V, W, new_rank = try_column_removal(
            best["U"], best["V"], best["W"], rank,
            max_removals=15,
            timeout=remaining() - 15
        )
        if new_rank < rank:
            best = {"U": U, "V": V, "W": W}
            rank = new_rank
    
    # Final verification
    if not verify_fast(best["U"], best["V"], best["W"], rank):
        best = schoolbook()
    
    return best
