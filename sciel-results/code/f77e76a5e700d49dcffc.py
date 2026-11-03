"""Matrix multiplication 3x4x8 with rank-reduced decomposition using searched 3x2x2 sub-blocks."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W


def _schoolbook_1x2x2():
    """Return U, V, W for 1x2x2 schoolbook decomposition (rank 4)."""
    U = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    V = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    W = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    return U, V, W


def _search_3x2x2(rank_target, time_limit=30, seed=42):
    """Search for a low-rank decomposition of the 3x2x2 matrix multiplication tensor.
    
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1,2}, k in {0,1}
    A has 6 entries (i,j), B has 4 entries (j,k), C has 6 entries (i,k).
    We need U (6 x r), V (4 x r), W (6 x r) such that
    sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*3+i, c] = 1 for all (i,j,k)
    """
    start = time.time()
    rng = np.random.RandomState(seed)
    
    def compute_error(U, V, W):
        err = 0.0
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    val = 0.0
                    for c in range(rank_target):
                        val += U[i*2+j, c] * V[j*2+k, c] * W[k*3+i, c]
                    err += (val - 1.0) ** 2
        return err
    
    def verify_exact(U, V, W):
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    val = 0.0
                    for c in range(rank_target):
                        val += U[i*2+j, c] * V[j*2+k, c] * W[k*3+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    best_err = float('inf')
    best_solution = None
    
    # Strategy 1: Try structured Strassen+schoolbook with perturbations
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    
    # Base rank-11 solution
    U_base = np.zeros((6, 11))
    V_base = np.zeros((4, 11))
    W_base = np.zeros((6, 11))
    U_base[0:4, 0:7] = U2
    V_base[0:4, 0:7] = V2
    W_base[0:4, 0:7] = W2
    U_base[4:6, 7:11] = U1
    V_base[0:4, 7:11] = V1
    W_base[4:6, 7:11] = W1
    
    if rank_target == 11:
        if verify_exact(U_base, V_base, W_base):
            return (U_base, V_base, W_base)
    
    # Strategy 2: Random search with gradient-like refinement
    for attempt in range(3000):
        if time.time() - start > time_limit:
            break
        
        # Start from base and perturb
        U = U_base[:, :rank_target].copy() if rank_target <= 11 else np.zeros((6, rank_target))
        V = V_base[:, :rank_target].copy() if rank_target <= 11 else np.zeros((4, rank_target))
        W = W_base[:, :rank_target].copy() if rank_target <= 11 else np.zeros((6, rank_target))
        
        if rank_target > 11:
            U = U_base.copy()
            V = V_base.copy()
            W = W_base.copy()
            # Add random columns
            U[:, 11:rank_target] = rng.randint(-1, 2, size=(6, rank_target-11)).astype(float)
            V[:, 11:rank_target] = rng.randint(-1, 2, size=(4, rank_target-11)).astype(float)
            W[:, 11:rank_target] = rng.randint(-1, 2, size=(6, rank_target-11)).astype(float)
        else:
            # Perturb the base
            U += rng.randint(-1, 2, size=(6, rank_target)).astype(float) * 0.1
            V += rng.randint(-1, 2, size=(4, rank_target)).astype(float) * 0.1
            W += rng.randint(-1, 2, size=(6, rank_target)).astype(float) * 0.1
        
        err = compute_error(U, V, W)
        if err < 1e-8:
            best_solution = (U, V, W)
            break
        if err < best_err:
            best_err = err
            best_solution = (U.copy(), V.copy(), W.copy())
    
    if best_solution is not None and best_err < 1e-8:
        U, V, W = best_solution
        if verify_exact(U, V, W):
            return best_solution
    
    # Strategy 3: Pure random search
    for attempt in range(2000):
        if time.time() - start > time_limit:
            break
        
        U = rng.randint(-2, 3, size=(6, rank_target)).astype(float)
        V = rng.randint(-2, 3, size=(4, rank_target)).astype(float)
        W = rng.randint(-2, 3, size=(6, rank_target)).astype(float)
        
        err = compute_error(U, V, W)
        if err < 1e-8:
            best_solution = (U, V, W)
            break
        if err < best_err:
            best_err = err
            best_solution = (U.copy(), V.copy(), W.copy())
    
    if best_solution is not None and best_err < 1e-8:
        U, V, W = best_solution
        if verify_exact(U, V, W):
            return best_solution
    
    return None


def _try_merge_columns(U, V, W, n, m, p):
    """Try to find and remove redundant columns from the decomposition."""
    rank = len(U[0])
    n_keep = list(range(rank))
    
    # Convert to numpy for faster operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor contribution matrix: for each column c and each (i,j,k),
    # the contribution is U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We want to find columns whose contribution vector is in the span of others.
    
    # For each column c, compute its contribution vector over all (i,j,k)
    # There are n*m*p = 96 (i,j,k) triples
    # contribution[c, i, j, k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    contrib = np.zeros((rank, n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                contrib[:, i, j, k] = U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i]
    
    # Reshape to 2D: (rank, n*m*p)
    contrib_2d = contrib.reshape(rank, n*m*p)
    
    # Try to remove columns whose contribution is in the span of previously kept columns
    # Process in order, keeping a basis
    kept = []
    for c in range(rank):
        if c not in n_keep:
            continue
        if time.time() > 100:
            break
        if not kept:
            kept.append(c)
            continue
        # Check if contrib_2d[c] is in the span of [contrib_2d[k] for k in kept]
        # Use least squares
        A = contrib_2d[kept, :]  # (len(kept), n*m*p)
        b = contrib_2d[c, :]
        # Check if b is in the row space of A
        # b is in row space of A iff ||A^T (A A^T)^+ b|| ≈ ||b||
        # Simpler: solve A^T x = b and check residual
        try:
            # Use SVD to check
            U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
            # b is in row space of A iff b is in column space of A^T
            # A^T has shape (n*m*p, len(kept))
            # Project b onto column space of A^T
            # col space of A^T = row space of A
            # Use: proj = U_svd[:, :rank_A] * (U_svd[:, :rank_A].T @ b)
            rank_A = np.sum(s > 1e-10)
            if rank_A == 0:
                # A is zero, c can be removed only if b is also zero
                if np.allclose(b, 0, atol=1e-10):
                    # Remove c
                    n_keep.remove(c)
                continue
            U_sub = U_svd[:, :rank_A]
            proj = U_sub @ (U_sub.T @ b)
            residual = np.linalg.norm(b - proj)
            norm_b = np.linalg.norm(b)
            if norm_b < 1e-10:
                # b is zero, c can be removed
                n_keep.remove(c)
            elif residual / max(norm_b, 1e-15) < 1e-8:
                # b is in the span, remove c
                n_keep.remove(c)
            else:
                kept.append(c)
        except np.linalg.LinAlgError:
            kept.append(c)
    
    if len(n_keep) == rank:
        return U, V, W, rank
    
    # Build new matrices
    new_rank = len(n_keep)
    final_U = [U[i][c] for i in range(len(U)) for c in [0]] if False else [[U[i][c] for c in n_keep] for i in range(len(U))]
    final_V = [[V[i][c] for c in n_keep] for i in range(len(V))]
    final_W = [[W[i][c] for c in n_keep] for i in range(len(W))]
    
    return final_U, final_V, final_W, new_rank


def _build_decomposition_3x2x2():
    """Build a 3x2x2 decomposition. Try to find lower rank than 11."""
    n, m, p = 3, 2, 2
    
    # Try to find a lower-rank solution first
    for r in [9, 10]:
        result = _search_3x2x2(r, time_limit=5, seed=42)
        if result is not None:
            U, V, W = result
            U_list = [[U[i][c] for c in range(r)] for i in range(6)]
            V_list = [[V[i][c] for c in range(r)] for i in range(4)]
            W_list = [[W[i][c] for c in range(r)] for i in range(6)]
            return U_list, V_list, W_list, r
    
    # Fall back to rank 11
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    
    rank = 7 + 4  # 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for rows 0,1 of A
    for i in range(4):
        for c in range(7):
            U[i][c] += U2[i][c]
            V[i][c] += V2[i][c]
            W[i][c] += W2[i][c]
    
    # Schoolbook for row 2 of A
    for i in range(2):
        for c in range(4):
            U[4 + i][7 + c] += U1[i][c]
    for i in range(4):
        for c in range(4):
            V[i][7 + c] += V1[i][c]
    for i in range(2):
        for c in range(4):
            W[4 + i][7 + c] += W1[i][c]
    
    return U, V, W, rank


def _build_full_decomposition(block_decomp):
    """Build the full 3x4x8 decomposition from a 3x2x2 block decomposition."""
    n, m, p = 3, 4, 8
    U_block, V_block, W_block, r = block_decomp
    
    total_rank = 8 * r  # 8 blocks (2 jp * 4 kp)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for jp in range(2):
        for kp in range(4):
            col_off = (jp * 4 + kp) * r
            j0 = 2 * jp
            k0 = 2 * kp
            
            # Map block indices to full indices
            # Block A: rows 0..5 (i*2+j for i in 0..2, j in 0..1)
            # Full A: i*m+j for i in 0..2, j in 0..3
            for bi in range(6):
                bi_i = bi // 2
                bi_j = bi % 2
                full_i = bi_i
                full_j = j0 + bi_j
                for c in range(r):
                    U[full_i * m + full_j][col_off + c] += U_block[bi][c]
            
            # Block B: rows 0..3 (j*2+k for j in 0..1, k in 0..1)
            # Full B: j*p+k for j in 0..3, k in 0..7
            for bi in range(4):
                bi_j = bi // 2
                bi_k = bi % 2
                full_j = j0 + bi_j
                full_k = k0 + bi_k
                for c in range(r):
                    V[full_j * p + full_k][col_off + c] += V_block[bi][c]
            
            # Block W: rows 0..5 (k*3+i for k in 0..1, i in 0..2)
            # Full W: k*n+i for k in 0..7, i in 0..2
            for bi in range(6):
                bi_k = bi // 3
                bi_i = bi % 3
                full_k = k0 + bi_k
                full_i = bi_i
                for c in range(r):
                    W[full_k * n + full_i][col_off + c] += W_block[bi][c]
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1) > 1e-9:
                    return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, max_iter=50):
    """Iteratively try to reduce rank by finding redundant columns."""
    for iteration in range(max_iter):
        U, V, W, new_rank = _try_merge_columns(U, V, W, n, m, p)
        if new_rank == len(U[0]):
            break
        # Verify after reduction
        if not _verify_decomposition(U, V, W, n, m, p):
            break
    return U, V, W


def construct():
    start = time.time()
    deadline = start + 135  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Build the base 3x2x2 block decomposition
    U_block, V_block, W_block, r_block = _build_decomposition_3x2x2()
    
    # Build full decomposition
    result = _build_full_decomposition((U_block, V_block, W_block, r_block))
    
    # Verify
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback to original construction
        U, V, W = _build_decomposition_fallback()
        result = {"U": U, "V": V, "W": W}
    
    # Try to reduce rank using improved column merging
    if time.time() < deadline - 10:
        result["U"], result["V"], result["W"] = _try_reduce_rank(
            result["U"], result["V"], result["W"], n, m, p
        )
    
    # Final verification
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Ultimate fallback: schoolbook
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
        result = {"U": U, "V": V, "W": W}
    
    return result


def _build_decomposition_fallback():
    """Original fallback: 8 Strassen + 8 schoolbook blocks."""
    n, m, p = 3, 4, 8
    total_rank = 8 * 7 + 8 * 4  # 88
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    
    col_offset = 0
    for jp in range(2):
        for kp in range(4):
            j0 = 2 * jp
            k0 = 2 * kp
            # Strassen for i=0,1
            u_rows = [0 * m + j0, 0 * m + j0 + 1, 1 * m + j0, 1 * m + j0 + 1]
            v_rows = [j0 * p + k0, j0 * p + k0 + 1, (j0+1) * p + k0, (j0+1) * p + k0 + 1]
            w_rows = [k0 * n + 0, (k0+1) * n + 0, k0 * n + 1, (k0+1) * n + 1]
            for i in range(4):
                for c in range(7):
                    U[u_rows[i]][col_offset + c] += U2[i][c]
                    V[v_rows[i]][col_offset + c] += V2[i][c]
                    W[w_rows[i]][col_offset + c] += W2[i][c]
            col_offset += 7
    
    for jp in range(2):
        for kp in range(4):
            j0 = 2 * jp
            k0 = 2 * kp
            # Schoolbook for i=2
            u_rows = [2 * m + j0, 2 * m + j0 + 1]
            v_rows = [j0 * p + k0, (j0+1) * p + k0, j0 * p + k0 + 1, (j0+1) * p + k0 + 1]
            w_rows = [k0 * n + 2, (k0+1) * n + 2]
            for i in range(2):
                for c in range(4):
                    U[u_rows[i]][col_offset + c] += U1[i][c]
                    W[w_rows[i]][col_offset + c] += W1[i][c]
            for i in range(4):
                for c in range(4):
                    V[v_rows[i]][col_offset + c] += V1[i][c]
            col_offset += 4
    
    return U, V, W
