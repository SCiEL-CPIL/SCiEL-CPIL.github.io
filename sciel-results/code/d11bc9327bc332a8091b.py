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
    
    # Build constraint tensor: T[i,j,k] = 1 for all (i,j,k)
    # Constraint: sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*3+i,c] = 1
    
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
    
    # Strategy: Start from a known valid higher-rank solution and try to zero out columns
    # First, build a rank-12 schoolbook solution
    U12 = np.zeros((6, 12))
    V12 = np.zeros((4, 12))
    W12 = np.zeros((6, 12))
    col = 0
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U12[i*2+j, col] = 1.0
                V12[j*2+k, col] = 1.0
                W12[k*3+i, col] = 1.0
                col += 1
    
    # Now try to find a rank-rank_target sub-decomposition
    # Use iterative approach: randomly project and refine
    
    best_err = float('inf')
    best_solution = None
    
    # Try multiple random initializations
    for attempt in range(5000):
        if time.time() - start > time_limit:
            break
        
        # Random initialization: mix of small integers and structured patterns
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
    
    # Try a more structured approach: use Strassen-like patterns
    # For 3x2x2, try combining Strassen blocks
    # Block 1: rows 0-1 of A (2x2x2 Strassen, rank 7)
    # Block 2: row 2 of A (1x2x2 schoolbook, rank 4)
    # Total: rank 11 (not 12)
    # But maybe we can do better by sharing structure
    
    # Try rank 11: 2 Strassen + 1 schoolbook but with shared columns
    if rank_target <= 11:
        # Try to find a rank-11 solution by combining Strassen and schoolbook
        # with some column sharing
        for attempt in range(3000):
            if time.time() - start > time_limit:
                break
            
            # Initialize with Strassen structure for first 7 columns
            U = np.zeros((6, rank_target))
            V = np.zeros((4, rank_target))
            W = np.zeros((6, rank_target))
            
            # Strassen part (columns 0-6)
            U2, V2, W2 = _strassen_2x2x2()
            U[0:4, 0:7] = U2
            V[0:4, 0:7] = V2
            W[0:4, 0:7] = W2
            
            # Schoolbook part for row 2 (columns 7-10)
            U1, V1, W1 = _schoolbook_1x2x2()
            U[4:6, 7:11] = U1
            V[0:4, 7:11] = V1
            W[4:6, 7:11] = W1
            
            # Try to reduce: set some entries to 0 and adjust
            # Or try random perturbation of the last few columns
            for c in range(7, rank_target):
                if c < 11:
                    # Perturb
                    U[4:6, c] += rng.randint(-1, 2)
                    V[:, c] += rng.randint(-1, 2)
                    W[4:6, c] += rng.randint(-1, 2)
            
            # If rank_target < 11, zero out extra columns and try to compensate
            if rank_target < 11:
                U[:, rank_target:] = 0
                V[:, rank_target:] = 0
                W[:, rank_target:] = 0
            
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
    
    # For each pair of columns, check if one can be expressed as a linear combination
    # that doesn't affect the tensor product
    # Simpler: try removing each column and see if the decomposition still works
    # (this only works if a column is exactly zero in the tensor product sense)
    
    # Better: for each column c, check if there exist scalars alpha, beta for another column c'
    # such that the tensor contribution of c equals alpha * tensor contribution of c'
    
    removed = 0
    new_U = [row[:] for row in U]
    new_V = [row[:] for row in V]
    new_W = [row[:] for row in W]
    
    # Try removing columns that are linearly dependent in a specific way
    # For column c, its contribution is: sum over (i,j,k) of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Two columns c1, c2 can be merged if U[:,c1] = a*U[:,c2], V[:,c1]=b*V[:,c2], W[:,c1]=c*W[:,c2]
    # and a*b*c = 1 (for the product to match)
    
    for c1 in range(rank):
        if time.time() > 100:  # safety
            break
        if c1 not in n_keep:
            continue
        for c2 in range(c1 + 1, rank):
            if c2 not in n_keep:
                continue
            # Check if column c1 is proportional to column c2 in all three matrices
            # with product of ratios = 1
            try:
                # Find ratio for U
                ratio_u = None
                for idx in range(len(U)):
                    if new_U[idx][c2] != 0:
                        if new_U[idx][c1] == 0:
                            ratio_u = 0
                            break
                        r = new_U[idx][c1] / new_U[idx][c2]
                        if ratio_u is None:
                            ratio_u = r
                        elif abs(r - ratio_u) > 1e-9:
                            ratio_u = None
                            break
                    else:
                        if new_U[idx][c1] != 0:
                            ratio_u = None
                            break
                
                if ratio_u is None:
                    continue
                
                # Find ratio for V
                ratio_v = None
                for idx in range(len(V)):
                    if new_V[idx][c2] != 0:
                        if new_V[idx][c1] == 0:
                            ratio_v = 0
                            break
                        r = new_V[idx][c1] / new_V[idx][c2]
                        if ratio_v is None:
                            ratio_v = r
                        elif abs(r - ratio_v) > 1e-9:
                            ratio_v = None
                            break
                    else:
                        if new_V[idx][c1] != 0:
                            ratio_v = None
                            break
                
                if ratio_v is None:
                    continue
                
                # Find ratio for W
                ratio_w = None
                for idx in range(len(W)):
                    if new_W[idx][c2] != 0:
                        if new_W[idx][c1] == 0:
                            ratio_w = 0
                            break
                        r = new_W[idx][c1] / new_W[idx][c2]
                        if ratio_w is None:
                            ratio_w = r
                        elif abs(r - ratio_w) > 1e-9:
                            ratio_w = None
                            break
                    else:
                        if new_W[idx][c1] != 0:
                            ratio_w = None
                            break
                
                if ratio_w is None:
                    continue
                
                # Check product: ratio_u * ratio_v * ratio_w should equal 1
                if abs(ratio_u * ratio_v * ratio_w - 1.0) > 1e-9:
                    continue
                
                # Columns are mergeable! Remove c1
                n_keep.remove(c1)
                removed += 1
                break  # Restart c2 loop after removal
                
            except (ZeroDivisionError, IndexError):
                continue
    
    if removed == 0:
        return U, V, W, rank
    
    # Build new matrices
    new_rank = len(n_keep)
    final_U = [[new_U[i][c] for c in n_keep] for i in range(len(U))]
    final_V = [[new_V[i][c] for c in n_keep] for i in range(len(V))]
    final_W = [[new_W[i][c] for c in n_keep] for i in range(len(W))]
    
    return final_U, final_V, final_W, new_rank


def _build_decomposition_3x2x2():
    """Build a 3x2x2 decomposition using Strassen + schoolbook (rank 11)."""
    n, m, p = 3, 2, 2
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
    deadline = start + 140  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Build the base 3x2x2 block decomposition (rank 11)
    U_block, V_block, W_block, r_block = _build_decomposition_3x2x2()
    
    # Build full decomposition
    result = _build_full_decomposition((U_block, V_block, W_block, r_block))
    
    # Verify
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback to original construction
        U, V, W = _build_decomposition_fallback()
        result = {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    if time.time() < deadline - 20:
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
