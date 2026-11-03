import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 6]
DEADLINE_SECONDS = 110  # Safety margin before 150s kill

def _build_target_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n, m, p) with ones at (i,j,k)."""
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify(U, V, W, atol=1e-9):
    """Verify that U, V, W form a valid matrix multiplication decomposition."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    if U_arr.shape != (n*m, rank) or V_arr.shape != (m*p, rank) or W_arr.shape != (p*n, rank):
        return False
    
    # Compute the tensor decomposition: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = U_arr[:, r].reshape(n, m)
        v = V_arr[:, r].reshape(m, p)
        w = W_arr[:, r].reshape(p, n)
        # For each (i,j,k): tensor[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        # This is an outer product structure
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    expected = np.ones((n, m, p))
    return np.allclose(tensor, expected, atol=atol)

def _verify_fast(U, V, W, atol=1e-9):
    """Faster verification using einsum."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    if U_arr.shape != (n*m, rank) or V_arr.shape != (m*p, rank) or W_arr.shape != (p*n, rank):
        return False
    
    # Reshape: U -> (rank, n, m), V -> (rank, m, p), W -> (rank, p, n)
    U_r = U_arr.T.reshape(rank, n, m)
    V_r = V_arr.T.reshape(rank, m, p)
    W_r = W_arr.T.reshape(rank, p, n)
    
    # tensor[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
    # This is a 3-way contraction
    # Use einsum: 'rin,rjm,rkm->ijk' but with the specific index structure
    # Actually: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
    # Let's do it step by step
    # First: A[i,j,k] = sum_r U_r[i,j] * V_r[j,k]  -> (n, m, p)
    # Then: tensor[i,j,k] = A[i,j,k] * W_r[k,i]... no, that's not right
    # The W part depends on r too.
    
    # Let's just do the loop but with numpy operations
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = U_arr[:, r].reshape(n, m)
        v = V_arr[:, r].reshape(m, p)
        w = W_arr[:, r].reshape(p, n)
        # u[i,j] * v[j,k] * w[k,i]
        # For fixed i: sum_j u[i,j]*v[j,k] gives a row, then multiply by w[k,i]
        # Actually let's compute u @ v first (n x p), then element-wise multiply with w transposed
        uv = u @ v  # (n, p), uv[i,k] = sum_j u[i,j]*v[j,k]
        # tensor[i,j,k] += u[i,j]*v[j,k]*w[k,i]
        # This isn't just uv[i,k]*w[k,i] because of the j index
        # We need the full 3D product
        # tensor_r[i,j,k] = u[i,j] * v[j,k] * w[k,i]
        # = outer(u[:,j], v[j,k]) * w[k,i]... 
        # Let's use: for each j, u[:,j] (n,) * v[j,:] (p,) = (n,p) matrix, then * w[:,j]... no
        # w is (p,n), so w[k,i] for fixed k is a row of w
        # Let's just do:
        # temp[i,j,k] = u[i,j] * v[j,k] * w[k,i]
        # = u[i,j] * (v[j,:] * w[:,i]) ... no
        # v[j,k]*w[k,i] = (v @ w.T)[j,i]
        # So temp[i,j,k] = u[i,j] * (v @ w.T)[j,i]... but that's not right either because k is free
        # Actually: sum_k v[j,k]*w[k,i] = (v @ w.T)[j,i]
        # But we don't want to sum over k, we want the product for each k
        # So temp[i,j,k] = u[i,j] * v[j,k] * w[k,i]
        # Let's compute this as: for each (i,j), u[i,j] * v[j,:] * w[:,i]... 
        # v[j,:] is (p,), w[:,i] is (p,), so v[j,:] * w[:,i] is (p,)
        # So temp[i,j,:] = u[i,j] * v[j,:] * w[:,i]
        # We can vectorize: 
        # For fixed j: u[:,j] (n,) * v[j,:] (p,) * w[:,j]... no, w is indexed by (k,i)
        # Let's just use broadcasting:
        # u is (n,m), v is (m,p), w is (p,n)
        # u[:,j,None] * v[j,:,None]... this is getting complicated
        # Let's just do the triple loop but with numpy for the inner part
        for i in range(n):
            for j in range(m):
                tensor[i, j, :] += u[i, j] * v[j, :] * w[:, i]
    
    expected = np.ones((n, m, p))
    return np.allclose(tensor, expected, atol=atol)

def _verify_einsum(U, V, W, atol=1e-9):
    """Verification using einsum for speed."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    if U_arr.shape != (n*m, rank) or V_arr.shape != (m*p, rank) or W_arr.shape != (p*n, rank):
        return False
    
    # U_arr: (n*m, rank), reshape to (rank, n, m)
    # V_arr: (m*p, rank), reshape to (rank, m, p)
    # W_arr: (p*n, rank), reshape to (rank, p, n)
    U_r = U_arr.T.reshape(rank, n, m)  # (rank, n, m)
    V_r = V_arr.T.reshape(rank, m, p)  # (rank, m, p)
    W_r = W_arr.T.reshape(rank, p, n)  # (rank, p, n)
    
    # tensor[i,j,k] = sum_r U_r[r,i,j] * V_r[r,j,k] * W_r[r,k,i]
    # Using einsum: 'ri j, rj k, rk i -> i j k'
    # But numpy einsum doesn't allow repeated indices in input like that easily
    # Let's do: 
    # Step 1: A[r,i,j,k] = U_r[r,i,j] * V_r[r,j,k]  -> (rank, n, m, p)
    # Step 2: tensor[i,j,k] = sum_r A[r,i,j,k] * W_r[r,k,i]
    
    # A[r,i,j,k] = U_r[r,i,j] * V_r[r,j,k]
    # U_r is (rank, n, m), V_r is (rank, m, p)
    # A = U_r[:, :, :, None] * V_r[:, None, :, :]  -> (rank, n, m, p)
    A = U_r[:, :, :, None] * V_r[:, None, :, :]  # (rank, n, m, p)
    
    # tensor[i,j,k] = sum_r A[r,i,j,k] * W_r[r,k,i]
    # W_r is (rank, p, n), so W_r[r,k,i]
    # We need to sum over r: A[r,i,j,k] * W_r[r,k,i]
    # Let's transpose W_r to (rank, n, p) so W_r_trans[r,i,k] = W_r[r,k,i]
    W_r_trans = W_r.transpose(0, 2, 1)  # (rank, n, p)
    
    # tensor[i,j,k] = sum_r A[r,i,j,k] * W_r_trans[r,i,k]
    # = einsum('rijk,rik->ijk', A, W_r_trans)
    tensor = np.einsum('rijk,rik->ijk', A, W_r_trans, optimize=True)
    
    expected = np.ones((n, m, p))
    return np.allclose(tensor, expected, atol=atol)

def _get_schoolbook():
    n, m, p = DIMENSIONS
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

def _strassen_2x2():
    """
    Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen's 7 products for 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M7
    
    # U[i*2+j, c]: A[i,j] coefficient for product c
    # V[j*2+k, c]: B[j,k] coefficient for product c
    # W[k*2+i, c]: C[k,i] coefficient for product c... wait, W[k*n+i] means output index
    
    # Let me use the standard convention:
    # Product r: U_r[i,j] * V_r[j,k] * W_r[k,i]
    # Sum over r gives C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So W_r[k,i] should be such that sum_r U_r[i,j]*V_r[j,k]*W_r[k,i] = C[i,k]
    # This means W_r[k,i] is the coefficient for output C[i,k] in product r
    
    # Let me redefine: W_r[i,k] would be more natural, but the convention is W[k*n+i]
    # So W[k*n+i, r] is the coefficient for C[i,k] in product r
    
    # Product 1: (a00+a11)(b00+b11)
    # A part: a00=1, a11=1, others 0
    # B part: b00=1, b11=1, others 0
    # C part: contributes to c00 (+1), c01 (-1), c10 (-1), c11 (+1)
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[0*2+1, 0] = -1; W[1*2+0, 0] = -1; W[1*2+1, 0] = 1
    
    # Product 2: (a10+a11)b00
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+0, 1] = 0; W[0*2+1, 1] = 0; W[1*2+0, 1] = 1; W[1*2+1, 1] = 0
    
    # Product 3: a00(b10-b11)
    U[0*2+0, 2] = 1
    V[1*2+0, 2] = 1; V[1*2+1, 2] = -1
    W[0*2+0, 2] = 0; W[0*2+1, 2] = 1; W[1*2+0, 2] = 0; W[1*2+1, 2] = 1
    
    # Product 4: a11(b01-b00)
    U[1*2+1, 3] = 1
    V[0*2+1, 3] = 1; V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1; W[0*2+1, 3] = 0; W[1*2+0, 3] = 1; W[1*2+1, 3] = 0
    
    # Product 5: (a00+a01)b11
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1; W[0*2+1, 4] = 0; W[1*2+0, 4] = 0; W[1*2+1, 4] = -1
    
    # Product 6: (a10-a00)(b00+b10)
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[1*2+0, 5] = 1
    W[0*2+0, 5] = 0; W[0*2+1, 5] = 0; W[1*2+0, 5] = 0; W[1*2+1, 5] = 0
    
    # Product 7: (a01-a11)(b01+b11)
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[0*2+1, 6] = 1; V[1*2+1, 6] = 1
    W[0*2+0, 6] = 0; W[0*2+1, 6] = 0; W[1*2+0, 6] = 0; W[1*2+1, 6] = 0
    
    return U, V, W

def _embed_block(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, i_off, j_off, k_off, n, m, p):
    """Embed a sub-decomposition into the full problem."""
    rank = U_sub.shape[1]
    U_full = np.zeros((n * m, rank), dtype=np.float64)
    V_full = np.zeros((m * p, rank), dtype=np.float64)
    W_full = np.zeros((p * n, rank), dtype=np.float64)
    
    for i_s in range(n_sub):
        for j_s in range(m_sub):
            i_f = i_off + i_s
            j_f = j_off + j_s
            U_full[i_f * m + j_f, :] = U_sub[i_s * m_sub + j_s, :]
    
    for j_s in range(m_sub):
        for k_s in range(p_sub):
            j_f = j_off + j_s
            k_f = k_off + k_s
            V_full[j_f * p + k_f, :] = V_sub[j_s * p_sub + k_s, :]
    
    for k_s in range(p_sub):
        for i_s in range(n_sub):
            k_f = k_off + k_s
            i_f = i_off + i_s
            W_full[k_f * n + i_f, :] = W_sub[k_s * n_sub + i_s, :]
    
    return U_full, V_full, W_full

def _block_decomposition(n, m, p, n_splits, m_splits, p_splits, use_strassen=True):
    """
    Block decomposition using splits.
    n_splits: list of sizes summing to n
    m_splits: list of sizes summing to m
    p_splits: list of sizes summing to p
    """
    U_total = None
    V_total = None
    W_total = None
    
    i_off = 0
    for i_size in n_splits:
        j_off = 0
        for j_size in m_splits:
            k_off = 0
            for k_size in p_splits:
                if i_size == 0 or j_size == 0 or k_size == 0:
                    k_off += k_size
                    continue
                
                # Get decomposition for this block
                if use_strassen and i_size == 2 and j_size == 2 and k_size == 2:
                    U_sub, V_sub, W_sub = _strassen_2x2()
                else:
                    # Schoolbook
                    r = i_size * j_size * k_size
                    U_sub = np.zeros((i_size * j_size, r), dtype=np.float64)
                    V_sub = np.zeros((j_size * k_size, r), dtype=np.float64)
                    W_sub = np.zeros((k_size * i_size, r), dtype=np.float64)
                    col = 0
                    for ii in range(i_size):
                        for jj in range(j_size):
                            for kk in range(k_size):
                                U_sub[ii * j_size + jj, col] = 1
                                V_sub[jj * k_size + kk, col] = 1
                                W_sub[kk * i_size + ii, col] = 1
                                col += 1
                
                U_block, V_block, W_block = _embed_block(
                    U_sub, V_sub, W_sub, i_size, j_size, k_size,
                    i_off, j_off, k_off, n, m, p
                )
                
                if U_total is None:
                    U_total = U_block
                    V_total = V_block
                    W_total = W_block
                else:
                    U_total = np.hstack([U_total, U_block])
                    V_total = np.hstack([V_total, V_block])
                    W_total = np.hstack([W_total, W_block])
                
                k_off += k_size
            j_off += j_size
        i_off += i_size
    
    return U_total, V_total, W_total

def _try_als_reduction(U, V, W, target_rank, max_iter=50, tol=1e-8):
    """
    Try to reduce the rank using ALS-like approach.
    Given a valid decomposition of rank R, try to find a decomposition of rank target_rank < R.
    This is done by projecting onto a lower-rank subspace.
    """
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    current_rank = U_arr.shape[1]
    if target_rank >= current_rank:
        return None
    
    # Initialize with random projection
    np.random.seed(42)
    # Create random mixing matrices to reduce rank
    # U: (n*m, R) -> (n*m, r)
    # V: (m*p, R) -> (m*p, r)
    # W: (p*n, R) -> (p*n, r)
    # But we need to preserve the tensor structure
    
    # Alternative: use SVD of the flattening
    # Flatten U to (n*m*rank, 1) and find structure
    # This is complex; let's try a simpler approach
    
    # Try to find linear combinations of columns that still span the tensor
    # Use gradient descent on the coefficients
    
    # For now, return None as this is complex
    return None

def construct():
    construct._start_time = time.time()
    
    # Start with schoolbook (rank 90)
    result = _get_schoolbook()
    best_rank = 90
    best_result = result
    
    n, m, p = DIMENSIONS
    
    # Try various block decompositions
    # Split strategies for n=3, m=5, p=6
    
    # Strategy 1: n=3 as [2,1], m=5 as [2,2,1], p=6 as [2,2,2]
    # This gives 2*3*3 = 18 blocks
    # 2x2x2 blocks: 1*2*3 = 6 blocks, rank 7 each = 42
    # 2x2x1 blocks: 1*2*0 = 0 (no p=1 blocks)
    # Wait, p_splits = [2,2,2], so all k_size = 2
    # n_splits = [2,1], m_splits = [2,2,1]
    # Blocks:
    # i_size=2, j_size=2, k_size=2: Strassen rank 7, count = 1*2*3 = 6? No
    # For each combination of (i_size, j_size, k_size):
    # i_size=2: j_size=2 (x2), j_size=1 (x1); k_size=2 (x3)
    # i_size=1: j_size=2 (x2), j_size=1 (x1); k_size=2 (x3)
    
    splits_list = [
        ([2,1], [2,2,1], [2,2,2]),
        ([2,1], [3,2], [3,3]),
        ([2,1], [2,3], [2,4]),
        ([3], [2,3], [2,4]),
        ([2,1], [2,3], [3,3]),
        ([3], [2,2,1], [2,2,2]),
    ]
    
    for n_sp, m_sp, p_sp in splits_list:
        if time.time() - construct._start_time > DEADLINE_SECONDS:
            break
        try:
            U_b, V_b, W_b = _block_decomposition(n, m, p, n_sp, m_sp, p_sp, use_strassen=True)
            rank_b = U_b.shape[1]
            if rank_b < best_rank:
                if _verify_einsum(U_b, V_b, W_b):
                    best_rank = rank_b
                    best_result = {"U": U_b.tolist(), "V": V_b.tolist(), "W": W_b.tolist()}
        except Exception:
            continue
    
    # Try more splits
    more_splits = [
        ([2,1], [1,2,2], [2,2,2]),
        ([2,1], [1,1,3], [2,2,2]),
        ([2,1], [2,1,2], [2,2,2]),
        ([2,1], [3,1,1], [2,2,2]),
        ([1,2], [2,2,1], [2,2,2]),
        ([1,2], [3,2], [3,3]),
    ]
    
    for n_sp, m_sp, p_sp in more_splits:
        if time.time() - construct._start_time > DEADLINE_SECONDS:
            break
        try:
            U_b, V_b, W_b = _block_decomposition(n, m, p, n_sp, m_sp, p_sp, use_strassen=True)
            rank_b = U_b.shape[1]
            if rank_b < best_rank:
                if _verify_einsum(U_b, V_b, W_b):
                    best_rank = rank_b
                    best_result = {"U": U_b.tolist(), "V": V_b.tolist(), "W": W_b.tolist()}
        except Exception:
            continue
    
    # Try even more diverse splits
    diverse_splits = [
        ([2,1], [2,2,1], [3,3]),
        ([2,1], [2,2,1], [2,4]),
        ([2,1], [2,2,1], [4,2]),
        ([2,1], [2,2,1], [1,5]),
        ([2,1], [2,2,1], [5,1]),
        ([2,1], [3,2], [2,4]),
        ([2,1], [3,2], [4,2]),
        ([2,1], [3,2], [1,5]),
        ([2,1], [3,2], [5,1]),
        ([2,1], [3,2], [2,2,2]),
    ]
    
    for n_sp, m_sp, p_sp in diverse_splits:
        if time.time() - construct._start_time > DEADLINE_SECONDS:
            break
        try:
            U_b, V_b, W_b = _block_decomposition(n, m, p, n_sp, m_sp, p_sp, use_strassen=True)
            rank_b = U_b.shape[1]
            if rank_b < best_rank:
                if _verify_einsum(U_b, V_b, W_b):
                    best_rank = rank_b
                    best_result = {"U": U_b.tolist(), "V": V_b.tolist(), "W": W_b.tolist()}
        except Exception:
            continue
    
    # Try splits that create more 2x2x2 blocks
    # n=3=[2,1], m=5=[2,2,1], p=6=[2,2,2] gives 6 2x2x2 blocks
    # Can we get more? n=3=[2,1] limits to 1 block with n=2
    # What about n=3=[1,2]? Same thing
    # Maximum 2x2x2 blocks: floor(3/2)*floor(5/2)*floor(6/2) = 1*2*3 = 6
    # So max 6 Strassen blocks, rank 6*7 + rest
    
    # Let's also try Winograd 3x3x3 if we can implement it
    # For now, return the best found
    
    return best_result
