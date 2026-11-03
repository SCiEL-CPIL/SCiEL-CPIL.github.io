"""Matrix multiplication 2x4x7 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 150s kill

    # Block decomposition: m=4 split into two 2-blocks, p=7 split into 2+2+2+1
    # Each 2x2x2 uses Strassen (rank 7), each 2x2x1 uses schoolbook (rank 4)
    # Total: 2 * (3*7 + 4) = 50

    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    m_blocks = [(0, 2), (2, 4)]
    k_blocks = [
        (0, 2, 'strassen'),
        (2, 4, 'strassen'),
        (4, 6, 'strassen'),
        (6, 7, 'school')
    ]
    
    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    
    for j_start, j_end in m_blocks:
        for k_start, k_end, k_type in k_blocks:
            if k_type == 'strassen':
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
            else:
                U_sub, V_sub, W_sub = U_sc, V_sc, W_sc
                r_sub = 4
            
            p_sub = k_end - k_start
            
            for i in range(n):
                for j_sub in range(j_end - j_start):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            
            for j_sub in range(j_end - j_start):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            
            col_offset += r_sub
    
    # Verify the rank-50 solution
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        r = n * m * p
        U_fb = [[0] * r for _ in range(n * m)]
        V_fb = [[0] * r for _ in range(m * p)]
        W_fb = [[0] * r for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    best_U, best_V, best_W = U, V, W
    best_rank = total_rank
    
    # Try to reduce rank by finding linear dependencies
    if time.time() < deadline - 15:
        U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p, deadline - 15)
        if U_red is not None and U_red.shape[1] < best_rank:
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = U_red.shape[1]
    
    # Try a direct shifted-Strassen construction
    if time.time() < deadline - 20:
        U_shift, V_shift, W_shift = _try_shifted_strassen(n, m, p, deadline - 20)
        if U_shift is not None and U_shift.shape[1] < best_rank:
            best_U, best_V, best_W = U_shift, V_shift, W_shift
            best_rank = U_shift.shape[1]
    
    # Try a more aggressive reduction on the best result found
    if best_rank > 45 and time.time() < deadline - 10:
        U_red2, V_red2, W_red2 = _try_reduce_rank_aggressive(best_U, best_V, best_W, n, m, p, deadline - 10)
        if U_red2 is not None and U_red2.shape[1] < best_rank:
            best_U, best_V, best_W = U_red2, V_red2, W_red2
            best_rank = U_red2.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to find a lower-rank decomposition by identifying linear dependencies."""
    r = U.shape[1]
    
    # Build the constraint rows
    rows = []
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    rows.append((i, j, jp, k))
    
    n_constraints = len(rows)
    
    # Try eliminating columns iteratively, starting from most likely redundant
    # Schoolbook columns are most likely redundant
    order = list(range(r - 1, r - 9, -1)) + list(range(r - 9, -1, -1))
    
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    r_cur = r
    removed = set()
    
    for c0 in order:
        if time.time() > deadline - 5:
            break
        if c0 in removed:
            continue
        
        # Build the system: find alpha such that
        # sum_{c not in removed, c != c0} alpha_c * U_c * V_c * W_c = U_c0 * V_c0 * W_c0
        active_cols = [c for c in range(r_cur) if c not in removed and c != c0]
        if len(active_cols) == 0:
            continue
        
        A = np.zeros((n_constraints, len(active_cols)))
        b = np.zeros(n_constraints)
        
        for col_idx, c in enumerate(active_cols):
            for idx2, (i, j, jp, k) in enumerate(rows):
                A[idx2, col_idx] = U_cur[i * m + j, c] * V_cur[jp * p + k, c] * W_cur[k * n + i, c]
        
        for idx2, (i, j, jp, k) in enumerate(rows):
            b[idx2] = U_cur[i * m + j, c0] * V_cur[jp * p + k, c0] * W_cur[k * n + i, c0]
        
        # Solve least squares
        try:
            alpha, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        # Check if the solution is accurate
        if len(residuals) > 0:
            if residuals[0] > 1e-8:
                continue
        
        # Verify the solution by constructing new matrices
        U_new = np.zeros((n * m, r_cur - 1))
        V_new = np.zeros((m * p, r_cur - 1))
        W_new = np.zeros((p * n, r_cur - 1))
        
        col_idx = 0
        for c in range(r_cur):
            if c in removed or c == c0:
                continue
            U_new[:, col_idx] = U_cur[:, c] * alpha[col_idx]
            V_new[:, col_idx] = V_cur[:, c]
            W_new[:, col_idx] = W_cur[:, c]
            col_idx += 1
        
        # Verify
        if _verify(U_new, V_new, W_new, n, m, p):
            U_cur = U_new
            V_cur = V_new
            W_cur = W_new
            r_cur -= 1
            removed.add(c0)
            continue
        
        # Try with different scaling: put alpha in V instead
        U_new = np.zeros((n * m, r_cur - 1))
        V_new = np.zeros((m * p, r_cur - 1))
        W_new = np.zeros((p * n, r_cur - 1))
        
        col_idx = 0
        for c in range(r_cur):
            if c in removed or c == c0:
                continue
            U_new[:, col_idx] = U_cur[:, c]
            V_new[:, col_idx] = V_cur[:, c] * alpha[col_idx]
            W_new[:, col_idx] = W_cur[:, c]
            col_idx += 1
        
        if _verify(U_new, V_new, W_new, n, m, p):
            U_cur = U_new
            V_cur = V_new
            W_cur = W_new
            r_cur -= 1
            removed.add(c0)
            continue
    
    if r_cur < r:
        return U_cur[:, :r_cur], V_cur[:, :r_cur], W_cur[:, :r_cur]
    
    return None, None, None

def _try_reduce_rank_aggressive(U, V, W, n, m, p, deadline):
    """More aggressive rank reduction: try all possible column removals multiple times."""
    r = U.shape[1]
    
    rows = []
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    rows.append((i, j, jp, k))
    
    n_constraints = len(rows)
    
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    r_cur = r
    removed = set()
    
    # Try multiple passes
    for pass_num in range(3):
        if time.time() > deadline - 5:
            break
        
        # Randomize the order for each pass
        order = list(range(r_cur))
        np.random.shuffle(order)
        # But prioritize schoolbook columns (last 8)
        school_cols = [c for c in order if c >= r - 8]
        strassen_cols = [c for c in order if c < r - 8]
        order = school_cols + strassen_cols
        
        progress = True
        while progress and time.time() < deadline - 5:
            progress = False
            for c0 in order:
                if time.time() > deadline - 5:
                    break
                if c0 in removed or c0 >= r_cur:
                    continue
                
                active_cols = [c for c in range(r_cur) if c not in removed and c != c0]
                if len(active_cols) == 0:
                    continue
                
                A = np.zeros((n_constraints, len(active_cols)))
                b = np.zeros(n_constraints)
                
                for col_idx, c in enumerate(active_cols):
                    for idx2, (i, j, jp, k) in enumerate(rows):
                        A[idx2, col_idx] = U_cur[i * m + j, c] * V_cur[jp * p + k, c] * W_cur[k * n + i, c]
                
                for idx2, (i, j, jp, k) in enumerate(rows):
                    b[idx2] = U_cur[i * m + j, c0] * V_cur[jp * p + k, c0] * W_cur[k * n + i, c0]
                
                try:
                    alpha, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    continue
                
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    continue
                
                # Try absorbing alpha into U
                U_new = np.zeros((n * m, r_cur - 1))
                V_new = np.zeros((m * p, r_cur - 1))
                W_new = np.zeros((p * n, r_cur - 1))
                
                col_idx = 0
                for c in range(r_cur):
                    if c in removed or c == c0:
                        continue
                    U_new[:, col_idx] = U_cur[:, c] * alpha[col_idx]
                    V_new[:, col_idx] = V_cur[:, c]
                    W_new[:, col_idx] = W_cur[:, c]
                    col_idx += 1
                
                if _verify(U_new, V_new, W_new, n, m, p):
                    U_cur = U_new
                    V_cur = V_new
                    W_cur = W_new
                    r_cur -= 1
                    removed.add(c0)
                    progress = True
                    break
                
                # Try with alpha in V
                U_new = np.zeros((n * m, r_cur - 1))
                V_new = np.zeros((m * p, r_cur - 1))
                W_new = np.zeros((p * n, r_cur - 1))
                
                col_idx = 0
                for c in range(r_cur):
                    if c in removed or c == c0:
                        continue
                    U_new[:, col_idx] = U_cur[:, c]
                    V_new[:, col_idx] = V_cur[:, c] * alpha[col_idx]
                    W_new[:, col_idx] = W_cur[:, c]
                    col_idx += 1
                
                if _verify(U_new, V_new, W_new, n, m, p):
                    U_cur = U_new
                    V_cur = V_new
                    W_cur = W_new
                    r_cur -= 1
                    removed.add(c0)
                    progress = True
                    break
                
                # Try with alpha in W
                U_new = np.zeros((n * m, r_cur - 1))
                V_new = np.zeros((m * p, r_cur - 1))
                W_new = np.zeros((p * n, r_cur - 1))
                
                col_idx = 0
                for c in range(r_cur):
                    if c in removed or c == c0:
                        continue
                    U_new[:, col_idx] = U_cur[:, c]
                    V_new[:, col_idx] = V_cur[:, c]
                    W_new[:, col_idx] = W_cur[:, c] * alpha[col_idx]
                    col_idx += 1
                
                if _verify(U_new, V_new, W_new, n, m, p):
                    U_cur = U_new
                    V_cur = V_new
                    W_cur = W_new
                    r_cur -= 1
                    removed.add(c0)
                    progress = True
                    break
    
    if r_cur < r:
        return U_cur[:, :r_cur], V_cur[:, :r_cur], W_cur[:, :r_cur]
    
    return None, None, None

def _verify(U, V, W, n, m, p):
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _get_strassen_222():
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_school_221():
    n, m, p = 2, 2, 1
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
        
    return U, V, W

def _try_shifted_strassen(n, m, p, deadline):
    """Try a shifted-Strassen decomposition exploiting overlapping 2x2 blocks in B."""
    U_sub, V_sub, W_sub = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    
    n_sub, m_sub, p_sub = 2, 2, 7
    r_total = 25
    U_full = np.zeros((n_sub * m_sub, r_total))
    V_full = np.zeros((m_sub * p_sub, r_total))
    W_full = np.zeros((p_sub * n_sub, r_total))
    
    col_offset = 0
    # Strassen for (0,1), (2,3), (4,5)
    for k_start in [0, 2, 4]:
        for i in range(n_sub):
            for j in range(m_sub):
                U_full[i * m_sub + j, col_offset:col_offset+7] = U_sub[i * m_sub + j, :]
        for j in range(m_sub):
            for k_sub in range(2):
                V_full[j * p_sub + k_start + k_sub, col_offset:col_offset+7] = V_sub[j * 2 + k_sub, :]
        for k_sub in range(2):
            for i in range(n_sub):
                W_full[(k_start + k_sub) * n_sub + i, col_offset:col_offset+7] = W_sub[k_sub * n_sub + i, :]
        col_offset += 7
    
    # Schoolbook for column 6
    for c in range(4):
        i = c // m_sub
        j = c % m_sub
        U_full[i * m_sub + j, col_offset + c] = 1.0
        V_full[j * p_sub + 6, col_offset + c] = 1.0
        W_full[6 * n_sub + i, col_offset + c] = 1.0
    col_offset += 4
    
    if not _verify(U_full, V_full, W_full, n_sub, m_sub, p_sub):
        return None, None, None
    
    # Now try to reduce this 2x2x7 from rank 25
    if time.time() > deadline - 10:
        U_sub_red, V_sub_red, W_sub_red = U_full, V_full, W_full
        r_sub = 25
    else:
        rows = []
        for i in range(n_sub):
            for j in range(m_sub):
                for jp in range(m_sub):
                    for k in range(p_sub):
                        rows.append((i, j, jp, k))
        
        n_constraints = len(rows)
        r_cur = 25
        U_cur = U_full.copy()
        V_cur = V_full.copy()
        W_cur = W_full.copy()
        removed = set()
        
        # Try schoolbook columns first (21-24), then Strassen columns
        order = list(range(24, 20, -1)) + list(range(20, -1, -1))
        
        for c0 in order:
            if time.time() > deadline - 5:
                break
            if c0 in removed:
                continue
            
            active_cols = [c for c in range(r_cur) if c not in removed and c != c0]
            if len(active_cols) == 0:
                continue
            
            A = np.zeros((n_constraints, len(active_cols)))
            b = np.zeros(n_constraints)
            
            for col_idx, c in enumerate(active_cols):
                for idx2, (i, j, jp, k) in enumerate(rows):
                    A[idx2, col_idx] = U_cur[i * m_sub + j, c] * V_cur[jp * p_sub + k, c] * W_cur[k * n_sub + i, c]
            
            for idx2, (i, j, jp, k) in enumerate(rows):
                b[idx2] = U_cur[i * m_sub + j, c0] * V_cur[jp * p_sub + k, c0] * W_cur[k * n_sub + i, c0]
            
            try:
                alpha, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            
            # Try absorbing alpha into U
            U_new = np.zeros((n_sub * m_sub, r_cur - 1))
            V_new = np.zeros((m_sub * p_sub, r_cur - 1))
            W_new = np.zeros((p_sub * n_sub, r_cur - 1))
            
            col_idx = 0
            for c in range(r_cur):
                if c in removed or c == c0:
                    continue
                U_new[:, col_idx] = U_cur[:, c] * alpha[col_idx]
                V_new[:, col_idx] = V_cur[:, c]
                W_new[:, col_idx] = W_cur[:, c]
                col_idx += 1
            
            if _verify(U_new, V_new, W_new, n_sub, m_sub, p_sub):
                U_cur = U_new
                V_cur = V_new
                W_cur = W_new
                r_cur -= 1
                removed.add(c0)
                continue
            
            # Try with alpha in V
            U_new = np.zeros((n_sub * m_sub, r_cur - 1))
            V_new = np.zeros((m_sub * p_sub, r_cur - 1))
            W_new = np.zeros((p_sub * n_sub, r_cur - 1))
            
            col_idx = 0
            for c in range(r_cur):
                if c in removed or c == c0:
                    continue
                U_new[:, col_idx] = U_cur[:, c]
                V_new[:, col_idx] = V_cur[:, c] * alpha[col_idx]
                W_new[:, col_idx] = W_cur[:, c]
                col_idx += 1
            
            if _verify(U_new, V_new, W_new, n_sub, m_sub, p_sub):
                U_cur = U_new
                V_cur = V_new
                W_cur = W_new
                r_cur -= 1
                removed.add(c0)
                continue
            
            # Try with alpha in W
            U_new = np.zeros((n_sub * m_sub, r_cur - 1))
            V_new = np.zeros((m_sub * p_sub, r_cur - 1))
            W_new = np.zeros((p_sub * n_sub, r_cur - 1))
            
            col_idx = 0
            for c in range(r_cur):
                if c in removed or c == c0:
                    continue
                U_new[:, col_idx] = U_cur[:, c]
                V_new[:, col_idx] = V_cur[:, c]
                W_new[:, col_idx] = W_cur[:, c] * alpha[col_idx]
                col_idx += 1
            
            if _verify(U_new, V_new, W_new, n_sub, m_sub, p_sub):
                U_cur = U_new
                V_cur = V_new
                W_cur = W_new
                r_cur -= 1
                removed.add(c0)
                continue
        
        if r_cur < 25:
            U_sub_red = U_cur[:, :r_cur]
            V_sub_red = V_cur[:, :r_cur]
            W_sub_red = W_cur[:, :r_cur]
        else:
            U_sub_red = U_full
            V_sub_red = V_full
            W_sub_red = W_full
        r_sub = U_sub_red.shape[1]
    
    # Now build the full 2x4x7 from the reduced 2x2x7
    total_rank = 2 * r_sub
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for j_start, j_end in [(0, 2), (2, 4)]:
        col_offset = (j_start // 2) * r_sub
        for i in range(n):
            for j_sub in range(j_end - j_start):
                j = j_start + j_sub
                U[i * m + j, col_offset:col_offset + r_sub] = U_sub_red[i * 2 + j_sub, :]
        
        for j_sub in range(j_end - j_start):
            j = j_start + j_sub
            for k in range(p):
                V[j * p + k, col_offset:col_offset + r_sub] = V_sub_red[j_sub * p + k, :]
        
        for k in range(p):
            for i in range(n):
                W[k * n + i, col_offset:col_offset + r_sub] = W_sub_red[k * n + i, :]
    
    if _verify(U, V, W, n, m, p):
        return U, V, W
    
    return None, None, None

# EVOLVE-BLOCK-END
