import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

def _verify(U, V, W):
    """
    Verify if U, V, W exactly decompose the matmul tensor.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    """
    r = U.shape[1]
    if r == 0:
        return False
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    try:
        T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    except Exception:
        return False
    
    if not np.allclose(T_reconstructed, 1.0, atol=1e-9, rtol=1e-9):
        return False
    
    # Check half-integer constraint
    U2 = 2 * U
    V2 = 2 * V
    W2 = 2 * W
    if not np.allclose(U2, np.round(U2), atol=1e-9): return False
    if not np.allclose(V2, np.round(V2), atol=1e-9): return False
    if not np.allclose(W2, np.round(W2), atol=1e-9): return False
    
    return True

def _try_different_blockings(deadline=None):
    """
    Try different blocking strategies to find the minimum rank.
    """
    best_rank = float('inf')
    best_result = None
    
    def _time_exceeded():
        return deadline is not None and time.time() > deadline
    
    def _build_from_spec(spec):
        """Build decomposition from spec: list of (type, params) where type is 'strassen' or 'direct'"""
        U_cols = []
        V_cols = []
        W_cols = []
        
        for item in spec:
            if item[0] == 'strassen':
                ib, jb, kb = item[1], item[2], item[3]
                U_terms, V_terms, W_terms = _strassen_2x2_block(ib, jb, kb)
                for u_t, v_t, w_t in zip(U_terms, V_terms, W_terms):
                    u_vec = np.zeros(n * m)
                    v_vec = np.zeros(m * p)
                    w_vec = np.zeros(p * n)
                    for idx, val in u_t.items():
                        u_vec[idx] = val
                    for idx, val in v_t.items():
                        v_vec[idx] = val
                    for idx, val in w_t.items():
                        w_vec[idx] = val
                    U_cols.append(u_vec)
                    V_cols.append(v_vec)
                    W_cols.append(w_vec)
            elif item[0] == 'direct':
                i, j, k = item[1], item[2], item[3]
                u_vec = np.zeros(n * m)
                v_vec = np.zeros(m * p)
                w_vec = np.zeros(p * n)
                u_vec[i * m + j] = 1
                v_vec[j * p + k] = 1
                w_vec[k * n + i] = 1
                U_cols.append(u_vec)
                V_cols.append(v_vec)
                W_cols.append(w_vec)
        
        if not U_cols:
            return None
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        return (U, V, W)
    
    def _make_block_decomposition(blocks, remaining_handling='pair_k'):
        """Create a decomposition from given Strassen blocks and handle remaining terms."""
        U_cols = []
        V_cols = []
        W_cols = []
        
        covered = set()
        for ib, jb, kb in blocks:
            for i in range(ib, min(ib+2, n)):
                for j in range(jb, min(jb+2, m)):
                    for k in range(kb, min(kb+2, p)):
                        covered.add((i, j, k))
            
            U_terms, V_terms, W_terms = _strassen_2x2_block(ib, jb, kb)
            for u_t, v_t, w_t in zip(U_terms, V_terms, W_terms):
                u_vec = np.zeros(n * m)
                v_vec = np.zeros(m * p)
                w_vec = np.zeros(p * n)
                for idx, val in u_t.items():
                    u_vec[idx] = val
                for idx, val in v_t.items():
                    v_vec[idx] = val
                for idx, val in w_t.items():
                    w_vec[idx] = val
                U_cols.append(u_vec)
                V_cols.append(v_vec)
                W_cols.append(w_vec)
        
        # Handle remaining terms - group by (i,j) and pair k values
        remaining = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        if (i, j) not in remaining:
                            remaining[(i, j)] = []
                        remaining[(i, j)].append(k)
        
        for (i, j), ks in remaining.items():
            ks = sorted(ks)
            if remaining_handling == 'pair_k':
                # Pair k's together: U has 1 at (i,j), V has 1 at (j,k1) and (j,k2), W has 1 at (k1,i) and (k2,i)
                # This gives contribution: U[i,j]*V[j,k1]*W[k1,i] + U[i,j]*V[j,k2]*W[k2,i] = 1*1*1 + 1*1*1 = 2
                # That's wrong! We need each to contribute 1.
                # Actually the decomposition is: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
                # If we have one term with U[i,j]=1, V[j,k1]=1, W[k1,i]=1 and another with V[j,k2]=1, W[k2,i]=1
                # then contribution to (i,j,k1) = 1*1*1 = 1 (from first term only if W[k1,i] is only in first)
                # Wait, this doesn't work that way. Let me think again.
                # The tensor is T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # For a single term r: it contributes U[i,j,r]*V[j,k,r]*W[k,i,r] to T[i,j,k]
                # If U[i,j,r]=1, V[j,k1,r]=1, W[k1,i,r]=1, then T[i,j,k1] gets +1, T[i,j,k2] gets 0 (since V[j,k2,r]=0)
                # So we need separate terms for each k.
                # Pairing doesn't help for the direct terms - each (i,j,k) needs its own term or shared structure.
                for k1 in ks:
                    u_vec = np.zeros(n * m)
                    v_vec = np.zeros(m * p)
                    w_vec = np.zeros(p * n)
                    u_vec[i * m + j] = 1
                    v_vec[j * p + k1] = 1
                    w_vec[k1 * n + i] = 1
                    U_cols.append(u_vec)
                    V_cols.append(v_vec)
                    W_cols.append(w_vec)
        
        if not U_cols:
            return None
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        return (U, V, W)
    
    # Systematically try all combinations of Strassen blocks + direct terms
    # A Strassen block at (ib,jb,kb) covers (ib..ib+1, jb..jb+1, kb..kb+1) = 8 elements with 7 terms
    # Direct terms cover 1 element each
    
    # Try all single Strassen block placements
    for ib in range(n - 1):
        for jb in range(m - 1):
            for kb in range(p - 1):
                if _time_exceeded():
                    return best_result
                # Build spec: one strassen + all uncovered direct
                spec = [('strassen', ib, jb, kb)]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if not (ib <= i <= ib+1 and jb <= j <= jb+1 and kb <= k <= kb+1):
                                spec.append(('direct', i, j, k))
                result = _build_from_spec(spec)
                if result is not None:
                    U, V, W = result
                    if _verify(U, V, W) and U.shape[1] < best_rank:
                        best_rank = U.shape[1]
                        best_result = (U, V, W)
    
    if _time_exceeded():
        return best_result
    
    # Try pairs of non-overlapping Strassen blocks
    block_positions = [(ib, jb, kb) for ib in range(n-1) for jb in range(m-1) for kb in range(p-1)]
    for idx1 in range(len(block_positions)):
        if _time_exceeded():
            return best_result
        for idx2 in range(idx1+1, len(block_positions)):
            ib1, jb1, kb1 = block_positions[idx1]
            ib2, jb2, kb2 = block_positions[idx2]
            # Check if blocks don't overlap in a way that causes issues
            # They can overlap - the key is that the sum of contributions equals 1
            # For now, only try non-overlapping blocks
            overlap_i = not (ib1+1 < ib2 or ib2+1 < ib1)
            overlap_j = not (jb1+1 < jb2 or jb2+1 < jb1)  
            overlap_k = not (kb1+1 < kb2 or kb2+1 < kb1)
            # Blocks are in different "regions" if they don't share any (i,j,k)
            shares = False
            for i in range(max(ib1,ib2), min(ib1+2,ib2+2)):
                for j in range(max(jb1,jb2), min(jb1+2,jb2+2)):
                    for k in range(max(kb1,kb2), min(kb1+2,kb2+2)):
                        if i < n and j < m and k < p:
                            shares = True
            if shares:
                continue  # Skip overlapping blocks for now
            
            spec = [('strassen', ib1, jb1, kb1), ('strassen', ib2, jb2, kb2)]
            covered = set()
            for ib, jb, kb in [(ib1,jb1,kb1), (ib2,jb2,kb2)]:
                for i in range(ib, min(ib+2,n)):
                    for j in range(jb, min(jb+2,m)):
                        for k in range(kb, min(kb+2,p)):
                            covered.add((i,j,k))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i,j,k) not in covered:
                            spec.append(('direct', i, j, k))
            
            result = _build_from_spec(spec)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W) and U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_result = (U, V, W)
    
    return best_result

def _local_search_refinement(U, V, W, deadline=None):
    """
    Try to reduce rank by removing columns that are linearly dependent 
    in the tensor product sense, or by finding equivalent lower-rank representations.
    """
    def _time_exceeded():
        return deadline is not None and time.time() > deadline
    
    r = U.shape[1]
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = r
    
    # Try removing each column and checking if the decomposition still works
    for col in range(r):
        if _time_exceeded():
            break
        if current_rank <= 1:
            break
        
        # Skip this column
        mask = [i for i in range(current_rank) if i != col]
        if len(mask) == 0:
            continue
            
        U_test = current_U[:, mask]
        V_test = current_V[:, mask]
        W_test = current_W[:, mask]
        
        if _verify(U_test, V_test, W_test):
            current_U, current_V, current_W = U_test, V_test, W_test
            current_rank -= 1
    
    return (current_U, current_V, current_W)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_rank = n * m * p  # schoolbook
    best_result = None
    
    def _time_exceeded():
        return time.time() > deadline
    
    # First, try all single-block strategies and combinations
    result = _try_different_blockings(deadline)
    if result is not None:
        U, V, W = result
        if _verify(U, V, W) and U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_result = (U, V, W)
    
    if _time_exceeded():
        if best_result is not None:
            U, V, W = best_result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return _schoolbook_fallback()
    
    # Try systematic local search: start from best found, try to merge/remove columns
    if best_result is not None:
        U, V, W = best_result
        U, V, W = _local_search_refinement(U, V, W, deadline)
        if _verify(U, V, W) and U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_result = (U, V, W)
    
    if _time_exceeded():
        if best_result is not None:
            U, V, W = best_result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return _schoolbook_fallback()
    
    # Try random perturbation + local search from schoolbook
    if best_rank > 15:
        U_rand, V_rand, W_rand = _random_start(deadline)
        if U_rand is not None:
            U_rand, V_rand, W_rand = _local_search_refinement(U_rand, V_rand, W_rand, deadline)
            if _verify(U_rand, V_rand, W_rand) and U_rand.shape[1] < best_rank:
                best_rank = U_rand.shape[1]
                best_result = (U_rand, V_rand, W_rand)
    
    if _time_exceeded():
        if best_result is not None:
            U, V, W = best_result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return _schoolbook_fallback()
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return _schoolbook_fallback()

def _schoolbook_fallback():
    U_sb = np.zeros((n * m, n * m * p))
    V_sb = np.zeros((m * p, n * m * p))
    W_sb = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def _random_start(deadline):
    """Try random initial decomposition and refine."""
    import random
    random.seed(42)
    
    def _time_exceeded():
        return time.time() > deadline
    
    # Start with schoolbook and try random column removals
    U_sb = np.zeros((n * m, n * m * p))
    V_sb = np.zeros((m * p, n * m * p))
    W_sb = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try removing pairs of columns that might be redundant
    r = n * m * p
    cols = list(range(r))
    random.shuffle(cols)
    
    current_U, current_V, current_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    current_rank = r
    
    for _ in range(200):
        if _time_exceeded():
            break
        if current_rank <= 1:
            break
        # Try removing a random column
        col_idx = random.randint(0, current_rank - 1)
        mask = [i for i in range(current_rank) if i != col_idx]
        if len(mask) == 0:
            break
        U_test = current_U[:, mask]
        V_test = current_V[:, mask]
        W_test = current_W[:, mask]
        if _verify(U_test, V_test, W_test):
            current_U, current_V, current_W = U_test, V_test, W_test
            current_rank -= 1
    
    if current_rank < r:
        return (current_U, current_V, current_W)
    return (None, None, None)

def _strassen_2x2_block(i_base, j_base, k_base):
    """
    Returns the U, V, W vectors for a 2x2x2 Strassen block.
    Correct Strassen: 7 multiplications for 2x2x2 matmul.
    """
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # M1 = (a11+a22)(b11+b22)
    U1 = {u_idx(i_base, j_base): 1, u_idx(i_base+1, j_base+1): 1}
    V1 = {v_idx(j_base, k_base): 1, v_idx(j_base+1, k_base+1): 1}
    W1 = {w_idx(k_base, i_base): 1, w_idx(k_base+1, i_base+1): 1,
          w_idx(k_base, i_base+1): -1, w_idx(k_base+1, i_base): -1}
    U_terms.append(U1); V_terms.append(V1); W_terms.append(W1)
    
    # M2 = (a21+a22)b11
    U2 = {u_idx(i_base+1, j_base): 1, u_idx(i_base+1, j_base+1): 1}
    V2 = {v_idx(j_base, k_base): 1}
    W2 = {w_idx(k_base, i_base+1): 1}
    U_terms.append(U2); V_terms.append(V2); W_terms.append(W2)
    
    # M3 = a11(b21-b11)
    U3 = {u_idx(i_base, j_base): 1}
    V3 = {v_idx(j_base+1, k_base): 1, v_idx(j_base, k_base): -1}
    W3 = {w_idx(k_base, i_base): 1}
    U_terms.append(U3); V_terms.append(V3); W_terms.append(W3)
    
    # M4 = a22(b12-b22)
    U4 = {u_idx(i_base+1, j_base+1): 1}
    V4 = {v_idx(j_base, k_base+1): 1, v_idx(j_base+1, k_base+1): -1}
    W4 = {w_idx(k_base+1, i_base+1): 1}
    U_terms.append(U4); V_terms.append(V4); W_terms.append(W4)
    
    # M5 = (a11+a12)b22
    U5 = {u_idx(i_base, j_base): 1, u_idx(i_base, j_base+1): 1}
    V5 = {v_idx(j_base+1, k_base+1): 1}
    W5 = {w_idx(k_base+1, i_base): 1}
    U_terms.append(U5); V_terms.append(V5); W_terms.append(W5)
    
    # M6 = (a21-a11)(b11+b12)
    U6 = {u_idx(i_base+1, j_base): 1, u_idx(i_base, j_base): -1}
    V6 = {v_idx(j_base, k_base): 1, v_idx(j_base, k_base+1): 1}
    W6 = {w_idx(k_base, i_base+1): 1, w_idx(k_base+1, i_base+1): 1}
    U_terms.append(U6); V_terms.append(V6); W_terms.append(W6)
    
    # M7 = (a12-a22)(b21+b22)
    U7 = {u_idx(i_base, j_base+1): 1, u_idx(i_base+1, j_base+1): -1}
    V7 = {v_idx(j_base+1, k_base): 1, v_idx(j_base+1, k_base+1): 1}
    W7 = {w_idx(k_base, i_base): 1, w_idx(k_base+1, i_base): 1}
    U_terms.append(U7); V_terms.append(V7); W_terms.append(W7)
    
    return U_terms, V_terms, W_terms

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]) if res['U'] else 0)
    # Verify
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    print("Valid:", _verify(U, V, W))
