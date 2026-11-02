"""Matrix multiplication 3x5x7 - rank minimization via block-structured decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _build_strassen_2x2x2():
    """Build rank-7 Strassen decomposition for 2x2 * 2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = (b-c)h
    U[1, 3] = 1; U[2, 3] = -1
    V[3, 3] = 1
    W[1, 3] = 1
    
    # M5 = (a+b)g
    U[0, 4] = 1; U[1, 4] = 1
    V[2, 4] = 1
    W[3, 4] = 1
    
    # M6 = (c-d)(f+h)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[3, 5] = 1
    W[2, 5] = 1
    
    # M7 = (a-c)(e+g)
    U[0, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_schoolbook_block(nb, mb, pb):
    """Build schoolbook decomposition for nb x mb * mb x pb."""
    rank = nb * mb * pb
    U = np.zeros((nb*mb, rank))
    V = np.zeros((mb*pb, rank))
    W = np.zeros((pb*nb, rank))
    col = 0
    for i in range(nb):
        for j in range(mb):
            for k in range(pb):
                U[i*mb+j, col] = 1
                V[j*pb+k, col] = 1
                W[k*nb+i, col] = 1
                col += 1
    return U, V, W

def _build_block_decomp(n, m, p, n_parts, m_parts, p_parts):
    """
    Build block decomposition for n x m * m x p.
    n_parts: list of row block sizes summing to n
    m_parts: list of col block sizes summing to m  
    p_parts: list of output col block sizes summing to p
    Uses Strassen for 2x2x2 blocks, schoolbook otherwise.
    """
    U = np.zeros((n*m, 0))
    V = np.zeros((m*p, 0))
    W = np.zeros((p*n, 0))
    
    # Compute offsets
    n_offsets = [0]
    for size in n_parts:
        n_offsets.append(n_offsets[-1] + size)
    m_offsets = [0]
    for size in m_parts:
        m_offsets.append(m_offsets[-1] + size)
    p_offsets = [0]
    for size in p_parts:
        p_offsets.append(p_offsets[-1] + size)
    
    col_offset = 0
    for ni, nb in enumerate(n_parts):
        for mi, mb in enumerate(m_parts):
            for pi, pb in enumerate(p_parts):
                if nb == 2 and mb == 2 and pb == 2:
                    Ub, Vb, Wb = _build_strassen_2x2x2()
                else:
                    Ub, Vb, Wb = _build_schoolbook_block(nb, mb, pb)
                
                r = Ub.shape[1]
                
                # Create full-size matrices with this block placed
                U_block = np.zeros((n*m, r))
                V_block = np.zeros((m*p, r))
                W_block = np.zeros((p*n, r))
                
                n_off = n_offsets[ni]
                m_off = m_offsets[mi]
                p_off = p_offsets[pi]
                
                # U: rows (n_off + i) * m + (m_off + j) for i in [0,nb), j in [0,mb)
                for i in range(nb):
                    for j in range(mb):
                        row_global = (n_off + i) * m + (m_off + j)
                        row_local = i * mb + j
                        U_block[row_global, :] = Ub[row_local, :]
                
                # V: rows (m_off + j) * p + (p_off + k) for j in [0,mb), k in [0,pb)
                for j in range(mb):
                    for k in range(pb):
                        row_global = (m_off + j) * p + (p_off + k)
                        row_local = j * pb + k
                        V_block[row_global, :] = Vb[row_local, :]
                
                # W: rows (p_off + k) * n + (n_off + i) for k in [0,pb), i in [0,nb)
                for k in range(pb):
                    for i in range(nb):
                        row_global = (p_off + k) * n + (n_off + i)
                        row_local = k * nb + i
                        W_block[row_global, :] = Wb[row_local, :]
                
                U = np.hstack([U, U_block])
                V = np.hstack([V, V_block])
                W = np.hstack([W, W_block])
                col_offset += r
    
    return U, V, W

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using numpy."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                val = np.sum(u_row * v_row * w_row)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_fraction(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def _to_fraction_list(mat):
    """Convert numpy array to list of lists of Fractions."""
    return [[Fraction(float(x)).limit_denominator(100) for x in row] for row in mat]

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 3, 5, 7
    
    # Try different block partitions
    partitions = [
        # (n_parts, m_parts, p_parts)
        ([2,1], [2,3], [3,4]),
        ([2,1], [3,2], [4,3]),
        ([1,2], [2,3], [3,4]),
        ([2,1], [2,3], [2,5]),
        ([2,1], [2,3], [5,2]),
        ([1,2], [3,2], [3,4]),
        ([2,1], [1,4], [3,4]),
        ([2,1], [4,1], [3,4]),
    ]
    
    best_result = None
    best_rank = n * m * p
    
    for n_parts, m_parts, p_parts in partitions:
        if time.time() > deadline:
            break
        
        # Validate partitions
        if sum(n_parts) != n or sum(m_parts) != m or sum(p_parts) != p:
            continue
        
        try:
            U, V, W = _build_block_decomp(n, m, p, n_parts, m_parts, p_parts)
            rank = U.shape[1]
            
            if rank >= best_rank:
                continue
            
            # Verify
            if _verify_np(U, V, W, n, m, p):
                # Convert to fractions for exact verification
                U_frac = _to_fraction_list(U)
                V_frac = _to_fraction_list(V)
                W_frac = _to_fraction_list(W)
                
                if _verify_fraction(U_frac, V_frac, W_frac, n, m, p):
                    best_result = {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }
                    best_rank = rank
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
