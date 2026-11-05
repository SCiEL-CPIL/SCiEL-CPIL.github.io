import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _block_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
    """Generate a decomposition using block-wise Strassen."""
    # n_blocks: list of block sizes for n dimension (rows of A)
    # m_blocks: list of block sizes for m dimension (cols of A / rows of B)
    # p_blocks: list of block sizes for p dimension (cols of B)
    
    # Compute block offsets
    n_offsets = [0]
    for s in n_blocks:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_blocks:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_blocks:
        p_offsets.append(p_offsets[-1] + s)
    
    # Count total rank
    rank = 0
    terms = []
    for ni in range(len(n_blocks)):
        for mi in range(len(m_blocks)):
            for pi in range(len(p_blocks)):
                ns = n_blocks[ni]
                ms = m_blocks[mi]
                ps = p_blocks[pi]
                if ns == 2 and ms == 2 and ps == 2:
                    # Strassen: 7 terms
                    rank += 7
                    terms.append((ni, mi, pi, 'strassen'))
                else:
                    # Trivial: ns*ms*ps terms
                    rank += ns * ms * ps
                    terms.append((ni, mi, pi, 'trivial'))
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for ni, mi, pi, method in terms:
        n0 = n_offsets[ni]
        m0 = m_offsets[mi]
        p0 = p_offsets[pi]
        ns = n_blocks[ni]
        ms = m_blocks[mi]
        ps = p_blocks[pi]
        
        if method == 'strassen':
            # Strassen for 2x2 x 2x2
            # A block: rows n0..n0+2, cols m0..m0+2
            # B block: rows m0..m0+2, cols p0..p0+2
            # C block: rows n0..n0+2, cols p0..p0+2
            
            # Standard Strassen:
            # M1 = (A11 + A22) × (B11 + B22)
            # M2 = (A21 + A22) × B11
            # M3 = A11 × (B12 − B22)
            # M4 = A22 × (B21 − B11)
            # M5 = (A11 + A12) × B22
            # M6 = (A21 − A11) × (B11 + B12)
            # M7 = (A12 − A22) × (B21 + B22)
            #
            # C11 = M1 + M4 − M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 − M2 + M3 + M6
            
            strassen_data = [
                # (A_factors, B_factors, C_factors)
                # M1
                ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
                # M2
                ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
                # M3
                ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
                # M4
                ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
                # M5
                ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
                # M6
                ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
                # M7
                ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
            ]
            
            for a_factors, b_factors, c_factors in strassen_data:
                for (ar, ac, acoeff) in a_factors:
                    U[(n0+ar)*m + (m0+ac), col] += acoeff
                for (br, bc, bcoeff) in b_factors:
                    V[(m0+br)*p + (p0+bc), col] += bcoeff
                for (cr, cc, ccoeff) in c_factors:
                    # cr is row of C (i), cc is col of C (k)
                    # W is indexed by (k, i) → W[k*n + i]
                    W[(p0+cc)*n + (n0+cr), col] += ccoeff
                col += 1
        
        else:
            # Trivial multiplication
            for i in range(ns):
                for j in range(ms):
                    for k in range(ps):
                        U[(n0+i)*m + (m0+j), col] = 1
                        V[(m0+j)*p + (p0+k), col] = 1
                        W[(p0+k)*n + (n0+i), col] = 1
                        col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    return U, V, W

def _find_best_block_split(n, m, p):
    """Search over block splits to find the one with minimum rank."""
    best_rank = n * m * p
    best_split = None
    
    # Generate all possible block splits for each dimension
    def get_splits(dim):
        splits = []
        # All compositions of dim into parts of size 1 and 2
        # For dim=3: [3], [2,1], [1,2], [1,1,1]
        # For dim=5: [5], [4,1], [3,2], [2,3], [1,4], [2,1,2], [1,2,2], [2,2,1], [1,1,3], [3,1,1], [1,3,1], [1,1,1,2], [1,1,2,1], [1,2,1,1], [2,1,1,1], [1,1,1,1,1]
        # This is getting complex. Let me just try a few common splits.
        return None
    
    # Let me just try a few specific splits
    n_splits = [[3], [2,1], [1,2], [1,1,1]]
    m_splits = [[5], [2,2,1], [2,1,2], [1,2,2], [3,2], [2,3], [1,4], [4,1], [1,1,3], [3,1,1], [1,3,1], [1,1,1,2], [1,1,2,1], [1,2,1,1], [2,1,1,1], [1,1,1,1,1]]
    p_splits = [[7], [2,2,2,1], [2,2,1,2], [2,1,2,2], [1,2,2,2], [3,2,2], [2,3,2], [2,2,3], [1,3,3], [3,1,3], [3,3,1], [1,1,5], [5,1,1], [1,5,1], [2,1,4], [4,1,2], [1,2,4], [4,2,1], [2,4,1], [1,4,2], [1,1,1,4], [1,1,4,1], [1,4,1,1], [4,1,1,1], [1,1,2,3], [1,1,3,2], [1,2,1,3], [1,2,3,1], [1,3,1,2], [1,3,2,1], [2,1,1,3], [2,1,3,1], [2,3,1,1], [3,1,1,2], [3,1,2,1], [3,2,1,1], [1,1,1,1,3], [1,1,1,3,1], [1,1,3,1,1], [1,3,1,1,1], [3,1,1,1,1], [1,1,1,1,1,2], [1,1,1,1,2,1], [1,1,1,2,1,1], [1,1,2,1,1,1], [1,2,1,1,1,1], [2,1,1,1,1,1], [1,1,1,1,1,1,1]]
    
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                if sum(ns) != n or sum(ms) != m or sum(ps) != p:
                    continue
                U, V, W = _block_decomposition(n, m, p, ns, ms, ps)
                rank = U.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_split = (ns, ms, ps)
    
    return best_rank, best_split

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # First, get the trivial decomposition as fallback
    U_trivial, V_trivial, W_trivial = _block_decomposition(n, m, p, [n], [m], [p])
    
    # Search for better block splits
    best_rank, best_split = _find_best_block_split(n, m, p)
    
    if time.time() > deadline:
        return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
    
    if best_split is not None and best_rank < n*m*p:
        U, V, W = _block_decomposition(n, m, p, best_split[0], best_split[1], best_split[2])
    else:
        U, V, W = U_trivial, V_trivial, W_trivial
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
