import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0 if j == k else 0.0
                    actual = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(actual - expected) > 1e-9:
                        return False, (i, j, k, actual, expected)
        return True, None

    def strassen_2x2x2(i0, j0, k0, n_dim, m_dim, p_dim):
        """
        Strassen 2x2x2 decomposition for block at (i0, j0, k0).
        Handles A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] = C[i0:i0+2, k0:k0+2]
        
        U index: i * m_dim + j
        V index: j * p_dim + k
        W index: k * n_dim + i
        """
        cols = []
        
        # M1 = (A11+A22)(B11+B22)
        # A11 = A[i0,j0], A22 = A[i0+1,j0+1]
        # B11 = B[j0,k0], B22 = B[j0+1,k0+1]
        # C11 += M1, C22 += M1
        cols.append((
            {i0*m_dim+j0: 1, (i0+1)*m_dim+(j0+1): 1},
            {j0*p_dim+k0: 1, (j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: 1, (k0+1)*n_dim+(i0+1): 1}
        ))
        
        # M2 = (A21+A22)B11
        # A21 = A[i0+1,j0], A22 = A[i0+1,j0+1]
        # B11 = B[j0,k0]
        # C21 += M2, C22 -= M2
        cols.append((
            {(i0+1)*m_dim+j0: 1, (i0+1)*m_dim+(j0+1): 1},
            {j0*p_dim+k0: 1},
            {k0*n_dim+(i0+1): 1, (k0+1)*n_dim+(i0+1): -1}
        ))
        
        # M3 = A11(B12-B22)
        # A11 = A[i0,j0]
        # B12 = B[j0,k0+1], B22 = B[j0+1,k0+1]
        # C12 += M3, C22 += M3
        cols.append((
            {i0*m_dim+j0: 1},
            {j0*p_dim+(k0+1): 1, (j0+1)*p_dim+(k0+1): -1},
            {(k0+1)*n_dim+i0: 1, (k0+1)*n_dim+(i0+1): 1}
        ))
        
        # M4 = A22(B21-B11)
        # A22 = A[i0+1,j0+1]
        # B21 = B[j0+1,k0], B11 = B[j0,k0]
        # C11 += M4, C21 += M4
        cols.append((
            {(i0+1)*m_dim+(j0+1): 1},
            {(j0+1)*p_dim+k0: 1, j0*p_dim+k0: -1},
            {k0*n_dim+i0: 1, k0*n_dim+(i0+1): 1}
        ))
        
        # M5 = (A11+A12)B22
        # A11 = A[i0,j0], A12 = A[i0,j0+1]
        # B22 = B[j0+1,k0+1]
        # C12 -= M5, C22 += M5
        cols.append((
            {i0*m_dim+j0: 1, i0*m_dim+(j0+1): 1},
            {(j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: -1, (k0+1)*n_dim+i0: 1}
        ))
        
        # M6 = (A21-A11)(B11+B12)
        # A21 = A[i0+1,j0], A11 = A[i0,j0]
        # B11 = B[j0,k0], B12 = B[j0,k0+1]
        # C21 += M6
        cols.append((
            {(i0+1)*m_dim+j0: 1, i0*m_dim+j0: -1},
            {j0*p_dim+k0: 1, j0*p_dim+(k0+1): 1},
            {(k0+1)*n_dim+(i0+1): 1}
        ))
        
        # M7 = (A12-A22)(B21+B22)
        # A12 = A[i0,j0+1], A22 = A[i0+1,j0+1]
        # B21 = B[j0+1,k0], B22 = B[j0+1,k0+1]
        # C12 += M7
        cols.append((
            {i0*m_dim+(j0+1): 1, (i0+1)*m_dim+(j0+1): -1},
            {(j0+1)*p_dim+k0: 1, (j0+1)*p_dim+(k0+1): 1},
            {(k0+1)*n_dim+i0: 1}
        ))
        
        return cols

    def naive_block(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim):
        """Naive decomposition of an nc x mc x pc block."""
        cols = []
        for di in range(nc):
            for dj in range(mc):
                for dk in range(pc):
                    i, j, k = ii+di, jj+dj, kk+dk
                    u_idx = i * m_dim + j
                    v_idx = j * p_dim + k
                    w_idx = k * n_dim + i
                    cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
        return cols

    def decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim):
        """
        Find minimum-rank decomposition of an nc x mc x pc block.
        Uses memoization and tries all reasonable splits.
        """
        if time.time() > deadline:
            return naive_block(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim)
        
        # Base case
        if nc == 1 and mc == 1 and pc == 1:
            i, j, k = ii, jj, kk
            u_idx = i * m_dim + j
            v_idx = j * p_dim + k
            w_idx = k * n_dim + i
            return [({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0})]
        
        best_cols = None
        best_rank = float('inf')
        
        # Try Strassen if all dims >= 2
        if nc >= 2 and mc >= 2 and pc >= 2:
            s_cols = strassen_2x2x2(ii, jj, kk, n_dim, m_dim, p_dim)
            
            # Remainders
            rem = list(s_cols)
            if nc > 2:
                rem.extend(decompose(nc-2, mc, pc, ii+2, jj, kk, n_dim, m_dim, p_dim))
            if mc > 2:
                rem.extend(decompose(2, mc-2, pc, ii, jj+2, kk, n_dim, m_dim, p_dim))
            if pc > 2:
                rem.extend(decompose(2, 2, pc-2, ii, jj, kk+2, n_dim, m_dim, p_dim))
            
            if len(rem) < best_rank:
                best_rank = len(rem)
                best_cols = rem
        
        # Try splitting n dimension
        if nc > 1:
            for split in range(1, nc):
                part1 = decompose(split, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim)
                part2 = decompose(nc-split, mc, pc, ii+split, jj, kk, n_dim, m_dim, p_dim)
                total = part1 + part2
                if len(total) < best_rank:
                    best_rank = len(total)
                    best_cols = total
        
        # Try splitting m dimension
        if mc > 1:
            for split in range(1, mc):
                part1 = decompose(nc, split, pc, ii, jj, kk, n_dim, m_dim, p_dim)
                part2 = decompose(nc, mc-split, pc, ii, jj+split, kk, n_dim, m_dim, p_dim)
                total = part1 + part2
                if len(total) < best_rank:
                    best_rank = len(total)
                    best_cols = total
        
        # Try splitting p dimension
        if pc > 1:
            for split in range(1, pc):
                part1 = decompose(nc, mc, split, ii, jj, kk, n_dim, m_dim, p_dim)
                part2 = decompose(nc, mc, pc-split, ii, jj, kk+split, n_dim, m_dim, p_dim)
                total = part1 + part2
                if len(total) < best_rank:
                    best_rank = len(total)
                    best_cols = total
        
        if best_cols is None:
            best_cols = naive_block(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim)
        
        return best_cols

    cols = decompose(n, m, p, 0, 0, 0, n, m, p)
    rank = len(cols)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(cols):
        for idx, val in u_dict.items():
            U[idx, r] = val
        for idx, val in v_dict.items():
            V[idx, r] = val
        for idx, val in w_dict.items():
            W[idx, r] = val
    
    # Verify
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if not valid:
        # Fallback to naive if verification fails
        cols = naive_block(n, m, p, 0, 0, 0, n, m, p)
        rank = len(cols)
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for r, (u_dict, v_dict, w_dict) in enumerate(cols):
            for idx, val in u_dict.items():
                U[idx, r] = val
            for idx, val in v_dict.items():
                V[idx, r] = val
            for idx, val in w_dict.items():
                W[idx, r] = val
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
