import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    def strassen_2x2x2_cols(i0, j0, k0, n_dim, m_dim, p_dim):
        """
        Returns a list of 7 tuples (U_dict, V_dict, W_dict) for Strassen 2x2x2
        applied to the block at top-left (i0, j0, k0).
        U index: i * m_dim + j
        V index: j * p_dim + k
        W index: k * n_dim + i
        """
        cols = []
        
        # M1 = (A11+A22)(B11+B22)
        cols.append((
            {i0*m_dim+j0: 1, (i0+1)*m_dim+(j0+1): 1},
            {j0*p_dim+k0: 1, (j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: 1, (k0+1)*n_dim+(i0+1): 1}
        ))
        
        # M2 = (A21+A22)B11
        cols.append((
            {(i0+1)*m_dim+j0: 1, (i0+1)*m_dim+(j0+1): 1},
            {j0*p_dim+k0: 1},
            {k0*n_dim+(i0+1): 1, (k0+1)*n_dim+(i0+1): -1}
        ))
        
        # M3 = A11(B12-B22)
        cols.append((
            {i0*m_dim+j0: 1},
            {j0*p_dim+(k0+1): 1, (j0+1)*p_dim+(k0+1): -1},
            {(k0+1)*n_dim+i0: 1, (k0+1)*n_dim+(i0+1): 1}
        ))
        
        # M4 = A22(B21-B11)
        cols.append((
            {(i0+1)*m_dim+(j0+1): 1},
            {(j0+1)*p_dim+k0: 1, j0*p_dim+k0: -1},
            {k0*n_dim+i0: 1, k0*n_dim+(i0+1): 1}
        ))
        
        # M5 = (A11+A12)B22
        cols.append((
            {i0*m_dim+j0: 1, i0*m_dim+(j0+1): 1},
            {(j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: -1, (k0+1)*n_dim+i0: 1}
        ))
        
        # M6 = (A21-A11)(B11+B12)
        cols.append((
            {(i0+1)*m_dim+j0: 1, i0*m_dim+j0: -1},
            {j0*p_dim+k0: 1, j0*p_dim+(k0+1): 1},
            {(k0+1)*n_dim+(i0+1): 1}
        ))
        
        # M7 = (A12-A22)(B21+B22)
        cols.append((
            {i0*m_dim+(j0+1): 1, (i0+1)*m_dim+(j0+1): -1},
            {(j0+1)*p_dim+k0: 1, (j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: 1}
        ))
        
        return cols

    def decompose(n_curr, m_curr, p_curr, i0, j0, k0, n_dim, m_dim, p_dim):
        """
        Returns a list of (U_dict, V_dict, W_dict) for the block.
        """
        cols = []
        
        def recurse(nc, mc, pc, ii, jj, kk):
            if time.time() > deadline:
                return
            
            if nc == 1 and mc == 1 and pc == 1:
                u_idx = ii * m_dim + jj
                v_idx = jj * p_dim + kk
                w_idx = kk * n_dim + ii
                cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
                return

            # Try Strassen 2x2x2
            if nc >= 2 and mc >= 2 and pc >= 2:
                cols.extend(strassen_2x2x2_cols(ii, jj, kk, n_dim, m_dim, p_dim))
                
                # Recurse on remainders
                # 1. (nc-2) x mc x pc at (ii+2, jj, kk)
                if nc > 2:
                    recurse(nc - 2, mc, pc, ii + 2, jj, kk)
                
                # 2. 2 x (mc-2) x pc at (ii, jj+2, kk)
                if mc > 2:
                    recurse(2, mc - 2, pc, ii, jj + 2, kk)
                
                # 3. 2 x 2 x (pc-2) at (ii, jj, kk+2)
                if pc > 2:
                    recurse(2, 2, pc - 2, ii, jj, kk + 2)
                
                return

            # Fallback: split one dimension
            if nc > 1:
                recurse(1, mc, pc, ii, jj, kk)
                recurse(nc - 1, mc, pc, ii + 1, jj, kk)
            elif mc > 1:
                recurse(nc, 1, pc, ii, jj, kk)
                recurse(nc, mc - 1, pc, ii, jj + 1, kk)
            elif pc > 1:
                recurse(nc, mc, 1, ii, jj, kk)
                recurse(nc, mc, pc - 1, ii, jj, kk + 1)
            else:
                u_idx = ii * m_dim + jj
                v_idx = jj * p_dim + kk
                w_idx = kk * n_dim + ii
                cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))

        recurse(n_curr, m_curr, p_curr, i0, j0, k0)
        return cols

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
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
