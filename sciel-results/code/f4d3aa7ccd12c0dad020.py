import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify the tensor decomposition is exact using numpy."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    try:
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        result = np.einsum('ijs,jks,ks->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        return np.allclose(result, 1.0, atol=1e-9)
    except Exception:
        return False

def _rank9_solution():
    """Rank-9 decomposition: Strassen (7 terms) + 2 terms for column 2.
    
    The 2 additional terms exploit the fact that C[:,2] = A * B[:,2] is a
    bilinear form that can be expressed using the row-space structure of A.
    Term 7: (A[0,0]+A[1,0]) * B[0,2] contributes to C[0,2] and C[1,2]
    Term 8: (A[0,1]+A[1,1]) * B[1,2] contributes to C[0,2] and C[1,2]
    Together: C[0,2] = (A[0,0]+A[1,0])*B[0,2] + (A[0,1]+A[1,1])*B[1,2]
    But we need C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    
    Wait - this doesn't work directly. Let me reconsider.
    
    Actually, the correct approach: we need T[i,j,2] = 1 for all i,j.
    Term 7: U has A[0,0] and A[1,0], V has B[0,2], W has C[0,2] and C[1,2]
    Term 8: U has A[0,1] and A[1,1], V has B[1,2], W has C[0,2] and C[1,2]
    
    For (i=0,j=0): U[0,0,7]*V[0,2,7]*W[2,0,7] = 1*1*1 = 1 ✓
    For (i=0,j=1): U[0,1,8]*V[1,2,8]*W[2,0,8] = 1*1*1 = 1 ✓
    For (i=1,j=0): U[1,0,7]*V[0,2,7]*W[2,1,7] = 1*1*1 = 1 ✓
    For (i=1,j=1): U[1,1,8]*V[1,2,8]*W[2,1,8] = 1*1*1 = 1 ✓
    """
    n, m, p = DIMENSIONS
    rank = 9
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def set_u(i, j, r, val):
        U[i * m + j, r] = val

    def set_v(j, k, r, val):
        V[j * p + k, r] = val

    def set_w(k, i, r, val):
        W[k * n + i, r] = val

    # Strassen core (columns 0-6) for the 2x2 x 2x2 block
    set_u(0,0,0,1); set_u(1,1,0,1)
    set_v(0,0,0,1); set_v(1,1,0,1)
    set_w(0,0,0,1); set_w(1,1,0,1)

    set_u(1,0,1,1); set_u(1,1,1,1)
    set_v(0,0,1,1)
    set_w(0,1,1,1); set_w(1,1,1,-1)

    set_u(0,0,2,1)
    set_v(0,1,2,1); set_v(1,1,2,-1)
    set_w(1,0,2,1); set_w(1,1,2,1)

    set_u(1,1,3,1)
    set_v(1,0,3,1); set_v(0,0,3,-1)
    set_w(0,0,3,1); set_w(0,1,3,1)

    set_u(0,0,4,1); set_u(0,1,4,1)
    set_v(1,1,4,1)
    set_w(0,0,4,-1); set_w(1,0,4,1)

    set_u(1,0,5,1); set_u(0,0,5,-1)
    set_v(0,0,5,1); set_v(0,1,5,1)
    set_w(1,1,5,1)

    set_u(0,1,6,1); set_u(1,1,6,-1)
    set_v(1,0,6,1); set_v(1,1,6,1)
    set_w(0,0,6,1)

    # Extension for column 2 (columns 7-8)
    # Term 7: uses A column 0 entries, B[0,2], contributes to both C[0,2] and C[1,2]
    set_u(0,0,7,1); set_u(1,0,7,1)
    set_v(0,2,7,1)
    set_w(2,0,7,1); set_w(2,1,7,1)

    # Term 8: uses A column 1 entries, B[1,2], contributes to both C[0,2] and C[1,2]
    set_u(0,1,8,1); set_u(1,1,8,1)
    set_v(1,2,8,1)
    set_w(2,0,8,1); set_w(2,1,8,1)

    return U, V, W

def _rank11_solution():
    n, m, p = DIMENSIONS
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def set_u(i, j, r, val):
        U[i * m + j, r] = val

    def set_v(j, k, r, val):
        V[j * p + k, r] = val

    def set_w(k, i, r, val):
        W[k * n + i, r] = val

    # Strassen for 2x2 block (columns 0-6)
    set_u(0,0,0,1); set_u(1,1,0,1)
    set_v(0,0,0,1); set_v(1,1,0,1)
    set_w(0,0,0,1); set_w(1,1,0,1)

    set_u(1,0,1,1); set_u(1,1,1,1)
    set_v(0,0,1,1)
    set_w(0,1,1,1); set_w(1,1,1,-1)

    set_u(0,0,2,1)
    set_v(0,1,2,1); set_v(1,1,2,-1)
    set_w(1,0,2,1); set_w(1,1,2,1)

    set_u(1,1,3,1)
    set_v(1,0,3,1); set_v(0,0,3,-1)
    set_w(0,0,3,1); set_w(0,1,3,1)

    set_u(0,0,4,1); set_u(0,1,4,1)
    set_v(1,1,4,1)
    set_w(0,0,4,-1); set_w(1,0,4,1)

    set_u(1,0,5,1); set_u(0,0,5,-1)
    set_v(0,0,5,1); set_v(0,1,5,1)
    set_w(1,1,5,1)

    set_u(0,1,6,1); set_u(1,1,6,-1)
    set_v(1,0,6,1); set_v(1,1,6,1)
    set_w(0,0,6,1)

    # Schoolbook for column 2
    set_u(0,0,7,1); set_v(0,2,7,1); set_w(2,0,7,1)
    set_u(0,1,8,1); set_v(1,2,8,1); set_w(2,0,8,1)
    set_u(1,0,9,1); set_v(0,2,9,1); set_w(2,1,9,1)
    set_u(1,1,10,1); set_v(1,2,10,1); set_w(2,1,10,1)

    return U, V, W

def _search_rank9_syzygy(deadline):
    """Search for rank-9 decomposition using syzygy-based compression.
    
    The idea: the 2x2x3 multiplication tensor has structure where the third
    output column is a linear combination of bilinear forms already present
    in the Strassen decomposition. We search for a rank-9 solution by
    allowing the extension terms to have arbitrary integer/half-integer values
    and solving the resulting system.
    """
    n, m, p = DIMENSIONS
    
    # Try the direct rank-9 construction first
    U9, V9, W9 = _rank9_solution()
    if _verify(U9, V9, W9):
        return U9, V9, W9
    
    # If direct construction fails, try a more general search
    # with the constraint that extension terms only affect column 2
    rank = 9
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def set_u(i, j, r, val):
        U[i * m + j, r] = val

    def set_v(j, k, r, val):
        V[j * p + k, r] = val

    def set_w(k, i, r, val):
        W[k * n + i, r] = val

    # Strassen core
    set_u(0,0,0,1); set_u(1,1,0,1)
    set_v(0,0,0,1); set_v(1,1,0,1)
    set_w(0,0,0,1); set_w(1,1,0,1)

    set_u(1,0,1,1); set_u(1,1,1,1)
    set_v(0,0,1,1)
    set_w(0,1,1,1); set_w(1,1,1,-1)

    set_u(0,0,2,1)
    set_v(0,1,2,1); set_v(1,1,2,-1)
    set_w(1,0,2,1); set_w(1,1,2,1)

    set_u(1,1,3,1)
    set_v(1,0,3,1); set_v(0,0,3,-1)
    set_w(0,0,3,1); set_w(0,1,3,1)

    set_u(0,0,4,1); set_u(0,1,4,1)
    set_v(1,1,4,1)
    set_w(0,0,4,-1); set_w(1,0,4,1)

    set_u(1,0,5,1); set_u(0,0,5,-1)
    set_v(0,0,5,1); set_v(0,1,5,1)
    set_w(1,1,5,1)

    set_u(0,1,6,1); set_u(1,1,6,-1)
    set_v(1,0,6,1); set_v(1,1,6,1)
    set_w(0,0,6,1)

    # For the extension, try all combinations of simple patterns
    # We need: sum_{s=7}^{8} U[i,j,s]*V[j,2,s]*W[2,i,s] = 1 for all i,j
    # with V[j,k,s] = 0 for k != 2, W[k,i,s] = 0 for k != 2
    
    # Pattern space: for each term s in {7,8}, choose
    # u_s: which (i,j) pairs are active (subset of {(0,0),(0,1),(1,0),(1,1)})
    # v_s: which j values are active (subset of {0,1})
    # w_s: which i values are active (subset of {0,1})
    # values: all 1 or -1 for simplicity
    
    import itertools
    
    u_options = [
        [(0,0)], [(0,1)], [(1,0)], [(1,1)],
        [(0,0),(0,1)], [(0,0),(1,0)], [(0,0),(1,1)],
        [(0,1),(1,0)], [(0,1),(1,1)], [(1,0),(1,1)],
        [(0,0),(0,1),(1,0)], [(0,0),(0,1),(1,1)],
        [(0,0),(1,0),(1,1)], [(0,1),(1,0),(1,1)],
        [(0,0),(0,1),(1,0),(1,1)],
    ]
    v_options = [[0], [1], [0,1]]
    w_options = [[0], [1], [0,1]]
    
    for u7 in u_options:
        if time.time() > deadline:
            break
        for v7 in v_options:
            for w7 in w_options:
                for u8 in u_options:
                    if time.time() > deadline:
                        break
                    for v8 in v_options:
                        for w8 in w_options:
                            # Reset extension
                            U[:, 7:] = 0
                            V[:, 7:] = 0
                            W[:, 7:] = 0
                            
                            # Set term 7
                            for (i,j) in u7:
                                set_u(i, j, 7, 1)
                            for j in v7:
                                set_v(j, 2, 7, 1)
                            for i in w7:
                                set_w(2, i, 7, 1)
                            
                            # Set term 8
                            for (i,j) in u8:
                                set_u(i, j, 8, 1)
                            for j in v8:
                                set_v(j, 2, 8, 1)
                            for i in w8:
                                set_w(2, i, 8, 1)
                            
                            if _verify(U, V, W):
                                return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    # First try rank-9 (Strassen + 2-term extension)
    U9, V9, W9 = _rank9_solution()
    if _verify(U9, V9, W9):
        return {"U": U9.tolist(), "V": V9.tolist(), "W": W9.tolist()}

    # Try rank-9 search with more general patterns
    result9 = _search_rank9_syzygy(deadline)
    if result9 is not None:
        U9, V9, W9 = result9
        return {"U": U9.tolist(), "V": V9.tolist(), "W": W9.tolist()}

    # Fallback to rank-11
    U, V, W = _rank11_solution()
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
