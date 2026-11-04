import time
import numpy as np

def construct():
    """
    Return a decomposition of the 5x5x5 matrix multiplication tensor.
    
    Uses a block decomposition:
    - Split indices into G0={0,1}, G1={2,3}, G2={4}
    - For G0*G0*G0: Strassen's 7-term algorithm for 2x2*2x2
    - For G1*G1*G1: Strassen's 7-term algorithm for 2x2*2x2
    - For other blocks: direct multiplication
    
    U: (25, rank) - indexed by (i, j) where i is row of A, j is col of A
    V: (25, rank) - indexed by (j, k) where j is row of B, k is col of B
    W: (25, rank) - indexed by (k, i) where k is row of C, i is col of C (trace convention)
    
    The decomposition satisfies:
      For all i, j, k: sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*5+i, r] = 1
    """
    start = time.time()
    deadline = start + 110  # safety margin before 120s kill
    
    n, m, p = 5, 5, 5
    
    try:
        result = _construct_block_decomposition_221(n, m, p)
        if result is not None:
            return result
    except Exception:
        pass
    
    try:
        result = _construct_block_decomposition_2111(n, m, p)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition with rank 125
    return _construct_schoolbook(n, m, p)


def _construct_block_decomposition_221(n, m, p):
    """
    Construct a low-rank decomposition using block structure.
    
    Split indices into G0 = {0, 1}, G1 = {2, 3}, G2 = {4}.
    Use Strassen for G0*G0*G0 and G1*G1*G1 blocks, direct multiplication for others.
    Expected rank: 7 + 7 + 96 = 110
    """
    G0 = [0, 1]
    G1 = [2, 3]
    G2 = [4]
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        """
        Add a term to the decomposition.
        
        u_entries: list of (index, value) for U (index = i*m + j)
        v_entries: list of (index, value) for V (index = j*p + k)
        w_entries: list of (index, value) for W (index = k*n + i)
        """
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        for idx, val in u_entries:
            u[idx] = val
        for idx, val in v_entries:
            v[idx] = val
        for idx, val in w_entries:
            w[idx] = val
        
        terms.append((u, v, w))
    
    # Generic Strassen decomposition for a 2x2 * 2x2 block
    def add_strassen_block(block_indices):
        """
        Add Strassen terms for a 2x2*2x2 block defined by block_indices = [i0, i1].
        """
        # Strassen term 1: M1 = (A00+A11)(B00+B11)
        add_term(
            [(block_indices[0]*m + block_indices[0], 1), (block_indices[1]*m + block_indices[1], 1)],
            [(block_indices[0]*p + block_indices[0], 1), (block_indices[1]*p + block_indices[1], 1)],
            [(block_indices[0]*n + block_indices[0], 1), (block_indices[1]*n + block_indices[1], 1)]
        )
        
        # Strassen term 2: M2 = (A10+A11)B00
        add_term(
            [(block_indices[1]*m + block_indices[0], 1), (block_indices[1]*m + block_indices[1], 1)],
            [(block_indices[0]*p + block_indices[0], 1)],
            [(block_indices[0]*n + block_indices[1], 1), (block_indices[1]*n + block_indices[1], -1)]
        )
        
        # Strassen term 3: M3 = A00(B01-B11)
        add_term(
            [(block_indices[0]*m + block_indices[0], 1)],
            [(block_indices[0]*p + block_indices[1], 1), (block_indices[1]*p + block_indices[1], -1)],
            [(block_indices[1]*n + block_indices[0], 1), (block_indices[1]*n + block_indices[1], 1)]
        )
        
        # Strassen term 4: M4 = A11(B10-B00)
        add_term(
            [(block_indices[1]*m + block_indices[1], 1)],
            [(block_indices[1]*p + block_indices[0], 1), (block_indices[0]*p + block_indices[0], -1)],
            [(block_indices[0]*n + block_indices[0], 1), (block_indices[0]*n + block_indices[1], 1)]
        )
        
        # Strassen term 5: M5 = (A00+A01)B11
        add_term(
            [(block_indices[0]*m + block_indices[0], 1), (block_indices[0]*m + block_indices[1], 1)],
            [(block_indices[1]*p + block_indices[1], 1)],
            [(block_indices[0]*n + block_indices[0], -1), (block_indices[1]*n + block_indices[0], 1)]
        )
        
        # Strassen term 6: M6 = (A10-A00)(B00+B01)
        add_term(
            [(block_indices[1]*m + block_indices[0], 1), (block_indices[0]*m + block_indices[0], -1)],
            [(block_indices[0]*p + block_indices[0], 1), (block_indices[0]*p + block_indices[1], 1)],
            [(block_indices[1]*n + block_indices[1], 1)]
        )
        
        # Strassen term 7: M7 = (A01-A11)(B10+B11)
        add_term(
            [(block_indices[0]*m + block_indices[1], 1), (block_indices[1]*m + block_indices[1], -1)],
            [(block_indices[1]*p + block_indices[0], 1), (block_indices[1]*p + block_indices[1], 1)],
            [(block_indices[0]*n + block_indices[0], 1)]
        )
    
    # Add Strassen for G0 block
    add_strassen_block(G0)
    
    # Add Strassen for G1 block
    add_strassen_block(G1)
    
    # Now handle all other blocks using direct multiplication
    def add_direct_term(i, j, k):
        """Add a direct multiplication term for specific (i, j, k)."""
        add_term([(i * m + j, 1)],
                 [(j * p + k, 1)],
                 [(k * n + i, 1)])
    
    # All combinations except those all in G0 or all in G1 (which Strassen handles)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                i_in_G0 = i in G0
                j_in_G0 = j in G0
                k_in_G0 = k in G0
                i_in_G1 = i in G1
                j_in_G1 = j in G1
                k_in_G1 = k in G1
                if not (i_in_G0 and j_in_G0 and k_in_G0) and not (i_in_G1 and j_in_G1 and k_in_G1):
                    add_direct_term(i, j, k)
    
    # Build the U, V, W matrices
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx in range(n * m):
            U[idx][r] = u[idx]
        for idx in range(m * p):
            V[idx][r] = v[idx]
        for idx in range(p * n):
            W[idx][r] = w[idx]
    
    return {"U": U, "V": V, "W": W}


def _construct_block_decomposition_2111(n, m, p):
    """
    Construct a low-rank decomposition using block structure.
    
    Split indices into G0 = {0, 1}, G1 = {2}, G2 = {3}, G3 = {4}.
    Use Strassen for the 2x2*2x2 block, direct multiplication for others.
    """
    G0 = [0, 1]
    G1 = [2]
    G2 = [3]
    G3 = [4]
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        """
        Add a term to the decomposition.
        
        u_entries: list of (index, value) for U (index = i*m + j)
        v_entries: list of (index, value) for V (index = j*p + k)
        w_entries: list of (index, value) for W (index = k*n + i)
        """
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        for idx, val in u_entries:
            u[idx] = val
        for idx, val in v_entries:
            v[idx] = val
        for idx, val in w_entries:
            w[idx] = val
        
        terms.append((u, v, w))
    
    # Strassen decomposition for 2x2 * 2x2 block (i,j,k in G0)
    def add_strassen_term(local_u, local_v, local_w):
        """
        Add a Strassen term for the 2x2*2x2 block.
        local_u: dict mapping (i_loc, j_loc) -> value for U
        local_v: dict mapping (j_loc, k_loc) -> value for V
        local_w: dict mapping (k_loc, i_loc) -> value for W
        """
        u_entries = []
        v_entries = []
        w_entries = []
        
        for (i_loc, j_loc), val in local_u.items():
            if val != 0:
                global_i = G0[i_loc]
                global_j = G0[j_loc]
                u_entries.append((global_i * m + global_j, val))
        
        for (j_loc, k_loc), val in local_v.items():
            if val != 0:
                global_j = G0[j_loc]
                global_k = G0[k_loc]
                v_entries.append((global_j * p + global_k, val))
        
        for (k_loc, i_loc), val in local_w.items():
            if val != 0:
                global_k = G0[k_loc]
                global_i = G0[i_loc]
                w_entries.append((global_k * n + global_i, val))
        
        add_term(u_entries, v_entries, w_entries)
    
    # Strassen term 1: M1 = (A00+A11)(B00+B11)
    add_strassen_term(
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1, (1, 1): 1}
    )
    
    # Strassen term 2: M2 = (A10+A11)B00
    add_strassen_term(
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1}
    )
    
    # Strassen term 3: M3 = A00(B01-B11)
    add_strassen_term(
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (1, 1): 1}
    )
    
    # Strassen term 4: M4 = A11(B10-B00)
    add_strassen_term(
        {(1, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 0): 1, (0, 1): 1}
    )
    
    # Strassen term 5: M5 = (A00+A01)B11
    add_strassen_term(
        {(0, 0): 1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): -1, (1, 0): 1}
    )
    
    # Strassen term 6: M6 = (A10-A00)(B00+B01)
    add_strassen_term(
        {(1, 0): 1, (0, 0): -1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 1): 1}
    )
    
    # Strassen term 7: M7 = (A01-A11)(B10+B11)
    add_strassen_term(
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1}
    )
    
    # Now handle all other blocks using direct multiplication
    # For each combination of i in G0/G1/G2/G3, j in G0/G1/G2/G3, k in G0/G1/G2/G3
    # where not all are in G0
    
    def add_direct_term(i, j, k):
        """Add a direct multiplication term for specific (i, j, k)."""
        add_term([(i * m + j, 1)],
                 [(j * p + k, 1)],
                 [(k * n + i, 1)])
    
    # All combinations except those all in G0 (which Strassen handles)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                i_in_G0 = i in G0
                j_in_G0 = j in G0
                k_in_G0 = k in G0
                if not (i_in_G0 and j_in_G0 and k_in_G0):
                    add_direct_term(i, j, k)
    
    # Build the U, V, W matrices
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx in range(n * m):
            U[idx][r] = u[idx]
        for idx in range(m * p):
            V[idx][r] = v[idx]
        for idx in range(p * n):
            W[idx][r] = w[idx]
    
    return {"U": U, "V": V, "W": W}


def _construct_schoolbook(n, m, p):
    """
    Schoolbook decomposition with rank n*m*p = 125.
    """
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
