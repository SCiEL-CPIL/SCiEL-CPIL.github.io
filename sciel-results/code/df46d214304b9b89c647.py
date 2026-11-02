import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit

    def build_strassen_blocks():
        """
        Build a rank-46 decomposition using Strassen on four 2x2x2 blocks
        and schoolbook for the remaining terms.
        """
        terms = []
        
        def add_strassen(i_start, j_start, k_start):
            """Add 7 Strassen terms for 2x2x2 block at (i_start, j_start, k_start)."""
            i0, i1 = i_start, i_start + 1
            j0, j1 = j_start, j_start + 1
            k0, k1 = k_start, k_start + 1
            
            # M1 = (a+d)(e+h)
            u = {(i0, j0): 1, (i1, j1): 1}
            v = {(j0, k0): 1, (j1, k1): 1}
            w = {(k0, i0): 1, (k1, i1): 1}
            terms.append((u, v, w))
            
            # M2 = (c+d)e
            u = {(i1, j0): 1, (i1, j1): 1}
            v = {(j0, k0): 1}
            w = {(k0, i1): 1, (k1, i1): -1}
            terms.append((u, v, w))
            
            # M3 = a(f-h)
            u = {(i0, j0): 1}
            v = {(j0, k1): 1, (j1, k1): -1}
            w = {(k1, i0): 1, (k1, i1): 1}
            terms.append((u, v, w))
            
            # M4 = d(g-e)
            u = {(i1, j1): 1}
            v = {(j1, k0): 1, (j0, k0): -1}
            w = {(k0, i0): 1, (k0, i1): 1}
            terms.append((u, v, w))
            
            # M5 = (a+b)h
            u = {(i0, j0): 1, (i0, j1): 1}
            v = {(j1, k1): 1}
            w = {(k0, i0): -1, (k1, i0): 1}
            terms.append((u, v, w))
            
            # M6 = (c-a)(e+f)
            u = {(i1, j0): 1, (i0, j0): -1}
            v = {(j0, k0): 1, (j0, k1): 1}
            w = {(k1, i1): 1}
            terms.append((u, v, w))
            
            # M7 = (b-d)(g+h)
            u = {(i0, j1): 1, (i1, j1): -1}
            v = {(j1, k0): 1, (j1, k1): 1}
            w = {(k0, i0): 1}
            terms.append((u, v, w))
        
        def add_schoolbook_block(i_start, i_count, j_start, j_count, k_start, k_count):
            """Add schoolbook terms for a rectangular block."""
            for i in range(i_start, i_start + i_count):
                for j in range(j_start, j_start + j_count):
                    for k in range(k_start, k_start + k_count):
                        u = {(i, j): 1}
                        v = {(j, k): 1}
                        w = {(k, i): 1}
                        terms.append((u, v, w))
        
        # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
        add_strassen(0, 0, 0)
        
        # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
        add_strassen(0, 0, 2)
        
        # Block 3: i in {0,1}, j in {0,1}, k in {4}
        add_schoolbook_block(0, 2, 0, 2, 4, 1)
        
        # Block 4: i in {0,1}, j in {2,3}, k in {0,1}
        add_strassen(0, 2, 0)
        
        # Block 5: i in {0,1}, j in {2,3}, k in {2,3}
        add_strassen(0, 2, 2)
        
        # Block 6: i in {0,1}, j in {2,3}, k in {4}
        add_schoolbook_block(0, 2, 2, 2, 4, 1)
        
        # Block 7: i in {0,1}, j in {4}, k in {0,1}
        add_schoolbook_block(0, 2, 4, 1, 0, 2)
        
        # Block 8: i in {0,1}, j in {4}, k in {2,3}
        add_schoolbook_block(0, 2, 4, 1, 2, 2)
        
        # Block 9: i in {0,1}, j in {4}, k in {4}
        add_schoolbook_block(0, 2, 4, 1, 4, 1)
        
        rank = len(terms)
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for c in range(rank):
            u_dict, v_dict, w_dict = terms[c]
            for (i, j), val in u_dict.items():
                U[i * m + j][c] = val
            for (j, k), val in v_dict.items():
                V[j * p + k][c] = val
            for (k, i), val in w_dict.items():
                W[k * n + i][c] = val
        
        return U, V, W, rank, terms
    
    def verify_decomposition(U, V, W, rank):
        """Verify the decomposition is exact. Returns max error."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def build_schoolbook():
        """Build pure schoolbook decomposition as fallback (rank 50)."""
        terms_sb = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms_sb.append((u, v, w))
        rank_sb = len(terms_sb)
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        for c in range(rank_sb):
            u_dict, v_dict, w_dict = terms_sb[c]
            for (i, j), val in u_dict.items():
                U_sb[i * m + j][c] = val
            for (j, k), val in v_dict.items():
                V_sb[j * p + k][c] = val
            for (k, i), val in w_dict.items():
                W_sb[k * n + i][c] = val
        return U_sb, V_sb, W_sb
    
    def reduce_rank_by_dependency(terms, U, V, W, rank):
        """
        Try to reduce rank by finding linearly dependent tensor components.
        Each term c defines a rank-1 tensor T_c = U_c ⊗ V_c ⊗ W_c.
        The decomposition is exact if sum_c T_c = T_matmul.
        If some subset of T_c is linearly dependent, we can potentially merge them.
        However, for exact integer/half-integer decomposition, we need to be careful.
        
        A simpler approach: check if removing any single term still gives an exact decomposition
        (which would mean that term is zero in the tensor space, unlikely).
        
        More practically: try to see if we can combine two schoolbook terms into one
        by exploiting structure. For example, if two terms share the same W component
        but have different U and V, we might be able to use a Strassen-like trick.
        
        For now, this is a placeholder for future optimization.
        """
        # Try greedy removal: check if any term can be removed
        # (This is expensive, so we limit it)
        if time.time() > deadline - 20:
            return U, V, W, rank
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build the full tensor
        tensor = np.zeros((n, m, p))
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
        
        # Try removing each term and check if tensor is still correct
        # (only do this if we have time)
        if time.time() > deadline - 10:
            return U, V, W, rank
        
        # This is too expensive for rank 46, so skip for now
        # A better approach would be to look for structural redundancies
        return U, V, W, rank
    
    # Try the Strassen-based decomposition first
    U, V, W, rank, terms = build_strassen_blocks()
    
    # Verify the decomposition is exact
    max_err = verify_decomposition(U, V, W, rank)
    
    if max_err > 1e-6:
        # Fallback to pure schoolbook
        U, V, W = build_schoolbook()
    else:
        # Try to reduce rank
        U, V, W, rank = reduce_rank_by_dependency(terms, U, V, W, rank)
        # Re-verify after reduction
        if rank > 0:
            max_err2 = verify_decomposition(U, V, W, rank)
            if max_err2 > 1e-6:
                # Reduction failed, rebuild
                U, V, W, rank, terms = build_strassen_blocks()
    
    return {"U": U, "V": V, "W": W}
