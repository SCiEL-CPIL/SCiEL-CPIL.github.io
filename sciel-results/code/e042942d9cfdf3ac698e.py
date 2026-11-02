"""Matrix multiplication 3x4 by 4x6: Strassen-based decomposition with local rank reduction search."""
import time
import random
import itertools

DIMENSIONS = [3, 4, 6]

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    
    Strategy:
    - Row 0 (1x4x6): schoolbook, 24 terms (fixed)
    - Rows 1-2 (2x4x6): Start with Strassen on 2x2x2 sub-blocks (6 blocks * 7 = 42 terms)
    - Then perform local search to reduce rank by finding linear dependencies
      among the Strassen terms and eliminating redundant ones.
    
    The local search tries to find a subset of columns that still spans the
    required tensor space, effectively reducing the rank.
    """
    n, m, p = 3, 4, 6
    base_rank = 66  # 24 schoolbook + 42 Strassen
    
    # Build the base decomposition
    U = [[0] * base_rank for _ in range(n * m)]
    V = [[0] * base_rank for _ in range(m * p)]
    W = [[0] * base_rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1

    # Part 1: Row 0 schoolbook
    for j in range(4):
        for k in range(6):
            add_schoolbook(0, j, k)
    
    # Part 2: Rows 1-2 Strassen
    for k_block in range(3):
        k_base = k_block * 2
        for j_block in range(2):
            j_base = j_block * 2
            i_base = 1
            add_strassen(i_base, j_base, k_base)
    
    # Now try to reduce rank by finding linear dependencies
    # The tensor is T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # We need to verify that the decomposition is correct, then try to reduce rank.
    
    # First, verify correctness by checking the tensor
    # T[i,j,k] should equal sum over l of A[i,l]*B[l,k]
    # In our decomposition: T[i,j,k] = sum_r U[i*4+j, r] * V[j*6+k, r] * W[k*3+i, r]
    
    # Build the tensor from the decomposition
    def build_tensor(U, V, W, rank):
        """Build the 3D tensor T[i,j,k] from the factor matrices."""
        T = [[[0]*p for _ in range(m)] for _ in range(n)]
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    u_val = U[i*m+j][r]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V[j*p+k][r]
                        if v_val == 0:
                            continue
                        w_val = W[k*n+i][r]
                        if w_val == 0:
                            continue
                        T[i][j][k] += u_val * v_val * w_val
        return T
    
    # Check if current decomposition is correct
    T_decomp = build_tensor(U, V, W, base_rank)
    
    # The correct tensor for matrix mult: T[i,j,k] = delta_{i==j? no...}
    # Actually T[i,j,k] = sum_l A[i,l] * B[l,k]
    # But we're decomposing the multiplication tensor, so:
    # T[i,j,k] = 1 if this is a valid multiplication term, 0 otherwise
    # More precisely: the multiplication tensor M has M[i,j,k] = 1 if we can get C[i,k] from A[i,j]*B[j,k]
    # Wait, let me reconsider. The tensor for matrix multiplication (n x m) * (m x p) -> (n x p)
    # is indexed as T[i,k,j] where i is row of A/C, k is col of B/C, j is the shared index.
    # But the task says W[k*n+i, r], so W is indexed by (k, i) -> k*n+i
    # And U[i*m+j, r], V[j*p+k, r]
    # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal the entry of C[i,k] contributed by A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) since C[i,k] = sum_j A[i,j]*B[j,k]
    # Wait no - the tensor decomposition represents the bilinear map.
    # The multiplication tensor has entries: for each (i,j,k), the coefficient is 1
    # because C[i,k] gets contribution A[i,j]*B[j,k].
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # Let's verify
    correct = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T_decomp[i][j][k] != 1:
                    correct = False
                    break
            if not correct:
                break
        if not correct:
            break
    
    if not correct:
        # Return the base decomposition if it's wrong (shouldn't happen)
        return {"U": U, V: V, W: W}
    
    # Now try to reduce rank by finding linear dependencies among columns
    # A column r is redundant if it can be expressed as a linear combination of others
    # This is equivalent to finding a smaller set of columns that still spans the same tensor
    
    # Use a greedy approach: try removing each column and check if the tensor is preserved
    # This is expensive, so we'll use a smarter approach.
    
    # Alternative: use Gaussian elimination on the "flattened" tensor
    # The tensor T can be viewed as a matrix of size (n*m) x (m*p) with entries...
    # Actually, let's think differently.
    
    # The key insight: if two columns are linearly dependent in a certain way,
    # we can combine them. But for exact integer decompositions, we need to be careful.
    
    # Let's try a different approach: search for a rank-r decomposition where r < 66
    # by trying to find a subset of columns that still gives the correct tensor.
    
    # Start with all 66 columns, try to remove one at a time.
    # For each column r, check if removing it still gives T[i,j,k] = 1 for all (i,j,k).
    
    active_cols = list(range(base_rank))
    current_rank = base_rank
    
    # Try to remove columns one by one
    for _ in range(20):  # Try up to 20 removals
        if _time_left() < 10:
            break
        
        removed_any = False
        for r in list(active_cols):
            if _time_left() < 10:
                break
            
            # Try removing column r
            test_active = [c for c in active_cols if c != r]
            
            # Check if the tensor is still correct
            T_test = build_tensor(U, V, W, len(test_active))
            # But we need to use only the active columns
            T_test = [[[0]*p for _ in range(m)] for _ in range(n)]
            for idx, r_idx in enumerate(test_active):
                for i in range(n):
                    for j in range(m):
                        u_val = U[i*m+j][r_idx]
                        if u_val == 0:
                            continue
                        for k in range(p):
                            v_val = V[j*p+k][r_idx]
                            if v_val == 0:
                                continue
                            w_val = W[k*n+i][r_idx]
                            if w_val == 0:
                                continue
                            T_test[i][j][k] += u_val * v_val * w_val
            
            # Check correctness
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if T_test[i][j][k] != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                active_cols.remove(r)
                current_rank -= 1
                removed_any = True
                break  # Restart the loop after removing one column
        
        if not removed_any:
            break
    
    # Build the final decomposition with only active columns
    final_U = [[0] * current_rank for _ in range(n * m)]
    final_V = [[0] * current_rank for _ in range(m * p)]
    final_W = [[0] * current_rank for _ in range(p * n)]
    
    for new_idx, old_idx in enumerate(active_cols):
        for i in range(n * m):
            final_U[i][new_idx] = U[i][old_idx]
        for j in range(m * p):
            final_V[j][new_idx] = V[j][old_idx]
        for k in range(p * n):
            final_W[k][new_idx] = W[k][old_idx]
    
    return {"U": final_U, "V": final_V, "W": final_W}
