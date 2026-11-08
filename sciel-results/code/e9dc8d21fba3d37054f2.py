import numpy as np
import time
from itertools import combinations, product

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    total_terms = n * m * p  # 75
    
    # We'll try to find a rank-minimal decomposition.
    # ... (comments omitted for brevity in thought process, kept in code) ...
    
    # Let's enumerate possible block decompositions and compute their ranks.
    # ... (comments omitted) ...
    
    # Current solution uses 4 blocks: (0,0,0), (0,2,0), (0,0,2), (0,2,2)
    # ... (comments omitted) ...
    
    # Can we do better? Let's try other placements.
    # ... (comments omitted) ...
    
    # But wait - we can also use non-2x2x2 Strassen-like decompositions.
    # ... (comments omitted) ...
    
    # For a 3x2x2 block, we can think of it as 3 separate 1x2x2 blocks.
    # ... (comments omitted) ...
    
    # What about a 2x3x2 block?
    # ... (comments omitted) ...
    
    # So 2x2x2 is the smallest block where Strassen gives a saving.
    
    # Alternative: use a different algebraic identity for a 2x2x2 block.
    # The standard Strassen 7-multiplication is optimal for 2x2x2.
    
    # Let me try a completely different approach: use a random search / optimization
    # to find a lower-rank decomposition.
    
    # Actually, let me check: what's the theoretical lower bound?
    # ... (comments omitted) ...
    
    # The correct lower bound: rank >= (nm + mp + pn)/... 
    # ... (comments omitted) ...
    
    # Given the 120s time limit, let me try a randomized search for a better decomposition.
    # Approach: start from the rank-71 solution and try to merge columns.
    
    # Build the rank-71 solution first.
    def build_base_solution():
        blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
        total_strassen_rank = len(blocks) * 7
        strassen_terms_count = len(blocks) * 8
        schoolbook_terms_count = total_terms - strassen_terms_count
        total_rank = total_strassen_rank + schoolbook_terms_count
        
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        col = 0
        
        def add_strassen_block(i0, j0, k0, col_offset):
            c = col_offset + 0
            U[i0*m + j0][c] += 1
            U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            c = col_offset + 1
            U[(i0+1)*m + j0][c] += 1
            U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1
            W[k0*n + (i0+1)][c] += 1
            W[(k0+1)*n + (i0+1)][c] += -1
            
            c = col_offset + 2
            U[i0*m + j0][c] += 1
            V[j0*p + (k0+1)][c] += 1
            V[(j0+1)*p + (k0+1)][c] += -1
            W[(k0+1)*n + i0][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            c = col_offset + 3
            U[(i0+1)*m + (j0+1)][c] += 1
            V[(j0+1)*p + k0][c] += 1
            V[j0*p + k0][c] += -1
            W[k0*n + i0][c] += 1
            W[k0*n + (i0+1)][c] += 1
            
            c = col_offset + 4
            U[i0*m + j0][c] += 1
            U[i0*m + (j0+1)][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += -1
            W[(k0+1)*n + i0][c] += 1
            
            c = col_offset + 5
            U[(i0+1)*m + j0][c] += 1
            U[i0*m + j0][c] += -1
            V[j0*p + k0][c] += 1
            V[j0*p + (k0+1)][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            c = col_offset + 6
            U[i0*m + (j0+1)][c] += 1
            U[(i0+1)*m + (j0+1)][c] += -1
            V[(j0+1)*p + k0][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1
        
        for i0, j0, k0 in blocks:
            add_strassen_block(i0, j0, k0, col)
            col += 7
        
        covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered_3d[i0+di][j0+dj][k0+dk] = True
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered_3d[i][j][k]:
                        U[i*m+j][col] += 1
                        V[j*p+k][col] += 1
                        W[k*n+i][col] += 1
                        col += 1
                        if time.time() > deadline:
                            break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        return U, V, W, col
    
    U, V, W, rank = build_base_solution()
    
    if time.time() > deadline:
        return {"U": np.array(U), "V": np.array(V), "W": np.array(W)}

    # Convert to numpy for efficient operations and to fix list slicing errors
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Verify the decomposition.
    # ... (comments omitted) ...
    
    def is_valid(U_arr, V_arr, W_arr):
        """Check if the decomposition is valid."""
        n, m, p = DIMENSIONS
        r = U_arr.shape[1]
        
        # Compute T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For this to be valid, T[i,j,k] should be 1 for all i,j,k.
        
        # Efficient check using broadcasting/einsum might be better, but simple loop is fine for small dims
        # Let's do a full check if fast enough, or random. 
        # 75 terms * rank (~70) is ~5000 ops. Very fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    best_rank = rank
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    
    # Try removing one column at a time.
    # We iterate through columns. If removing a column keeps validity, we keep it removed.
    # This is a greedy approach.
    current_rank = rank
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    
    # We need to check if removing column c is valid.
    # Instead of removing and re-checking from scratch (O(R * N*M*P)), we can just check.
    # But removing a column changes the indices.
    # Let's just try removing each column index from the current set.
    
    # To make this efficient, we can check if the column is "redundant" in a linear algebra sense?
    # No, tensor rank reduction is not just linear dependence of columns.
    # We must verify the tensor equality.
    
    # Greedy removal:
    # Loop while we can remove columns.
    # To avoid O(R^2 * CheckCost), we just do one pass of trying to remove each column.
    # If we remove one, we restart the pass or continue? 
    # Let's do a single pass where we try to remove each column from the *original* set? 
    # No, that might miss combinations. 
    # Let's do a loop: 
    # for c in range(current_rank):
    #    if is_valid(U without c, V without c, W without c):
    #        remove c
    #        restart loop? or continue?
    #        If we remove c, the indices shift. 
    #        It's safer to build a new set of indices.
    
    indices_to_keep = list(range(rank))
    
    for _ in range(rank): # Max rank removals
        if time.time() > deadline:
            break
        removed_any = False
        for c in indices_to_keep[:]: # Iterate over a copy
            if time.time() > deadline:
                break
            # Check if removing c is valid
            mask = [i for i in indices_to_keep if i != c]
            if len(mask) == 0:
                continue
            
            U_trial = U_np[:, mask]
            V_trial = V_np[:, mask]
            W_trial = W_np[:, mask]
            
            if is_valid(U_trial, V_trial, W_trial):
                indices_to_keep.remove(c)
                removed_any = True
                # Update current state to the new best found? 
                # Or just update indices?
                # We should update the "best" if this is smaller.
                if len(indices_to_keep) < best_rank:
                    best_rank = len(indices_to_keep)
                    best_U = U_np[:, indices_to_keep]
                    best_V = V_np[:, indices_to_keep]
                    best_W = W_np[:, indices_to_keep]
                # If we removed one, we should probably restart the inner loop or at least
                # ensure we don't check indices that are now out of bounds?
                # Since we use `indices_to_keep` list, `mask` is always valid relative to original U_np.
                # So we can continue the loop.
                break # Restart the inner loop to check against the new reduced set? 
                      # Actually, breaking the inner loop and restarting the outer loop is safer 
                      # to ensure we check all columns against the new reduced basis.
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}

if __name__ == "__main__":
    res = construct()
