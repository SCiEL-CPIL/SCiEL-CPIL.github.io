"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    def validate(U, V, W):
        """Check if the decomposition is exact using vectorized numpy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=1e-9)
    
    def build_solution(blocks, schoolbook_terms):
        """Build U, V, W from Strassen blocks and schoolbook terms."""
        total_rank = len(blocks) * 7 + len(schoolbook_terms)
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        col = 0
        for (i0, j0, k0) in blocks:
            # M1
            c = col
            U[i0*m + j0][c] += 1; U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1; V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1; W[(k0+1)*n + (i0+1)][c] += 1
            col += 1
            # M2
            c = col
            U[(i0+1)*m + j0][c] += 1; U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1
            W[k0*n + (i0+1)][c] += 1; W[(k0+1)*n + (i0+1)][c] += -1
            col += 1
            # M3
            c = col
            U[i0*m + j0][c] += 1
            V[j0*p + (k0+1)][c] += 1; V[(j0+1)*p + (k0+1)][c] += -1
            W[(k0+1)*n + i0][c] += 1; W[(k0+1)*n + (i0+1)][c] += 1
            col += 1
            # M4
            c = col
            U[(i0+1)*m + (j0+1)][c] += 1
            V[(j0+1)*p + k0][c] += 1; V[j0*p + k0][c] += -1
            W[k0*n + i0][c] += 1; W[k0*n + (i0+1)][c] += 1
            col += 1
            # M5
            c = col
            U[i0*m + j0][c] += 1; U[i0*m + (j0+1)][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += -1; W[(k0+1)*n + i0][c] += 1
            col += 1
            # M6
            c = col
            U[(i0+1)*m + j0][c] += 1; U[i0*m + j0][c] += -1
            V[j0*p + k0][c] += 1; V[j0*p + (k0+1)][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            col += 1
            # M7
            c = col
            U[i0*m + (j0+1)][c] += 1; U[(i0+1)*m + (j0+1)][c] += -1
            V[(j0+1)*p + k0][c] += 1; V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1
            col += 1
        
        for (i, j, k) in schoolbook_terms:
            c = col
            U[i*m + j][c] += 1
            V[j*p + k][c] += 1
            W[k*n + i][c] += 1
            col += 1
        
        return U, V, W
    
    def get_schoolbook_for_blocks(blocks):
        """Get schoolbook terms not covered by blocks."""
        covered = set()
        for (i0, j0, k0) in blocks:
            for i in (i0, i0+1):
                for j in (j0, j0+1):
                    for k in (k0, k0+1):
                        covered.add((i, j, k))
        return [(i, j, k) for i in range(n) for j in range(m) for k in range(p) if (i, j, k) not in covered]
    
    # Generate all possible Strassen block positions
    all_possible_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_possible_blocks.append((i0, j0, k0))
    
    # Build the initial best solution with 4 disjoint Strassen blocks
    best_rank = 75
    best_solution = None
    initial_blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    sb = get_schoolbook_for_blocks(initial_blocks)
    best_rank = len(initial_blocks) * 7 + len(sb)
    U, V, W = build_solution(initial_blocks, sb)
    if validate(U, V, W):
        best_solution = (U, V, W)
    
    # Try different block combinations - allow overlapping blocks
    from itertools import combinations
    for size in range(1, min(8, len(all_possible_blocks) + 1)):
        if time.time() > deadline - 30:
            break
        for combo in combinations(all_possible_blocks, size):
            # Calculate rank: each block adds 7, but overlapping blocks may share terms
            # For now, just count total terms
            sb = get_schoolbook_for_blocks(list(combo))
            rank = size * 7 + len(sb)
            if rank < best_rank:
                best_rank = rank
                U, V, W = build_solution(list(combo), sb)
                if validate(U, V, W):
                    best_solution = (U, V, W)
    
    # Now attempt rank reduction by checking for linear dependencies among columns
    if best_solution is not None and time.time() < deadline - 20:
        U_arr = np.array(best_solution[0], dtype=float)
        V_arr = np.array(best_solution[1], dtype=float)
        W_arr = np.array(best_solution[2], dtype=float)
        current_rank = U_arr.shape[1]
        
        # Try to find linear dependencies: for each column t, check if its tensor
        # contribution can be expressed as a linear combination of other columns
        n_terms = n * m * p  # total number of tensor entries
        
        # Create the tensor for each column
        def get_column_tensor(U_col, V_col, W_col):
            """Get the tensor contribution of a single column."""
            U3 = U_col.reshape(n, m)
            V3 = V_col.reshape(m, p)
            W3 = W_col.reshape(p, n)
            # The tensor entry (i,j,k) is U[i,j] * V[j,k] * W[k,i]
            tensor = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] = U3[i, j] * V3[j, k] * W3[k, i]
            return tensor.flatten()
        
        # Build the tensor matrix where each column is a tensor contribution
        tensor_cols = []
        for t in range(current_rank):
            col_tensor = get_column_tensor(
                U_arr[:, t], V_arr[:, t], W_arr[:, t]
            )
            tensor_cols.append(col_tensor)
        
        tensor_matrix = np.array(tensor_cols).T  # shape (n*m*p, rank)
        
        # Try to find columns that are linear combinations of others
        # Use rank-revealing decomposition
        try:
            # SVD to find rank
            U_svd, s, Vt_svd = np.linalg.svd(tensor_matrix, full_matrices=False)
            # Find numerical rank
            tol = 1e-10 * max(s) if len(s) > 0 else 1e-10
            num_rank = np.sum(s > tol)
            
            if num_rank < current_rank:
                # There are dependencies - try to remove redundant columns
                # Find which columns to keep (those with largest singular values)
                # For simplicity, just check if we can remove the last column
                # that is approximately in the span of others
                
                # Try removing each column and see if the remaining still span
                for t_to_remove in range(current_rank - 1, -1, -1):
                    if time.time() > deadline - 5:
                        break
                    
                    # Get the remaining columns
                    keep_cols = [c for c in range(current_rank) if c != t_to_remove]
                    if len(keep_cols) == 0:
                        continue
                    
                    # Check if column t_to_remove is in the span of others
                    remaining_matrix = tensor_matrix[:, keep_cols]
                    
                    # Try to solve for coefficients
                    try:
                        # Least squares: find alpha such that remaining_matrix @ alpha ≈ tensor_matrix[:, t_to_remove]
                        alpha, residuals, rank_lsq, sv = np.linalg.lstsq(
                            remaining_matrix, tensor_matrix[:, t_to_remove], rcond=1e-10
                        )
                        
                        # Check if the fit is good
                        fitted = remaining_matrix @ alpha
                        error = np.linalg.norm(tensor_matrix[:, t_to_remove] - fitted)
                        
                        if error < 1e-8:
                            # Column is redundant - remove it
                            new_U = U_arr[:, keep_cols]
                            new_V = V_arr[:, keep_cols]
                            new_W = W_arr[:, keep_cols]
                            
                            # Validate
                            U3 = new_U.reshape(n, m, -1)
                            V3 = new_V.reshape(m, p, -1)
                            W3 = new_W.reshape(p, n, -1)
                            T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
                            
                            if np.allclose(T, 1.0, atol=1e-9):
                                best_solution = (new_U.tolist(), new_V.tolist(), new_W.tolist())
                                best_rank = len(keep_cols)
                                U_arr = new_U
                                V_arr = new_V
                                W_arr = new_W
                                current_rank = len(keep_cols)
                                # Rebuild tensor matrix
                                tensor_cols = []
                                for t in range(current_rank):
                                    col_tensor = get_column_tensor(
                                        U_arr[:, t], V_arr[:, t], W_arr[:, t]
                                    )
                                    tensor_cols.append(col_tensor)
                                tensor_matrix = np.array(tensor_cols).T
                                break  # Restart the removal process
                    except np.linalg.LinAlgError:
                        continue
        except Exception:
            pass
    
    if best_solution is not None:
        return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
    
    # Fallback
    U, V, W = build_solution(initial_blocks, get_schoolbook_for_blocks(initial_blocks))
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
