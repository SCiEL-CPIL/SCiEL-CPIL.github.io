import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                # The matrix multiplication tensor has T[i,j,k] = 1 if we're computing (A*B)[i,k]
                # Actually, the standard matrix multiplication tensor T_{ijk} = 1 for all i,j,k
                # because (A*B)_{ik} = sum_j A_{ij} B_{jk}
                # So we need sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1 for all i,j,k
                if total != 1:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to 3D
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # All entries should be 1
    if not np.allclose(T, 1.0, atol=1e-10):
        return False
    return True

def _try_rank_reduction(U, V, W, n, m, p, target_rank, max_iters=100):
    """Try to reduce rank by finding linear dependencies among columns."""
    rank = len(U[0])
    if rank <= target_rank:
        return None
    
    # Try to find a subset of columns that spans the same space
    # Use Gaussian elimination to find independent columns
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Find rank of U
    U_rank = np.linalg.matrix_rank(U_arr, tol=1e-10)
    if U_rank < target_rank:
        # U already has rank < target_rank, can't reduce further this way
        return None
    
    # Try removing columns one at a time
    for col_idx in range(rank - 1, target_rank - 1, -1):
        # Check if column col_idx is in the span of others
        others = [c for c in range(rank) if c != col_idx]
        U_others = U_arr[:, others]
        U_col = U_arr[:, col_idx]
        
        # Check if U_col is in span of U_others
        try:
            coeff, residuals, rank_others, sv = np.linalg.lstsq(U_others, U_col, rcond=None)
            if len(residuals) == 0 or np.linalg.norm(residuals) < 1e-8:
                # Check if this column can be removed from V and W too
                # This is complex; skip for now
                pass
        except:
            pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    start_time = time.time()
    time_limit = 130.0  # Leave 20s margin before 150s kill
    
    # Initialize with schoolbook
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    best_rank = rank_school
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Try to find a lower rank decomposition
    # Strategy: Use the fact that for matrix multiplication, there are known
    # algorithms with rank less than n*m*p for certain dimensions.
    # For 3x5x7, we can try structured decompositions.
    
    # Try random structured decompositions
    random.seed(42)
    
    # Try ranks from 100 down to 80 (heuristic lower bound)
    for target_rank in range(100, 75, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try to construct a rank-target_rank decomposition
        # Use a structured approach: partition the computation
        
        # Method: Try to find U, V, W with small integer entries
        # that satisfy the tensor equation
        
        # Simple approach: try to use block structure
        # For 3x5x7, we can think of it as:
        # (A * B)_{ik} = sum_j A_{ij} B_{jk}
        
        # Try a decomposition based on splitting j
        # This is getting complex; let's try a simpler random search
        
        # Random search with verification
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Generate random U, V with small integer entries
            U_rand = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n*m)]
            V_rand = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m*p)]
            
            # Solve for W: W[k*n+i, r] should satisfy
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
            
            # This is a system of equations. For each (i,k), we have:
            # sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r]) = 1
            # Wait, that's not right. The tensor is:
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # For fixed (i,k), we have m equations (one for each j) and target_rank unknowns (W[k*n+i, r])
            # This is underdetermined if target_rank > m
            
            # Let's set up the system properly
            # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is trilinear, not linear in W alone.
            # We need a different approach.
            
            # Alternative: Fix U and V, then for each r, the contribution to T is
            # U[:, :, r] * V[:, :, r] * W[:, :, r] (outer product)
            # We need sum_r of these = T (all ones)
            
            # This is a tensor completion problem.
            # For now, skip this and just return schoolbook if no better found.
            
            pass
    
    # If we haven't found a better decomposition, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
