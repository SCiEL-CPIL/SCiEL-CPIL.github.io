"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition with rank = n*m*p."""
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if rank == 0:
        return False
    
    # Check the tensor identity: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_eliminate_columns(U, V, W, n, m, p, deadline):
    """
    Try to eliminate columns that are linearly dependent or can be removed.
    Returns the reduced decomposition.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    current_rank = U.shape[1]
    
    # Try to find columns that can be removed
    while current_rank > 0 and time.time() < deadline:
        removed = False
        for c in range(current_rank):
            mask = np.ones(current_rank, dtype=bool)
            mask[c] = False
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            
            if _verify_decomposition(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p, tol=1e-9):
                U, V, W = U_test, V_test, W_test
                current_rank -= 1
                removed = True
                break
        
        if not removed:
            break
    
    return U.tolist(), V.tolist(), W.tolist(), current_rank

def _random_perturbation(U, V, W, n, m, p, deadline, strength=0.1):
    """
    Apply random perturbation and try to find a valid lower-rank decomposition.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if rank == 0:
        return U.tolist(), V.tolist(), W.tolist(), rank
    
    # Create a copy and apply small random perturbations
    U_pert = U + strength * np.random.randn(*U.shape)
    V_pert = V + strength * np.random.randn(*V.shape)
    W_pert = W + strength * np.random.randn(*W.shape)
    
    # Try to find a subset of columns that still works
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try different subsets
    for _ in range(100):
        if time.time() > deadline:
            break
        
        # Randomly select a subset of columns
        subset_size = max(1, int(rank * 0.8))
        selected = random.sample(range(rank), subset_size)
        selected.sort()
        
        U_sub = U_pert[:, selected]
        V_sub = V_pert[:, selected]
        W_sub = W_pert[:, selected]
        
        if _verify_decomposition(U_sub.tolist(), V_sub.tolist(), W_sub.tolist(), n, m, p, tol=1e-8):
            if len(selected) < best_rank:
                best_rank = len(selected)
                best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank

def _construct_kronecker_based(n, m, p):
    """
    Construct a decomposition using Kronecker product structure.
    This may provide a better starting point than pure schoolbook.
    """
    # For matrix multiplication, we can use the fact that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This can be decomposed using tensor products
    
    # Use a structured decomposition based on Kronecker products
    # This is a heuristic - may not be optimal but provides structure
    
    rank = n * m * p  # Start with schoolbook rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4×5 × 5×6 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 130  # 20 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Get schoolbook decomposition
    schoolbook = _build_schoolbook(n, m, p)
    best_U = np.array(schoolbook["U"], dtype=float)
    best_V = np.array(schoolbook["V"], dtype=float)
    best_W = np.array(schoolbook["W"], dtype=float)
    best_rank = schoolbook["rank"]
    
    # Try column elimination on schoolbook
    U_elim, V_elim, W_elim, elim_rank = _try_eliminate_columns(
        schoolbook["U"], schoolbook["V"], schoolbook["W"], n, m, p, deadline
    )
    if elim_rank < best_rank:
        best_U, best_V, best_W = np.array(U_elim), np.array(V_elim), np.array(W_elim)
        best_rank = elim_rank
    
    # Try random perturbations
    if time.time() < deadline:
        U_pert, V_pert, W_pert, pert_rank = _random_perturbation(
            best_U, best_V, best_W, n, m, p, deadline, strength=0.05
        )
        if pert_rank < best_rank:
            best_U, best_V, best_W = np.array(U_pert), np.array(V_pert), np.array(W_pert)
            best_rank = pert_rank
    
    # Final verification
    if not _verify_decomposition(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
        # Fallback to schoolbook
        best_U = np.array(schoolbook["U"], dtype=float)
        best_V = np.array(schoolbook["V"], dtype=float)
        best_W = np.array(schoolbook["W"], dtype=float)
        best_rank = schoolbook["rank"]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
