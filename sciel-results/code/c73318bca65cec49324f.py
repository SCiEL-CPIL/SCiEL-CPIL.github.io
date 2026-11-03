"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact using the tensor check."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, rank):
    """Fast verification using numpy - vectorized over all (i,j,k) triples."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    # Build all required dot products at once
    # For each (i,j,k): need dot(U[i*m+j], V[j*p+k] * W[k*n+i])
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                errors.append(s - 1.0)
    errors = np.array(errors)
    return np.max(np.abs(errors)) < 1e-9

def _verify_with_tolerance(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify with explicit tolerance, returns max error."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _strassen_2x2x2(offset_u, offset_v, offset_w, U, V, W, col_start):
    """
    Add a Strassen 2x2x2 block at given offsets, starting at col_start.
    U rows: offset_u to offset_u+3 (i=0,1; j=0,1)
    V rows: offset_v to offset_v+3 (j=0,1; k=0,1)  
    W rows: offset_w to offset_w+3 (k=0,1; i=0,1)
    Returns the next column index after writing 7 columns.
    """
    c = col_start
    # M1 = (A00+A11)*(B00+B11)
    U[offset_u + 0][c] = 1
    U[offset_u + 3][c] = 1
    V[offset_v + 0][c] = 1
    V[offset_v + 3][c] = 1
    W[offset_w + 0][c] = 1
    W[offset_w + 3][c] = 1
    c += 1
    
    # M2 = (A10+A11)*B00
    U[offset_u + 2][c] = 1
    U[offset_u + 3][c] = 1
    V[offset_v + 0][c] = 1
    W[offset_w + 2][c] = 1
    W[offset_w + 3][c] = -1
    c += 1
    
    # M3 = A00*(B01-B11)
    U[offset_u + 0][c] = 1
    V[offset_v + 1][c] = 1
    V[offset_v + 3][c] = -1
    W[offset_w + 0][c] = 1
    W[offset_w + 3][c] = 1
    c += 1
    
    # M4 = A11*(B10-B00)
    U[offset_u + 3][c] = 1
    V[offset_v + 2][c] = 1
    V[offset_v + 0][c] = -1
    W[offset_w + 0][c] = 1
    W[offset_w + 2][c] = 1
    c += 1
    
    # M5 = (A00+A01)*B11
    U[offset_u + 0][c] = 1
    U[offset_u + 1][c] = 1
    V[offset_v + 3][c] = 1
    W[offset_w + 0][c] = -1
    W[offset_w + 1][c] = 1
    c += 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[offset_u + 2][c] = 1
    U[offset_u + 0][c] = -1
    V[offset_v + 0][c] = 1
    V[offset_v + 1][c] = 1
    W[offset_w + 3][c] = 1
    c += 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[offset_u + 1][c] = 1
    U[offset_u + 3][c] = -1
    V[offset_v + 2][c] = 1
    V[offset_v + 3][c] = 1
    W[offset_w + 0][c] = 1
    c += 1
    
    return c

def _construct_hybrid_rank36():
    """Construct the rank-36 hybrid decomposition."""
    n, m, p = 2, 4, 5
    total_rank = 36
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    c = 0
    # Block 1: A1 (U rows 0-3) * B1 (V rows 0-9)
    # Strassen for cols 0-1
    c = _strassen_2x2x2(0, 0, 0, U, V, W, c)
    # Schoolbook for col 2
    U[0][c] = 1; V[2][c] = 1; W[4][c] = 1; c += 1
    U[1][c] = 1; V[3][c] = 1; W[4][c] = 1; c += 1
    U[2][c] = 1; V[2][c] = 1; W[5][c] = 1; c += 1
    U[3][c] = 1; V[3][c] = 1; W[5][c] = 1; c += 1
    # Strassen for cols 3-4
    c = _strassen_2x2x2(0, 6, 6, U, V, W, c)
    
    # Block 2: A2 (U rows 4-7) * B2 (V rows 10-19)
    # Strassen for cols 0-1
    c = _strassen_2x2x2(4, 10, 0, U, V, W, c)
    # Schoolbook for col 2
    U[4][c] = 1; V[12][c] = 1; W[4][c] = 1; c += 1
    U[5][c] = 1; V[13][c] = 1; W[4][c] = 1; c += 1
    U[6][c] = 1; V[12][c] = 1; W[5][c] = 1; c += 1
    U[7][c] = 1; V[13][c] = 1; W[5][c] = 1; c += 1
    # Strassen for cols 3-4
    c = _strassen_2x2x2(4, 16, 6, U, V, W, c)
    
    return {"U": U, "V": V, "W": W}

def _construct_rank32():
    """
    Try to construct a rank-32 decomposition using Winograd-style
    factorization of the 2x4x5 tensor.
    
    The idea: split the 4 intermediate dimension into two groups of 2,
    and the 5 output dimension into (2,2,1) or (3,2) groups.
    
    Strategy: Use Strassen for 2x2 blocks where possible, and exploit
    the rectangular structure.
    """
    n, m, p = 2, 4, 5
    total_rank = 32
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Split m=4 into two groups of 2: j in {0,1} and j in {2,3}
    # Split p=5 into groups: k in {0,1}, k in {2,3}, k in {4}
    
    # For each (i_group, j_group, k_group) triple, we need to compute
    # C[i, k] += sum over j in j_group of A[i,j] * B[j,k]
    
    # Let's use a different approach: exploit that n=2
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    
    # For fixed k, this is a 2-vector output from a 4-dim inner product
    # We can use the fact that for each k, we need 2 outputs from 4 inputs
    
    # Actually, let's try a direct construction with rank 32
    # by being more efficient with the Strassen blocks.
    
    # Alternative: use rank-7 Strassen for the full 2x4x4 part (k=0..3)
    # and handle k=4 separately.
    
    # For k=0..3: 2x4x4 multiplication
    # Split j into {0,1} and {2,3}
    # For j in {0,1}: 2x2x4 -> use 2 Strassen blocks (each 2x2x2) = 14 rank
    # For j in {2,3}: 2x2x4 -> use 2 Strassen blocks (each 2x2x2) = 14 rank
    # Total for k=0..3: 28 rank
    
    # For k=4: 2x4x1 -> 8 rank (schoolbook)
    # Total: 36 rank (same as before)
    
    # To get 32, we need to save 4 more rank.
    # Idea: combine the two 2x2x4 blocks more efficiently.
    
    # For j in {0,1} and k in {0,1}: Strassen (7 rank)
    # For j in {0,1} and k in {2,3}: Strassen (7 rank)  
    # For j in {2,3} and k in {0,1}: Strassen (7 rank)
    # For j in {2,3} and k in {2,3}: Strassen (7 rank)
    # For k=4: 8 rank
    # Total: 36 rank
    
    # To reduce to 32, we need a smarter decomposition.
    # Let's try using a 2x4x4 decomposition with rank 24 instead of 28.
    # Is there a rank-24 decomposition for 2x4x4?
    
    # Actually, let's just try to verify if rank 32 is achievable
    # by random search with a good initialization.
    
    return None  # Placeholder - will be implemented below

def _try_lower_rank_random(n, m, p, max_rank, deadline, num_trials=2000):
    """Try to find a lower rank decomposition using random search."""
    for rank in range(max_rank, 24, -1):
        if time.time() > deadline:
            break
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            for i in range(n*m):
                nnz = random.randint(1, 3)
                cols = random.sample(range(rank), nnz)
                for c in cols:
                    U[i, c] = random.choice([-1, 1])
            for j in range(m*p):
                nnz = random.randint(1, 3)
                cols = random.sample(range(rank), nnz)
                for c in cols:
                    V[j, c] = random.choice([-1, 1])
            for k in range(p*n):
                nnz = random.randint(1, 3)
                cols = random.sample(range(rank), nnz)
                for c in cols:
                    W[k, c] = random.choice([-1, 1])
            
            if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _column_merge_search(U_init, V_init, W_init, n, m, p, deadline, target_rank=32):
    """
    Try to reduce rank by merging pairs of columns.
    Start from a rank-r decomposition and try to express it with rank r-1
    by finding if one column is a linear combination of others (in the tensor sense).
    """
    U = np.array(U_init, dtype=np.float64)
    V = np.array(V_init, dtype=np.float64)
    W = np.array(W_init, dtype=np.float64)
    current_rank = U.shape[1]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    while best_rank > target_rank and time.time() < deadline:
        found_reduction = False
        
        for col_to_remove in range(best_rank):
            if time.time() > deadline:
                break
            
            # Try removing this column and see if we can compensate
            mask = [c for c in range(best_rank) if c != col_to_remove]
            U_new = np.zeros((n*m, best_rank - 1))
            V_new = np.zeros((m*p, best_rank - 1))
            W_new = np.zeros((p*n, best_rank - 1))
            for new_c, old_c in enumerate(mask):
                U_new[:, new_c] = best_U[:, old_c]
                V_new[:, new_c] = best_V[:, old_c]
                W_new[:, new_c] = best_W[:, old_c]
            
            # Check if the removed column's contribution can be absorbed
            # by slightly adjusting other columns
            # This is a system of linear equations:
            # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # When we remove col_to_remove, we need to adjust remaining columns
            
            # Simple approach: try small perturbations
            for perturb_trial in range(20):
                if time.time() > deadline:
                    break
                
                U_pert = U_new.copy()
                V_pert = V_new.copy()
                W_pert = W_new.copy()
                
                # Perturb a few entries
                num_perturb = random.randint(3, 8)
                for _ in range(num_perturb):
                    mat = random.choice([U_pert, V_pert, W_pert])
                    row = random.randint(0, mat.shape[0]-1)
                    col = random.randint(0, mat.shape[1]-1)
                    mat[row, col] += random.choice([-0.5, 0.5, -1.0, 1.0, -1.5, 1.5])
                    mat[row, col] = round(mat[row, col] * 2) / 2
                
                err = _verify_with_tolerance(U_pert.tolist(), V_pert.tolist(), W_pert.tolist(), n, m, p, best_rank - 1, tol=1e15)
                if err < 1e-9:
                    best_U, best_V, best_W = U_pert, V_pert, W_pert
                    best_rank -= 1
                    found_reduction = True
                    break
            
            if found_reduction:
                break
        
        if not found_reduction:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _local_search_reduce_rank(U, V, W, n, m, p, deadline, max_iterations=5000):
    """
    Local search: try to eliminate columns one at a time by redistributing
    their contribution to other columns. This is a structured perturbation
    of the existing decomposition.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    iteration = 0
    while best_rank > 25 and iteration < max_iterations and time.time() < deadline:
        iteration += 1
        # Try each column in the current best decomposition
        found = False
        for col_to_remove in range(best_rank):
            if time.time() > deadline:
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            
            # Try to absorb this column's contribution into other columns
            # by slightly perturbing other columns
            new_rank = best_rank - 1
            if new_rank <= 0:
                break
            
            # Create copies without the column to remove
            mask = [c for c in range(best_rank) if c != col_to_remove]
            U_new = np.zeros((n*m, new_rank))
            V_new = np.zeros((m*p, new_rank))
            W_new = np.zeros((p*n, new_rank))
            for new_c, old_c in enumerate(mask):
                U_new[:, new_c] = best_U[:, old_c]
                V_new[:, new_c] = best_V[:, old_c]
                W_new[:, new_c] = best_W[:, old_c]
            
            # Compute the error if we just drop the column
            max_err = _verify_with_tolerance(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p, new_rank, tol=1e15)
            
            if max_err < 1e-9:
                # Column was redundant
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = new_rank
                found = True
                break
            
            # Try small perturbations of remaining columns to absorb the error
            # This is expensive, so only try a few random perturbations
            for perturb_trial in range(3):
                if time.time() > deadline:
                    break
                
                U_pert = U_new.copy()
                V_pert = V_new.copy()
                W_pert = W_new.copy()
                
                # Perturb a random subset of entries
                num_perturb = min(5, new_rank)
                for _ in range(num_perturb):
                    mat = random.choice([U_pert, V_pert, W_pert])
                    row = random.randint(0, mat.shape[0]-1)
                    col = random.randint(0, mat.shape[1]-1)
                    # Small perturbation
                    mat[row, col] += random.choice([-0.5, 0.5, -1.0, 1.0])
                    # Snap to nearest half-integer
                    mat[row, col] = round(mat[row, col] * 2) / 2
                
                err = _verify_with_tolerance(U_pert.tolist(), V_pert.tolist(), W_pert.tolist(), n, m, p, new_rank, tol=1e15)
                if err < 1e-9:
                    best_U, best_V, best_W = U_pert, V_pert, W_pert
                    best_rank = new_rank
                    found = True
                    break
            
            if found:
                break
        
        if not found:
            # No more reductions possible with this strategy
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    # First try the constructive rank-36
    result = _construct_hybrid_rank36()
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0])
    
    if _verify_fast(U, V, W, n, m, p, rank):
        best_result = result
        best_rank = rank
    else:
        # Fallback to schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    # Try column merge search to reduce rank
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        merge_result = _column_merge_search(
            best_result["U"], best_result["V"], best_result["W"],
            n, m, p, deadline, target_rank=30
        )
        merge_rank = len(merge_result["U"][0])
        if merge_rank < best_rank:
            best_result = merge_result
            best_rank = merge_rank
    
    # Try local search to reduce rank further
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        local_result = _local_search_reduce_rank(
            best_result["U"], best_result["V"], best_result["W"],
            n, m, p, deadline, max_iterations=2000
        )
        local_rank = len(local_result["U"][0])
        if local_rank < best_rank:
            best_result = local_result
            best_rank = local_rank
    
    # Try random search for even lower rank
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        random_result = _try_lower_rank_random(n, m, p, best_rank - 1, deadline, num_trials=2000)
        if random_result is not None:
            rand_rank = len(random_result["U"][0])
            if rand_rank < best_rank:
                best_result = random_result
                best_rank = rand_rank
    
    return best_result

def construct():
    return _construct()
# EVOLVE-BLOCK-END
