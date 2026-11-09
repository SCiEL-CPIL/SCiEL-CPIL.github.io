"""Matrix multiplication 2,5,5 rank search with deadline guard."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

# Deadline: 150s wall-clock; keep safety margin.
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, tol=1e-9):
    """Check that U,V,W exactly decompose the 2x5x5 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = 1 if (i,j,k) is a valid index, else 0
    # The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # for all valid (i,j,k), and == 0 for invalid (but all are valid here since n,m,p>0).
    # Actually for matrix multiplication, the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j]*B[j,k], so each (i,j,k) contributes.
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape for efficient computation
    U3 = U_np.reshape(n, m, rank)  # U[i*m+j, c] -> U3[i,j,c]
    V3 = V_np.reshape(m, p, rank)  # V[j*p+k, c] -> V3[j,k,c]
    W3 = W_np.reshape(p, n, rank)  # W[k*n+i, c] -> W3[k,i,c]
    
    # For each (i,j,k), check sum_c U3[i,j,c]*V3[j,k,c]*W3[k,i,c] == 1
    # This is expensive for large rank, but we can do it incrementally.
    # Let's just check all entries.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    best_result = _schoolbook()
    best_rank = best_result["U"][0].__len__() if best_result["U"] else 0
    
    # Try to find a lower rank decomposition
    n, m, p = DIMENSIONS
    total_elements = n * m * p  # 50
    
    # Known lower bounds: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # Upper bound: schoolbook = 50
    # Let's try ranks from a reasonable lower bound upward
    
    min_possible = max(n * m, m * p, p * n)
    
    # For 2x5x5, known results suggest rank might be around 25-35
    # Let's try a random search with random initialization and gradient-free optimization
    
    # Since we need exact integer/half-integer entries, let's use a different approach:
    # Try structured decompositions based on known low-rank algorithms for small matrices.
    
    # Approach: Use the fact that for matrix multiplication, we can sometimes
    # share computations. For 2x5x5, let's try a few known patterns.
    
    # First, let's verify the schoolbook is valid
    if _verify(best_result["U"], best_result["V"], best_result["W"]):
        pass  # It's valid
    
    # Try random search with small ranks
    for rank in range(min_possible, min_possible + 15):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Try a few random initializations
        attempts = 0
        max_attempts = 50 if rank < 30 else 20
        
        while attempts < max_attempts:
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            attempts += 1
            
            # Random integer/half-integer entries
            # Scale: entries should be small to avoid overflow
            scale = 1.0
            
            U = np.random.randint(0, 3, (n * m, rank)) * 0.5  # 0, 0.5, 1.0
            V = np.random.randint(0, 3, (m * p, rank)) * 0.5
            W = np.random.randint(0, 3, (p * n, rank)) * 0.5
            
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Quick check using numpy
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        val = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
            
            if valid:
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                break  # Found valid at this rank, move to next rank
        
        # If we found a valid decomposition at this rank, we can stop
        if best_rank == rank:
            break
    
    return best_result

# Keep the function name as expected
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank
    min_possible = max(n * m, m * p, p * n)  # 25
    
    # Search ranks from min_possible upward
    for rank in range(min_possible, min_possible + 10):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Try structured approach: use the fact that matrix multiplication
        # can sometimes be done with fewer multiplications by exploiting structure.
        
        # For 2x5x5, let's try a specific pattern based on sharing.
        # Idea: For each pair (i,k), we need sum_j A[i,j]*B[j,k].
        # If we can share some intermediate results across different (i,k) pairs,
        # we can reduce the rank.
        
        # Random search with restarts
        found = False
        max_attempts = 30
        
        for attempt in range(max_attempts):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            # Random initialization with small integers/half-integers
            U = np.random.randint(0, 3, (n * m, rank)) * 0.5
            V = np.random.randint(0, 3, (m * p, rank)) * 0.5
            W = np.random.randint(0, 3, (p * n, rank)) * 0.5
            
            # Verify
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        val = float(np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :]))
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
            
            if valid:
                best = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                found = True
                break
        
        if found:
            break
    
    return best
