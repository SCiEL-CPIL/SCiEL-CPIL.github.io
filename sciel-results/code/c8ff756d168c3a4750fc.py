import time
import numpy as np
import random

def construct():
    n, m, p = 3, 4, 5
    deadline = time.time() + 110  # 110s to be safe within 120s
    
    # Schoolbook baseline
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Strategy: Random U, V; Solve for W linearly.
    # For each (i, k), we have a system A_{i,k} w_{i,k} = 1
    # where A_{i,k}[j, s] = U[i, j, s] * V[j, k, s]
    # We reshape U to (n, m, r), V to (m, p, r), W to (n, p, r)
    
    # Try ranks from 25 to 59
    # We prioritize lower ranks first to find the minimum.
    # However, lower ranks are harder to satisfy.
    # Let's scan r from 25 to 59.
    
    for r in range(25, 60):
        if time.time() > deadline - 10:
            break
            
        attempts = 0
        max_attempts = 50000  # High number of attempts for each rank
        
        while attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            
            # Generate random U and V with small integer entries
            # Using {-1, 0, 1} is a good start. 
            # Sometimes {-1, 0, 1, 2} or just {-1, 1} works better.
            # Let's mix: mostly -1, 0, 1.
            
            # Reshape for easier indexing: U[i, j, s]
            U_flat = np.random.choice([-1, 0, 1], size=(n * m, r))
            V_flat = np.random.choice([-1, 0, 1], size=(m * p, r))
            
            # Check if solution exists and is "nice"
            # We need to solve for W_flat of shape (p * n, r)
            # Actually, W is indexed by (k, i) in the output? 
            # Task says: W[k*n+i, r]. So W is (p*n, r).
            # The term in the sum is U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s].
            
            # For a fixed i, k, we solve for W[k*n+i, s] for s=0..r-1.
            # The equations are for j=0..m-1:
            # sum_{s} U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
            
            valid_solution = True
            W_solution = np.zeros((p * n, r))
            
            for i in range(n):
                for k in range(p):
                    # Build the m x r matrix A
                    # A[j, s] = U[i*m+j, s] * V[j*p+k, s]
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U_flat[i * m + j, :] * V_flat[j * p + k, :]
                    
                    # Solve A * w = 1
                    # If A is not full rank (rank < m), there might be no solution or infinite.
                    # We require a unique solution for simplicity, or check if 1 is in the column space.
                    # Using least squares:
                    try:
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        # Check if the solution is exact
                        if np.allclose(A @ w, np.ones(m), atol=1e-9):
                            # Check if w entries are close to integers or half-integers
                            # The task allows integer/half-integer.
                            # Let's check if 2*w is close to an integer.
                            if not np.allclose(2 * w, np.round(2 * w), atol=1e-6):
                                valid_solution = False
                                break
                            W_solution[k * n + i, :] = w
                        else:
                            valid_solution = False
                            break
                    except np.linalg.LinAlgError:
                        valid_solution = False
                        break
                
                if not valid_solution:
                    break
            
            if valid_solution:
                # Found a solution!
                best_rank = r
                best_U, best_V, best_W = U_flat, V_flat, W_solution
                # We can break out of the attempts loop for this r
                # Since we are scanning r from low to high, the first found is minimal for this search strategy.
                # But we should probably continue a bit to see if we find a better one?
                # No, we are iterating r upwards. The first r we find is the best we can get with this r.
                # We should break out of the r loop as well, unless we want to verify if a lower r was missed?
                # We started at 25. If we find one at 30, we should stop.
                break
                
        if best_rank < rank_school:
            # We found a better rank. Since we iterate r from low to high, this is the best found so far.
            # We can stop searching higher ranks.
            break

    # Convert to lists with integer/half-integer values
    def to_list(arr):
        # Round to nearest 0.5
        return [[float(round(x * 2) / 2) for x in row] for row in arr]

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
    
    # Verification
    n, m, p = 3, 4, 5
    r = U.shape[1]
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for s in range(r):
                    val += U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
                if abs(val - 1) > 1e-6:
                    valid = False
                    print(f"Failed at i={i}, j={j}, k={k}, val={val}")
                    break
            if not valid: break
        if not valid: break
    print(f"Valid: {valid}")
