"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _check_decomposition(U, V, W, n, m, p):
    """Check if decomposition is valid."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_trivial
    
    # Search for lower rank decompositions
    # Strategy: Random U, V (sparse integers), solve for W, check if W is "simple" (small denominator).
    # If yes, scale to integers and verify.
    
    max_attempts_per_rank = 2000
    
    for rank in range(rank_trivial - 1, 15, -1):
        if time.time() > deadline:
            break
            
        for attempt in range(max_attempts_per_rank):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 1000 + rank)
            
            # Generate sparse U and V with entries in {-1, 0, 1}
            # Probability of non-zero is 0.4
            U_r = np.where(np.random.rand(n * m, rank) < 0.4, np.random.choice([-1, 1], size=(n * m, rank)), 0)
            V_r = np.where(np.random.rand(m * p, rank) < 0.4, np.random.choice([-1, 1], size=(m * p, rank)), 0)
            
            U_np = U_r.astype(float)
            V_np = V_r.astype(float)
            
            W_np = np.zeros((p * n, rank))
            valid = True
            scale_factor = 1
            
            # For each (i, k), solve for W_{k, i, :}
            for i in range(n):
                for k in range(p):
                    # Construct the system A W = 1
                    # A has shape (m, rank)
                    # A[j, r] = U[i, j, r] * V[j, k, r]
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            A[j, r] = U_np[i * m + j, r] * V_np[j * p + k, r]
                    
                    # Solve A W_col = 1
                    try:
                        W_col, residuals, rank_l, singular_values = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        
                        # Check if solution is close to exact
                        if np.linalg.norm(A @ W_col - np.ones(m)) > 1e-6:
                            valid = False
                            break
                            
                        # Check if W_col entries are "simple"
                        # Try denominators 1, 2, 3, 4, 5
                        simple = True
                        W_simple = [0.0] * rank
                        local_scale = 1
                        for r in range(rank):
                            val = W_col[r]
                            found = False
                            for den in [1, 2, 3, 4, 5]:
                                num = round(val * den)
                                if abs(val - num / den) < 1e-6:
                                    W_simple[r] = num / den
                                    local_scale = den
                                    found = True
                                    break
                            if not found:
                                simple = False
                                break
                        if not simple:
                            valid = False
                            break
                            
                        W_np[k * n + i, :] = W_simple
                        # Update scale factor to LCM of all scales found so far
                        # We just need to ensure the final result is half-integer.
                        # If we use denominators up to 5, we might need to scale by 5.
                        # But the task allows integers.
                        # So we can just keep the rational W and scale at the end?
                        # No, we need to return the matrices.
                        # We can return the scaled matrices.
                        
                        # For now, just record that we need to scale.
                        # We'll determine the global scale after the loop.
                        if not valid:
                            break
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    if not valid:
                        break
            
            if valid:
                # Determine global scale factor
                # We need to find a common denominator for all W entries.
                # Since we checked against denominators 1..5, the LCM is 60.
                # But we can just check the actual values.
                # Let's assume we can scale by 60 to be safe, or find the min scale.
                # Actually, if we found W entries as num/den, we can just multiply U, V, W by 60.
                # But U, V are integers.
                # So U' = 60*U, V' = 60*V, W' = 60*W.
                # Then U', V', W' are integers.
                
                # However, we want the *smallest* rank. Scaling doesn't change rank.
                # So we can just return the scaled integer matrices.
                
                # Let's verify the decomposition with the scaled matrices.
                # Actually, _check_decomposition works with floats.
                # We can just pass the float versions.
                
                U_list = U_np.tolist()
                V_list = V_np.tolist()
                W_list = W_np.tolist()
                
                if _check_decomposition(U_list, V_list, W_list, n, m, p):
                    # Scale to integers to be safe and clean
                    # Find max denominator
                    # Since we checked 1..5, 60 is safe.
                    # But we can just return the floats if they are close to half-integers?
                    # The task says "integer/half-integer entries".
                    # So we should ensure they are.
                    # If W has 1/3, it's not half-integer.
                    # So we MUST scale.
                    
                    # Let's scale by 60.
                    U_int = [[int(round(x * 60)) for x in row] for row in U_list]
                    V_int = [[int(round(x * 60)) for x in row] for row in V_list]
                    W_int = [[int(round(x * 60)) for x in row] for row in W_list]
                    
                    # Verify with integers
                    if _check_decomposition(U_int, V_int, W_int, n, m, p):
                        best_result = {"U": U_int, "V": V_int, "W": W_int}
                        best_rank = rank
                        break # Found a valid decomposition for this rank, move to next lower rank
        
        if best_rank < rank_trivial:
            # Continue searching for lower rank
            pass
        
    return best_result
# EVOLVE-BLOCK-END
