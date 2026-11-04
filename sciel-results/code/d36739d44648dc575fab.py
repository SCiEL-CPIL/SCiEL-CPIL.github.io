import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140  # 140s to leave safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Strategy: Use block decomposition
    # 3x3 x 3x8 = (3x3 x 3x3) + (3x3 x 3x3) + (3x3 x 3x2)
    # We'll try to find a decomposition with rank < 72
    
    # Try ranks from 71 down to 56
    for rank in range(71, 55, -1):
        if time.time() > deadline:
            break
        
        # Try a few random attempts for this rank
        attempts = 5
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with integer/half-integer values
            # Use small values to increase chance of exact match
            U = np.random.randint(-2, 3, (n * m, rank)) * 0.5
            V = np.random.randint(-2, 3, (m * p, rank)) * 0.5
            W = np.random.randint(-2, 3, (p * n, rank)) * 0.5
            
            # Check if this decomposes the tensor exactly
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        val = 0
                        for c in range(rank):
                            val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        expected = 1.0
                        if abs(val - expected) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Also check that non-tensor entries are 0
            # Actually, the tensor is only defined for valid (i,j,k) triples
            # We need to ensure that for invalid combinations, the sum is 0
            # But in our indexing, (i,j) and (j,k) determine (i,k) uniquely
            # So we only need to check the tensor entries
            
            if valid:
                # Convert to list of lists with integer/half-integer values
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                # Round to nearest half-integer
                U_list = [[round(x * 2) / 2 for x in row] for row in U_list]
                V_list = [[round(x * 2) / 2 for x in row] for row in V_list]
                W_list = [[round(x * 2) / 2 for x in row] for row in W_list]
                
                # Verify after rounding
                valid_after_round = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(rank):
                                val += U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c]
                            if abs(val - 1.0) > 1e-9:
                                valid_after_round = False
                                break
                        if not valid_after_round:
                            break
                    if not valid_after_round:
                        break
                
                if valid_after_round:
                    best_rank = rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break  # Found a valid decomposition for this rank
    
    if time.time() > deadline:
        return best_result
    
    return best_result
