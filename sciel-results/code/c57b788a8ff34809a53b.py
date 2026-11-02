"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try a low-rank decomposition first (e.g., rank 9 for all-1 tensor)
    # If the tensor is T[i,j,k] = 1 for all i,j,k, rank 9 suffices.
    # U[i*m+j, r] = 1/3 if r == j
    # V[j*p+k, r] = 1 if r == j
    # W[k*n+i, r] = 1 if r == j
    # Sum_r (1/3 * 1 * 1) = 1.
    
    rank_try = 9
    U = [[0] * rank_try for _ in range(n * m)]
    V = [[0] * rank_try for _ in range(m * p)]
    W = [[0] * rank_try for _ in range(p * n)]
    
    # Fill U, V, W for the rank 9 hypothesis
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1/3
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][j] = 1 # Wait, j is not in scope here. 
            # Correct: W[k*n+i, r] should be 1 if r == j? 
            # But j is the index of V. 
            # The term is U[i,j,r] V[j,k,r] W[k,i,r].
            # If r=j, U[i,j,j]=1/3, V[j,k,j]=1, W[k,i,j]=1.
            # So W[k*n+i, j] should be 1.
            # But j is a loop variable in the previous blocks. 
            # We need to fill W for all k,i and r.
            # W[k*n+i, r] = 1 if r is the shared index?
            # The shared index is j.
            # So W[k*n+i, r] = 1 for all r? No.
            # If W[k*n+i, r] = 1 for all r, then sum is 3 * 1/3 = 1.
            # So W[k*n+i, r] = 1 for all r.
            pass

    # Let's re-evaluate the rank 9 construction.
    # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    # Let r correspond to j.
    # U[i,j,r] = 1/3 if r==j else 0.
    # V[j,k,r] = 1 if r==j else 0.
    # W[k,i,r] = 1 if r==j else 0? 
    # If W[k,i,r] = 1 if r==j, then for a fixed i,k, we sum over j.
    # But W is fixed for k,i. It has entries for all r.
    # W[k,i,r] must be 1 for all r? 
    # If W[k,i,r] = 1 for all r, then:
    # Sum_r (1/3 delta_{jr} * 1 delta_{jr} * 1) = Sum_r (1/3 delta_{jr}) = 1/3.
    # This gives 1/3, not 1.
    # So we need U[i,j,r] = 1 if r==j.
    # V[j,k,r] = 1 if r==j.
    # W[k,i,r] = 3 if r==j? 
    # Then sum is 3.
    # Or U=1, V=1, W=1 gives 1.
    # If U[i,j,r]=1 (r=j), V[j,k,r]=1 (r=j), W[k,i,r]=1 (r=j).
    # Sum is 1.
    # So:
    # U[i*m+j, r] = 1 if r==j else 0.
    # V[j*p+k, r] = 1 if r==j else 0.
    # W[k*n+i, r] = 1 if r==j else 0.
    
    U = [[0] * rank_try for _ in range(n * m)]
    V = [[0] * rank_try for _ in range(m * p)]
    W = [[0] * rank_try for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
    for k in range(p):
        for i in range(n):
            for r in range(m):
                W[k * n + i][r] = 1
                
    # Verify this decomposition
    def is_valid(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Check T[i,j,k] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    if is_valid(U, V, W):
        return {"U": U, "V": V, "W": W}
        
    # Fallback to schoolbook rank 27
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
# EVOLVE-BLOCK-END
