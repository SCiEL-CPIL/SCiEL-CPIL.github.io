import numpy as np

def construct():
    n, m, p = 3, 5, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Initialize schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Tensor entries: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For schoolbook, T[i,j,k] = 1 for all i,j,k
    
    # Greedy elimination: try to remove columns
    # For each column, check if we can still achieve T[i,j,k]=1 for all i,j,k
    # without that column, by adjusting the remaining columns
    
    active_cols = list(range(rank))
    
    for col_to_remove in range(rank):
        if col_to_remove not in active_cols:
            continue
        
        # Try removing this column
        new_active = [c for c in active_cols if c != col_to_remove]
        
        # Check if the tensor can be represented with remaining columns
        # We need to solve: for all i,j,k, sum_{c in new_active} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is nonlinear, so we check if the current values with this column zeroed out still work
        # If we just zero out this column, the tensor entry becomes 0 for the (i,j,k) that this column was responsible for
        # So we need to check if other columns cover it
        
        # For schoolbook, each column covers exactly one (i,j,k) triple
        # So removing a column makes that triple have sum=0, not 1
        # Thus we can't simply remove columns from schoolbook
        
        # Instead, try to find a better initial decomposition
        pass
    
    # Return schoolbook as fallback
    return {"U": U, "V": V, "W": W}
