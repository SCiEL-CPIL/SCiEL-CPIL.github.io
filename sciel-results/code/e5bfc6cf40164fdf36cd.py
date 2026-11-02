import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Try to find a decomposition with rank < schoolbook_rank
    # We'll use a randomized approach to search for lower rank decompositions
    
    best_result = None
    best_rank = schoolbook_rank
    
    # First, provide the schoolbook decomposition as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank using random initialization and optimization
    # For matrix multiplication tensor, we can try to find a rank-r decomposition
    # where r < n*m*p
    
    # Try ranks from schoolbook_rank - 1 down to some minimum
    for target_rank in range(schoolbook_rank - 1, max(n * m, m * p, p * n) - 1, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        max_attempts = 10
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Check if this decomposition is valid
            # For matrix multiplication, we need:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
            # Actually, the tensor decomposition condition is:
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # where T is the matrix multiplication tensor
            
            # Let's verify by checking a few random indices
            valid = True
            for _ in range(100):
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                
                # Compute the decomposition value
                val = 0
                for r in range(target_rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                
                # The true tensor value is 1 if this is the "diagonal" element
                # Actually, for matrix multiplication C = A * B where A is n x m and B is m x p,
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # So the tensor T[i,j,k] = 1 for all valid (i,j,k) in the product
                # Wait, let me reconsider. The tensor for matrix multiplication is:
                # T_{ijk} = 1 if we're computing C_{ik} = sum_j A_{ij} B_{jk}
                # So T_{ijk} = 1 for all i,j,k in their respective ranges
                
                expected = 1.0
                if abs(val - expected) > 1e-6:
                    valid = False
                    break
            
            if valid:
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
        
        if best_rank < schoolbook_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
