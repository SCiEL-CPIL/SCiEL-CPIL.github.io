import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 3, 8
    deadline = time.time() + 145  # Leave 5s safety margin
    
    # Start with schoolbook (rank 48)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    
    # Try to find lower rank decompositions
    # The multiplication tensor T[i,j,k] = delta_{i',i} * delta_{j',j} * delta_{k',k}
    # We want T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    
    # For validation: check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if indices match, 0 otherwise
    # Actually, the convention is: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] where i is row of A, j is column of A / row of B, k is column of B
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # But W is indexed by (k*n+i) per the problem statement: W[k*n+i, r]
    
    def validate(U, V, W, rank):
        """Check if the decomposition exactly reproduces matrix multiplication."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # For each (i, j, k) in the tensor, check:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal:
        # 1 if this is a valid multiplication term, 0 otherwise
        # Actually, the tensor for matrix mult is: T[i,j,k] = 1 for all valid (i,j,k)
        # No wait - the tensor T is defined such that:
        # sum_j A[i,j] * B[j,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] * A[i,j] * B[j,k]
        # This should equal sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k])
        # For this to equal C[i,k] = sum_j A[i,j] * B[j,k] for all A, B,
        # we need: for each r, the "intermediate" computation to work out.
        # 
        # The correct condition is:
        # For all i, k: sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) = sum_j A[i,j] * B[j,k]
        # 
        # This must hold for ALL A and B, which means the tensor identity:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j,k} 
        # where delta is 1 if (i,j,k) forms a valid multiplication term
        # 
        # Actually, for matrix multiplication, the tensor is simply:
        # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # because C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Try ranks from 47 down
    # Use a simple approach: try to find integer/half-integer solutions
    # This is a system of bilinear equations, so we can try random initialization + local search
    
    import random
    random.seed(42)
    
    for target_rank in range(47, 20, -1):
        if time.time() > deadline:
            break
        
        # Try a few random attempts for this rank
        max_attempts = 3
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize with random values in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n*m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m*p, target_rank))
            W = np.random.choice([-1, 0, 1], size=(p*n, target_rank))
            
            if validate(U, V, W, target_rank):
                best_rank = target_rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                found = True
                break
        
        if found:
            break
        if time.time() > deadline - 10:
            break
    
    # Convert to lists if still using schoolbook
    if isinstance(best_U, np.ndarray):
        best_U = best_U.tolist()
        best_V = best_V.tolist()
        best_W = best_W.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}
