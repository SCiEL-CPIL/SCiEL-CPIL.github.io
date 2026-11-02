"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_trivial
    
    # The multiplication tensor T[i,j,k,l] = delta_{j,k}
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,l,r] = delta_{j,k}
    # 
    # Key observation: For fixed (j,k), the equation is:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,l,r] = delta_{j,k} for all i,l
    #
    # This means: for each (j,k) pair, the outer product of vectors
    # (U[:,j,r] weighted by V[j,k,r]) and (W[k,:,r] weighted by V[j,k,r])
    # must sum to delta_{j,k} * (all-ones matrix)
    #
    # Let's try a structured approach where we use rank-1 terms that
    # correspond to specific (j,k) pairs or combinations thereof.
    
    # Try to find a decomposition with rank < trivial
    # Use a greedy/approximate approach with exact verification
    
    def verify_decomposition(U_mat, V_mat, W_mat, rank):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for l in range(p):
                        # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,l,r]
                        total = Fraction(0)
                        for r in range(rank):
                            total += Fraction(U_mat[i*m+j][r]) * Fraction(V_mat[j*p+k][r]) * Fraction(W_mat[k*n+l][r])
                        expected = Fraction(1) if j == k else Fraction(0)
                        if total != expected:
                            return False
        return True
    
    def try_structured_decomposition(rank):
        """Try to find a structured decomposition of given rank."""
        # Strategy: Use the fact that the tensor has a specific structure.
        # For matrix multiplication, we can use decompositions based on
        # the Kronecker product structure.
        
        # One approach: use rank = n*m + m*p + p*n - m (or similar)
        # Another: use known bounds
        
        # Let's try a direct construction:
        # For each j, we need terms that handle the j-th column of A and j-th row of B
        # 
        # Actually, let's think about this differently.
        # The tensor T has entries T[i,j,k,l] = delta_{j,k}
        # 
        # A rank-1 decomposition term is U_r ⊗ V_r ⊗ W_r where:
        # (U_r ⊗ V_r ⊗ W_r)[i,j,k,l] = U_r[i,j] * V_r[j,k] * W_r[k,l]
        # 
        # We need sum_r U_r[i,j] * V_r[j,k] * W_r[k,l] = delta_{j,k}
        
        # Let's try: for each j, create terms that isolate the j-th "slice"
        # Term for j: U_r[i,j] = 1 if j matches, else 0
        #              V_r[j,k] = 1 if k == j, else 0  
        #              W_r[k,l] = 1 for all i,l (but this doesn't work directly)
        
        # Better approach: Use the identity that delta_{j,k} can be written as
        # sum over basis vectors. Let's use a different strategy.
        
        # For now, let's try random initialization with local search
        if time.time() > deadline - 5:
            return None
        
        # Initialize with small random values
        np.random.seed(42)
        U_mat = np.random.randint(0, 2, size=(n*m, rank))
        V_mat = np.random.randint(0, 2, size=(m*p, rank))
        W_mat = np.random.randint(0, 2, size=(p*n, rank))
        
        # Try a few iterations of improvement
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            
            # Check if current solution works
            if verify_decomposition(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), rank):
                return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
            
            # Try to fix one entry at a time
            # This is a heuristic approach - try setting entries to 0, 1, or -1
            changed = False
            for idx in range(n*m):
                for r in range(rank):
                    old_val = U_mat[idx][r]
                    for new_val in [0, 1, -1]:
                        if new_val != old_val:
                            U_mat[idx][r] = new_val
                            if verify_decomposition(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), rank):
                                return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
                            U_mat[idx][r] = old_val
                            changed = True
                            if time.time() > deadline - 10:
                                break
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
            if not changed:
                break
        
        return None
    
    # Try ranks from lower bound up to trivial
    # Lower bound for matrix multiplication rank: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But we can often do better
    
    lower_bound = max(n*m, m*p, p*n)
    
    for rank in range(lower_bound, min(rank_trivial, lower_bound + 30)):
        if time.time() > deadline - 15:
            break
        
        result = try_structured_decomposition(rank)
        if result is not None:
            # Verify once more
            if verify_decomposition(result["U"], result["V"], result["W"], rank):
                return result
    
    # If we haven't found anything better, return the trivial solution
    return best_result
# EVOLVE-BLOCK-END
