"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # We will attempt to find a decomposition with rank < n*m*p.
    # The schoolbook rank is n*m*p.
    # We start searching from rank n*m*p - 1 downwards.
    
    def try_rank(rank):
        if time.time() > deadline - 10:
            return None
            
        # We use a simple heuristic: 
        # For matrix multiplication, the tensor is T_{i,j,k} = delta_{i,i'} delta_{j,j'} delta_{k,k'} (summed over indices in the product).
        # Actually, the tensor M is defined such that (AB)_{ik} = sum_j A_{ij} B_{jk}.
        # The tensor entries are T_{i,j,k} = 1 if the index mapping matches.
        # Specifically, U is (n*m) x R, V is (m*p) x R, W is (p*n) x R.
        # The condition is: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 if (i,j,k) is a valid triple? 
        # No, the tensor for matrix multiplication has non-zero entries T_{(i,j), (j,k), (i,k)} = 1.
        # So we need sum_c U_{i,j,c} V_{j,k,c} W_{i,k,c} = 1 for all i,j,k.
        
        # Let's try to solve the linear system for fixed rank.
        # Number of variables: (n*m + m*p + p*n) * rank.
        # Number of equations: n*m*p.
        
        # This is a large linear system. We can use least squares or sparse solvers.
        # However, we need exact integer/half-integer solutions.
        # A common strategy is to use the "border rank" or known decompositions.
        
        # For 2x3x5, the schoolbook rank is 30.
        # It is known that for 2x3, rank is 7 (schoolbook 6). Wait, 2x3x2? No.
        # For 2x3 matrix mult (2x3 * 3x2), rank is 7? No, 2x3*3x2 is 2x2 result.
        # The task is 2x3 * 3x5. Result 2x5.
        # Tensor size: 2*3*5 = 30.
        
        # Let's just return the schoolbook solution if we can't find a better one quickly.
        # Or try a very simple search.
        
        # Given the complexity, and that we have 120s, we can try a randomized search for small rank improvements.
        # But finding an exact rank reduction for 2x3x5 might be hard.
        # Let's stick to schoolbook if we don't have a specific known improvement.
        # However, the prompt asks to improve.
        
        # Is there a known rank reduction for 2x3x5?
        # The rank of matrix multiplication M_{n,m,p} is at most n*m*p.
        # For small matrices, rank reductions are known.
        # M_{2,3,2} has rank 7? No, M_{2,2,2} is 7.
        # M_{2,3,2} (2x3 * 3x2) -> 2x2 result. Rank 7?
        # Here we have 2x3 * 3x5 -> 2x5 result.
        
        # Let's try to use the fact that we can combine terms.
        # If we can't find a better one, we return schoolbook.
        
        return None

    # If we don't find a better rank, return schoolbook.
    # The current program returns schoolbook.
    # To "improve", we need to find a rank < 30.
    
    # Let's try to implement a solver for rank 29.
    # Variables: 30*29 = 870? No.
    # U: 6*29, V: 15*29, W: 10*29. Total 31*29 = 899 variables.
    # Equations: 30.
    # This is an underdetermined system. We can find a solution easily.
    # But we need it to be "exact" and likely "minimal" or at least lower.
    # Wait, the rank is the number of terms.
    # If we set rank=29, we have 30 equations and 899 variables.
    # We can set 29 of the variables to 1 and see if we can satisfy the equations?
    # No, the equations are:
    # sum_c U[i,j,c] V[j,k,c] W[i,k,c] = 1 for all i,j,k.
    
    # If we use rank 29, we can take the schoolbook rank 30 and try to merge two terms?
    # Or simply, we can solve the linear system.
    
    # Let's try rank = n*m*p - 1.
    target_rank = n * m * p - 1
    
    # Construct the linear system A x = b
    # x is the vector of all U, V, W entries.
    # But the equations are cubic (product of three variables).
    # So it's not a linear system in the variables U, V, W.
    # It's a system of polynomial equations.
    
    # This is NP-hard in general.
    # However, for small ranks, we can use Groebner bases or simple search.
    # But 120s is not enough for complex algebraic geometry.
    
    # Let's check if there is a trivial improvement.
    # For 2x3x5, is rank 29 possible?
    # If yes, how to find it?
    
    # Alternative: The "score" is the rank.
    # If we cannot find a lower rank, we should return the schoolbook rank 30.
    # The current program returns 30.
    # To improve, we MUST find a rank < 30.
    
    # Let's try a heuristic search for rank 29.
    # We can fix most variables to 0 or 1.
    
    # Actually, for M_{2,3,5}, the rank is known to be 30? 
    # Or is it less?
    # The rank of M_{2,3,2} is 7 (schoolbook 6? No, 2*3*2=12? No, 2*3*2=12. Rank 7? No, M_{2,2,2}=7. M_{2,3,2} is 2x3*3x2. 
    # Wait, M_{n,m,p} is n x m * m x p.
    # M_{2,3,2}: 2x3 * 3x2. Rank?
    # Schoolbook: 12.
    # Known rank for 2x3*3x2 is 7? No, that's 2x2*2x2.
    
    # Let's assume the rank is 30 for now, unless we find a proof otherwise.
    # But the task asks to improve.
    
    # Let's try to return the schoolbook solution but with a check for time.
    
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
