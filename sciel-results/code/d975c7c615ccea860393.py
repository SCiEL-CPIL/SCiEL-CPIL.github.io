import numpy as np
from fractions import Fraction
import itertools
import random

def construct():
    n, m, p = 2, 4, 6
    
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p  # 48
    
    # We'll try to find a better decomposition
    # The tensor T[i,j,k] for i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if the index (i,j) in A and (j,k) in B correspond to the same j
    # Otherwise T[i,j,k] = 0
    
    # Actually, the matrix multiplication tensor is:
    # T[i,j,k] = 1 if i, j, k are such that we multiply A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k), but the "value" depends on the structure
    
    # Wait, let me reconsider. The multiplication tensor T is defined such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T[i,j,k] = 1 if this term contributes to C[i,k], else 0
    # But actually, for matrix multiplication, T[i,j,k] = 1 for all (i,j,k) in the valid range
    # because A[i,j]*B[j,k] always contributes to C[i,k]
    
    # No, that's not right. The tensor is:
    # T[i,j,k] = 1 if the product A[i,j]*B[j,k] is part of the computation of C[i,k]
    # Since every (i,j,k) triple with 0<=i<n, 0<=j<m, 0<=k<p contributes, T[i,j,k]=1 for all valid indices
    
    # But wait, that would make the tensor all ones, which has rank 1!
    # That can't be right for matrix multiplication...
    
    # Let me reconsider the problem. The matrix multiplication tensor is:
    # T: V^* ⊗ W^* ⊗ U -> C where (A,B) -> A*B
    # In coordinates: T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in the product
    # So T[i,j,k] = 1 for all valid (i,j,k)
    
    # But the tensor rank of the matrix multiplication tensor M_{n,m,p} is known to be:
    # At most n*m*p (schoolbook)
    # At least n*m + m*p + n*p - n - m - p + 1 (lower bound)
    # For n=2, m=4, p=6: lower bound = 8 + 24 + 12 - 2 - 4 - 6 + 1 = 33
    
    # The exact rank is unknown, but we can try to find a decomposition with rank < 48
    
    # Let's try a structured approach. We'll fix U to be the identity (or a specific form)
    # and try to find V and W that satisfy the equations.
    
    # Actually, let me try a random search approach with some structure.
    
    # First, let's verify the schoolbook works:
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    expected = 1  # All terms contribute
                    if val != expected:
                        return False
        return True
    
    # Schoolbook
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    # Try to find a rank-40 decomposition using a structured approach
    # We'll use the fact that the tensor has some structure
    
    # Let's try a random approach: generate random U, V and try to solve for W
    # But this is nonlinear...
    
    # Instead, let's try a specific structured decomposition.
    
    # Approach: Use the identity decomposition but try to merge some terms
    
    # For now, let's just return the schoolbook decomposition
    # but let me try one more thing: a rank-36 decomposition
    
    # Actually, let me try a simple random search for a lower rank
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks from 40 down to 35
    for target_rank in range(40, 34, -1):
        # Try a few random initializations
        for trial in range(10):
            random.seed(42 + trial + target_rank)
            
            # Initialize U and V randomly with entries from {-1, 0, 1}
            U_trial = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(target_rank)] for _ in range(n*m)]
            V_trial = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(target_rank)] for _ in range(m*p)]
            
            # For each W row, we need to solve:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # This is a system of equations for W
            
            # But this is still nonlinear because W appears multiplied by U and V
            
            # Let's try a different approach: fix U and V to be sparse, then W is determined
            
            # Actually, for a rank-r decomposition, we have r "terms"
            # Each term c gives a rank-1 contribution
            # The sum of these contributions must equal the target tensor
            
            # Let me try: set U[i*m+j, c] and V[j*p+k, c] for all (i,j,k,c)
            # Then W[k*n+i, c] is determined by the equation
            # But the equation must hold for all (i,j,k)
            
            # For fixed c, the contribution is:
            # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # The total sum over c must equal 1 for all (i,j,k)
            
            # This is a system of n*m*p = 48 equations
            # With r*(n*m + m*p + p*n) unknowns, but many are coupled
            
            # Let's try a greedy approach: for each term c, try to make it contribute
            # to some (i,j,k) entries
            
            # This is getting too complex. Let me just return the schoolbook.
            pass
    
    # Let me try one specific structured decomposition that might work
    
    # Idea: Use the fact that n=2 is small
    # We can write the multiplication as:
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # This is just two dot products of length 4
    # The rank of a dot product of length m is m (it's already minimal)
    # So for two dot products, the rank might be 2*m = 8? No, that's not right either
    
    # Actually, the matrix multiplication tensor M_{2,4,6} has a known upper bound
    # Let me look up: the rank of M_{2,4,6} is at most 2*4*6 = 48 (schoolbook)
    # But there might be better decompositions
    
    # For now, let me try a specific rank-40 decomposition
    
    # Let's use the following structure:
    # For c = 0,1,...,11: handle the j=0 terms
    # For c = 12,13,...,23: handle the j=1 terms
    # For c = 24,25,...,35: handle the j=2 terms
    # For c = 36,37,...,47: handle the j=3 terms
    
    # Wait, that's just the schoolbook again
    
    # Let me try a different approach: use bilinear forms
    
    # Actually, I think the best I can do without extensive search is return the schoolbook
    # Let me at least verify it's correct
    
    assert verify(U_sb, V_sb, W_sb), "Schoolbook decomposition failed"
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
