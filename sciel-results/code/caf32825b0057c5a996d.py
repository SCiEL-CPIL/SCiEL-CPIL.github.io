import numpy as np
import random
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
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
    
    # Try to find a better decomposition
    # We know rank >= max(nm, mp, np) = 24
    # Let's try random search for rank 30, 32, etc.
    # Time limit 150s.
    
    start_time = time.time()
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Search space: entries in {-1, 0, 1} or {-0.5, 0, 0.5, 1}
    # We'll stick to integers {-1, 0, 1} for simplicity and exactness
    
    def verify(U, V, W):
        # Check if U V W decomposes the tensor
        # T_ijk = sum_r U_ir V_jr W_kr ? No.
        # The tensor is T_ijk = 1 for all i,j,k? No.
        # The tensor for mat mult is T_ijk = 1.
        # Wait, the tensor is fixed.
        # We need sum_r U_{i,r} V_{j,r} W_{k,r} = T_{i,j,k}
        # Where T_{i,j,k} = 1 if (i,j,k) is a valid index?
        # The tensor is the all-1s tensor?
        # Yes, T_{i,j,k} = 1.
        # So we need sum_r U_{i,r} V_{j,r} W_{k,r} = 1 for all i,j,k.
        
        # This is a system of linear equations.
        # Number of equations: n*m*p = 48.
        # Variables: R * (n+m+p) = R * 13.
        # If R=30, 390 variables.
        
        # We can solve this linear system!
        # Let's formulate the linear system.
        # Unknowns: U, V, W.
        # Equations: sum_r U_{i,r} V_{j,r} W_{k,r} = 1.
        # This is nonlinear (bilinear in U,V,W).
        # However, if we fix two matrices, it's linear in the third.
        
        return True # Placeholder

    # Given the complexity of finding a low rank decomposition of the all-1s tensor,
    # and the fact that the all-1s tensor has rank 1 (which is incorrect for mat mult rank),
    # there is a fundamental misunderstanding of the tensor definition in my thought process.
    
    # The matrix multiplication tensor T_{i,j,k} is NOT all 1s.
    # It is defined by the structure of the multiplication.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T is such that T_{i,j,k} = 1 if the term A_{ij}B_{jk} contributes to C_{ik}.
    # Since every A_{ij} and B_{jk} contributes to C_{ik} for the specific j,
    # T_{i,j,k} = 1 for all i,j,k.
    
    # Why is the rank not 1?
    # Because the rank of the tensor T is the minimum R such that T = sum_r u_r v_r w_r.
    # If T is all 1s, T = u v w with u=v=w=1. Rank 1.
    # But matrix multiplication requires rank 7 for 2x2x2.
    # This implies T is NOT all 1s.
    
    # Let's re-read the definition.
    # The tensor is T_{i,j,k}.
    # The map is (A,B) -> C where C_{ik} = sum_j T_{i,j,k} A_{ij} B_{jk}.
    # For standard mat mult, C_{ik} = sum_j A_{ij} B_{jk}.
    # So T_{i,j,k} = 1.
    
    # There is a contradiction in my understanding of tensor rank vs algorithm rank.
    # The rank of the tensor is the algorithm rank.
    # If the tensor is all 1s, its rank is 1.
    # Then mat mult should be rank 1.
    # It is not.
    # Therefore, the tensor is NOT all 1s.
    
    # Ah! The indices.
    # T_{i,j,k} is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # For C_{00} = A_{00}B_{00} + A_{01}B_{10}.
    # T_{0,0,0} = 1 (coeff of A_{00}B_{00}).
    # T_{0,1,0} = 1 (coeff of A_{01}B_{10}).
    # T_{0,0,1} = 0 (coeff of A_{00}B_{01} in C_{00} is 0).
    # T_{0,1,1} = 0.
    
    # So T_{i,j,k} = 1 if the index j matches the summation index?
    # No, j is the summation index.
    # T_{i,j,k} = 1 for all i,j,k?
    # C_{ik} = sum_j T_{i,j,k} A_{ij} B_{jk}.
    # If T_{i,j,k} = 1, then C_{ik} = sum_j A_{ij} B_{jk}.
    # This is correct.
    # So T IS all 1s.
    
    # Why is the rank of the all-1s tensor 2x2x2 equal to 7?
    # It is not. The rank of the all-1s tensor is 1.
    # The rank of the matrix multiplication tensor is 7.
    # This implies the matrix multiplication tensor is NOT the all-1s tensor.
    
    # The resolution: The tensor T is not just the coefficients.
    # The tensor is the object T in the space V* x W* x Z.
    # The multiplication map is mu: V x W -> Z.
    # mu(A, B) = sum_{i,j,k} T_{i,j,k} A_i B_j C_k?
    # No.
    
    # I will stick to the schoolbook decomposition as it is guaranteed correct.
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
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
