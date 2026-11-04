"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a lower rank decomposition using structured patterns
    # For 2x3 * 3x9, we can try rank = n*p + m*p + n*m - something
    # Or use a block-diagonal approach
    
    # Strategy: Use the fact that the tensor is C_{ik} = sum_j A_{ij} B_{jk}
    # We can try to find a decomposition with rank less than 54
    
    # One approach: treat the 3 in the middle as 2+1 and use Strassen-like ideas
    # But let's try a direct search for small rank patterns
    
    # For small dimensions, we can try to construct explicitly
    # Let's try rank = 40 (a guess)
    
    # Actually, let me think about this differently.
    # The tensor T_{ijk} = delta for the matmul is T[i,j,k] where 
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # So T_{ijk} = 1 if we're computing that term, 0 otherwise
    # More precisely, the matmul tensor has T[i,j,k] such that
    # C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k in range, which means it's the all-ones tensor?
    # No, that's not right. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 (it's the "matmul tensor" where each entry is 1)
    # And the decomposition is: T = sum_c U[i,c] V[j,c] W[k,c]
    # where U is n*m x rank, V is m*p x rank, W is p*n x rank
    
    # Wait, let me re-read the interface.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition is: for each output position (i,k), 
    # (A*B)[i,k] = sum_c U[i*m + ?, c] V[?, c] W[k*n + i, c]
    
    # Actually, the standard formulation:
    # Let A be n x m, B be m x p, C = A*B be n x p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T has T[i,j,k] = 1 (all ones) for the matmul operation
    # Decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] ... 
    # Hmm, this doesn't quite work with the given shapes.
    
    # Let me reconsider. The given shapes are:
    # U: (n*m, rank) - indexed by (i,j) where i in [0,n), j in [0,m)
    # V: (m*p, rank) - indexed by (j,k) where j in [0,m), k in [0,p)
    # W: (p*n, rank) - indexed by (k,i) where k in [0,p), i in [0,n)
    
    # So the decomposition is:
    # For each (i,j,k), the tensor entry is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And this should equal 1 for all i,j,k (since matmul tensor is all ones)
    
    # Wait, that's not right either. The matmul tensor T[i,j,k] represents
    # the coefficient of A[i,j]*B[j,k] in C[i,k], which is 1.
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # for all i in [0,n), j in [0,m), k in [0,p)
    
    # This is a system of n*m*p = 2*3*9 = 54 equations.
    # With rank r, we have 54*r unknowns (well, 3*54*r = 162*r unknowns total,
    # but they're coupled).
    
    # The schoolbook solution has r=54 with each equation satisfied by one term.
    # Can we do better?
    
    # Let's try r=36 as a starting guess and see if we can construct a solution.
    # Actually, let me just provide a valid lower-rank solution if I can find one.
    
    # For now, let's provide the schoolbook solution to be safe, but structured differently.
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
# EVOLVE-BLOCK-END
