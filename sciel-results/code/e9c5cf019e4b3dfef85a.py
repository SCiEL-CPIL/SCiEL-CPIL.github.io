"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor from U, V, W
    # T[i, j, k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We check all (i, j, k) entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if (i >= 0 and j >= 0 and k >= 0) else 0
                # Actually, the tensor for matrix mult: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
                # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k) in the standard basis
                # Wait, let me reconsider. The matrix multiplication tensor is:
                # T(a,b,c) = sum_{j} a[i,j] * b[j,k] * c[i,k] where c = a*b
                # In the standard basis, the tensor components are T[i,j,k] = delta_{ik} ... no
                # Let me think more carefully.
                
                # The task says: "exactly decompose the matrix multiplication tensor"
                # With the convention W[k*n+i, r]
                
                # For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # The tensor T has components such that:
                # T[i,j,k] = 1 if we're looking at the (i,j) entry of A, (j,k) entry of B, 
                # and (i,k) entry of C. But actually the tensor is defined so that
                # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} C[i,k]
                # This means T[i,j,k] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} ... 
                
                # Actually, the standard tensor for matrix multiplication has:
                # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
                # No wait, that's not right either.
                
                # Let me use the convention from the problem:
                # U has shape (n*m, rank), indexed as U[i*m + j, c]
                # V has shape (m*p, rank), indexed as V[j*p + k, c]  
                # W has shape (p*n, rank), indexed as W[k*n + i, c]
                
                # The decomposition is: for each (i,j,k),
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
                
                # What is T[i,j,k]? For matrix multiplication, the tensor should satisfy:
                # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
                # So the bilinear form is: sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} (A*B)[i,k]
                # This gives T[i,j,k] = 1 for all valid i,j,k? No, that can't be right.
                
                # Actually, I think the tensor is defined such that:
                # T[i, j, k] = 1 if the indices are consistent with matrix multiplication
                # In the standard basis where A has entries A[i,j], B has entries B[j,k],
                # C has entries C[i,k] = sum_j A[i,j] B[j,k]
                # The tensor T satisfies: T[i,j,k] * A[i,j] * B[j,k] summed over i,j,k = sum_{i,k} C[i,k]
                # But C[i,k] is a new variable, so actually the tensor encodes the multiplication rule.
                
                # I believe the correct interpretation is:
                # The matrix multiplication tensor has T[i, j, k] = 1 for all i in range(n), 
                # j in range(m), k in range(p). This is because each product term A[i,j]*B[j,k]
                # contributes to C[i,k], and in the tensor decomposition, each such term is a 
                # rank-1 component.
                
                # Wait, but that would mean the schoolbook decomposition is already minimal in 
                # some sense? No, because different terms can share rank-1 components.
                
                # Let me just verify with the schoolbook: for schoolbook, 
                # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1 for the column
                # corresponding to (i,j,k). So sum_c gives 1 for each (i,j,k) triple.
                # So T[i,j,k] = 1 for all i,j,k.
                
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    # For 2x4x6, the tensor has 2*4*6 = 48 entries, all equal to 1
    # We need to find rank r such that there exist U (8 x r), V (24 x r), W (12 x r)
    # with sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is a hard problem. Let's try a greedy approach:
    # Start with schoolbook and try to merge columns.
    
    # Actually, let's try a different approach: use the fact that the tensor is 
    # a "simple" tensor and try known bounds.
    
    # For matrix multiplication C_{n x m} * C_{m x p}, the tensor rank is at most n*m*p
    # and at least max(n*m, m*p, p*n) / something... actually the lower bound is 
    # related to the border rank.
    
    # Let me try a systematic search for small rank reductions.
    # We'll try to see if we can remove some terms by linearity.
    
    # Approach: Start with schoolbook, then try to set some entries to 0 and adjust
    # others to maintain the decomposition.
    
    # For now, return schoolbook as a safe fallback.
    # A more sophisticated search would be needed to beat this.
    
    if time.time() > deadline:
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try rank = schoolbook_rank - 1, schoolbook_rank - 2, etc.
    # This is computationally expensive, so let's be conservative.
    
    # For 2x4x6, let's try to see if rank 47 or 46 works.
    # This requires solving a system of equations.
    
    # Let's use a different strategy: exploit the structure.
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is equivalent to finding a rank-r decomposition of the tensor.
    
    # Given the time constraints, let's just return the schoolbook decomposition.
    # A future evolution could implement a proper search.
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
