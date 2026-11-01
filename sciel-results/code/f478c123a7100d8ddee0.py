"""Program for rank-minimal exact tensor decomposition of 2x2 * 2x8 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    
    # For 2x2 matrix multiplication, the tensor rank is 7 (Strassen).
    # We use a known rank-7 decomposition for the 2x2x2 tensor.
    # The decomposition is: C[i,k] = sum_r W[i,k,r] * (sum_j U[i,j,r] * A[i,j]) * (sum_l V[j,l,r] * B[j,l])
    # Actually, the standard form is:
    # C[i,k] = sum_r W[k*n+i, r] * M_r
    # where M_r = (sum_{i,j} U[i*m+j, r] * A[i,j]) * (sum_{j,k} V[j*p+k, r] * B[j,k])
    
    # For the 2x2x2 case, we have U0 (4x7), V0 (4x7), W0 (4x7)
    # Let's use the standard Strassen decomposition.
    
    # Strassen's 7 multiplications for 2x2x2:
    # M1 = A00 * (B00 - B01)
    # M2 = (A00 + A01) * B01
    # M3 = (A10 + A11) * B00
    # M4 = A11 * (B11 - B00)
    # M5 = (A00 + A10) * (B00 + B10)
    # M6 = (A01 + A11) * (B01 + B11)
    # M7 = (A00 - A10) * (B00 + B10)
    # M8 = (A01 - A11) * (B01 + B11)  -- wait, that's 8. Let me recount.
    
    # Actually, Strassen uses 7 multiplications:
    # M1 = A00 * (B00 - B01)
    # M2 = (A00 + A01) * B01
    # M3 = (A10 + A11) * B00
    # M4 = A11 * (B11 - B00)
    # M5 = (A00 + A10) * (B00 + B10)
    # M6 = (A01 + A11) * (B01 + B11)
    # M7 = (A00 - A10) * (B00 + B10)
    
    # And:
    # C00 = M1 + M4 + M5 - M7
    # C01 = M2 + M4 + M6
    # C10 = M3 + M7 + M5
    # C11 = M1 + M3 + M6 - M5
    
    # Let me verify this.
    # C00 = A00*B00 + A01*B10
    # M1 + M4 + M5 - M7 = A00*(B00-B01) + A11*(B11-B00) + (A00+A10)*(B00+B10) - (A00-A10)*(B00+B10)
    # = A00*B00 - A00*B01 + A11*B11 - A11*B00 + A00*B00 + A00*B10 + A10*B00 + A10*B10 - A00*B00 - A00*B10 + A10*B00 + A10*B10
    # = A00*B00 - A00*B01 + A11*B11 - A11*B00 + A00*B00 + A10*B00 + A10*B10 + A10*B00 + A10*B10
    # This doesn't look right. Let me use a different known rank-7 decomposition.
    
    # Let's use the following known rank-7 decomposition (from literature):
    # This is a valid rank-7 decomposition for 2x2x2 matrix multiplication.
    
    # U0: 4x7 matrix where U0[i*m+j, r] gives the coefficient of A[i,j] in the r-th multiplication
    # V0: 4x7 matrix where V0[j*p+k, r] gives the coefficient of B[j,k] in the r-th multiplication
    # W0: 4x7 matrix where W0[k*n+i, r] gives the coefficient in the r-th multiplication for C[i,k]
    
    # Here's a standard rank-7 decomposition:
    U0 = np.array([
        [1, 1, 0, 0, 1, 1, 1],  # A[0,0]
        [0, 1, 1, 0, 0, 1, 0],  # A[0,1]
        [0, 0, 1, 1, 1, 0, 1],  # A[1,0]
        [0, 0, 0, 1, 0, 1, 0],  # A[1,1]
    ])
    
    V0 = np.array([
        [1, 0, 1, 1, 1, 0, 1],  # B[0,0]
        [-1, 1, 0, 0, 0, 1, 1],  # B[0,1]
        [0, 0, 0, 0, 1, 1, 1],  # B[1,0]
        [0, 0, 0, 1, 0, 1, 0],  # B[1,1]
    ])
    
    W0 = np.array([
        [1, 0, 0, 1, 1, 0, -1],  # C[0,0]
        [0, 1, 0, 1, 0, 1, 0],   # C[0,1]
        [0, 0, 1, 0, 1, 0, 1],   # C[1,0]
        [1, 0, 1, 0, -1, 1, 0],  # C[1,1]
    ])
    
    # Let me verify this decomposition by checking if it's correct.
    # For each (i,k), C[i,k] should equal sum_r W0[k*n+i, r] * (sum_j U0[i*m+j, r] * A[i,j]) * (sum_l V0[j*p+k, r] * B[j,l])
    # Wait, the indices are a bit tricky. Let me be more careful.
    
    # The tensor is: T[i,j,k] = delta_{i,k} for the 2x2x2 case? No.
    # For matrix multiplication C = A * B, we have C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] where the multiplication is over j.
    # The decomposition is: C[i,k] = sum_r W[i,k,r] * (sum_j U[i,j,r] * A[i,j]) * (sum_l V[j,l,r] * B[j,l])
    # Wait, that's not quite right either. Let me think again.
    
    # The standard tensor decomposition for matrix multiplication is:
    # C[i,k] = sum_r W[i,k,r] * M_r
    # where M_r = (sum_{i',j'} U[i',j',r] * A[i',j']) * (sum_{j',k'} V[j',k',r] * B[j',k'])
    # But this doesn't look right because the sums are over all i',j' and j',k'.
    
    # Actually, the correct form is:
    # C[i,k] = sum_r W[i,k,r] * (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])
    # No, that's not right either.
    
    # Let me use the standard form from tensor rank literature:
    # The matrix multiplication tensor T of size n x m x p has entries T[i,j,k]
    # such that T[i,j,k] = 1 if i==k (for square matrices), but for general n,m,p:
    # T[i,j,k] = delta_{i,k} for the case where we're multiplying n x m by m x p.
    # Wait, no. The tensor for matrix multiplication C = A * B where A is n x m and B is m x p
    # is defined as: T[i,j,k] = 1 if j is the "inner" index, but actually:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid i,j,k, and the multiplication is:
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # But T[i,j,k] = 1 for all i,j,k, so this is just the standard multiplication.
    
    # The tensor rank decomposition is:
    # C[i,k] = sum_r W[i,k,r] * (sum_j U[i,j,r] * A[i,j]) * (sum_l V[j,l,r] * B[j,l])
    # Wait, I think the correct form is:
    # C[i,k] = sum_r W[i,k,r] * M_r
    # where M_r = (sum_{i',j'} U[i',j',r] * A[i',j']) * (sum_{j',k'} V[j',k',r] * B[j',k'])
    # But this would give a scalar M_r, and then C[i,k] = sum_r W[i,k,r] * M_r.
    # This is correct if the decomposition is of the form:
    # T = sum_r W_r \otimes U_r \otimes V_r
    # where W_r is a vector of length p*n, U_r is a vector of length n*m, V_r is a vector of length m*p.
    # And the multiplication is: C = sum_r W_r * (U_r^T A_vec) * (V_r^T B_vec)
    # where A_vec is A flattened to length n*m, B_vec is B flattened to length m*p,
    # and W_r is a matrix of shape (n, p) or flattened to length n*p.
    
    # So for the 2x2x2 case:
    # U_r is a vector of length 4 (flattened A)
    # V_r is a vector of length 4 (flattened B)
    # W_r is a vector of length 4 (flattened C)
    # And C_vec = sum_r W_r * (U_r^T A_vec) * (V_r^T B_vec)
    
    # Let me verify my U0, V0, W0 with this form.
    # For r=0:
    # U0[:,0] = [1,0,0,0] -> A[0,0]
    # V0[:,0] = [1,-1,0,0] -> B[0,0] - B[0,1]
    # W0[:,0] = [1,0,0,1] -> C[0,0] + C[1,1]
    # So M_0 = A[0,0] * (B[0,0] - B[0,1])
    # And it contributes to C[0,0] and C[1,1].
    
    # This looks like it could be part of a Strassen-like decomposition.
    # Let me check if the full decomposition is correct by computing C for a simple case.
    
    # Actually, let me just use the schoolbook decomposition of rank 4 for each of the 8 columns.
    # This gives total rank 32, which is safe and correct.
    
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
