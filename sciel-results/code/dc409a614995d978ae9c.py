"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # We need to decompose the 3x3x7 matrix multiplication tensor.
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k] for i in 0..2, k in 0..6
    # Tensor T[i,j,k] = 1 for all i,j,k
    # U shape: (n*m, rank) = (9, rank), indexed by (i*m+j) = (i*3+j)
    # V shape: (m*p, rank) = (21, rank), indexed by (j*p+k) = (j*7+k)
    # W shape: (p*n, rank) = (21, rank), indexed by (k*n+i) = (k*3+i)
    
    # Strategy: Use a verified rank-23 decomposition for 3x3x3,
    # then extend to 3x3x7 by handling the extra 4 columns.
    # 
    # For 3x3x3: C[i,k] = sum_j A[i,j]*B[j,k], k=0,1,2
    # For 3x3x7: C[i,k] = sum_j A[i,j]*B[j,k], k=0,1,2,3,4,5,6
    #
    # Approach: Split into 3x3x3 (k=0,1,2) + 3x3x4 (k=3,4,5,6)
    # But we can do better by finding a joint decomposition.
    #
    # Simplest correct approach: schoolbook gives rank 63.
    # Better: use rank-23 for the first 3 columns, and handle remaining 4 columns.
    #
    # Actually, let's use a direct approach: find a low-rank decomposition
    # of the full 3x3x7 tensor.
    #
    # The tensor T has T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3,4,5,6}
    # This is a rank-1 tensor in the j dimension! T[i,j,k] = 1 * 1 * 1
    # Wait no - the tensor is T[i,j,k] which represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # For matrix multiplication, T[i,j,k] = 1 for ALL i,j,k. So it's the all-ones tensor.
    # The rank of the all-ones tensor of size n x m x p is... well, it's not just 1
    # because we need the decomposition to respect the matrix multiplication structure.
    #
    # Actually wait. Let me re-read the problem. The matrix multiplication tensor
    # for multiplying an n x m matrix by an m x p matrix has entries:
    # T[i, j, k] = 1 if the (i,k)-th output depends on A[i,j]*B[j,k]
    # For standard matrix multiplication, T[i,j,k] = 1 for ALL valid i,j,k.
    # So yes, it's the all-ones tensor of shape (n, m, p) = (3, 3, 7).
    #
    # The rank of the all-ones tensor of size a x b x c is min(a*b, b*c, a*c) in general?
    # No, that's not right either. The rank of the all-ones tensor 1_{a,b,c} is actually
    # the minimum number of rank-1 tensors that sum to it.
    #
    # A rank-1 tensor u ⊗ v ⊗ w has entries u[i]*v[j]*w[k].
    # We need sum_t u_t[i] * v_t[j] * w_t[k] = 1 for all i,j,k.
    #
    # This means: for each (i,j,k), the sum equals 1.
    # One way: use the fact that 1 = sum of products.
    #
    # Actually, the rank of the all-ones tensor 1_{a,b,c} is known.
    # For 1_{3,3,7}: we can write it as a sum of rank-1 tensors.
    # 
    # Note: 1_{a,b,c} = (1_a) ⊗ (1_b) ⊗ (1_c) which is rank 1!
    # Wait, that's true! The all-ones tensor IS rank 1!
    # u[i] = 1 for all i, v[j] = 1 for all j, w[k] = 1 for all k.
    # Then u[i]*v[j]*w[k] = 1*1*1 = 1 for all i,j,k.
    #
    # So the rank is 1?! But that can't be right for matrix multiplication...
    #
    # OH WAIT. I'm confusing myself. The matrix multiplication tensor is NOT the all-ones tensor
    # in the way I'm thinking. Let me re-read the interface.
    #
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # U has shape (n*m, rank) = (9, rank)
    # V has shape (m*p, rank) = (21, rank) 
    # W has shape (p*n, rank) = (21, rank)
    #
    # The decomposition says: for each rank term t:
    #   U[:,t] is a vector in R^{n*m}, V[:,t] is a vector in R^{m*p}, W[:,t] is a vector in R^{p*n}
    #   The contribution to the multiplication is: 
    #   For output element C[i,k] (which is at position k*n+i in W):
    #     C[i,k] = sum_t U[i*m+j] * V[j*p+k] * W[k*n+i]  ... NO
    #
    # Actually, the standard tensor decomposition for matrix multiplication is:
    # C = A * B means C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] where T[i,j,k] = 1
    #
    # The rank-r decomposition means:
    # T[i,j,k] = sum_{t=1}^{r} U[i,j,t] * V[j,k,t] * W[k,i,t]
    # where U is reshaped to (n,m,r), V to (m,p,r), W to (p,n,r)
    #
    # So we need: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1 for all i,j,k
    #
    # This IS the all-ones tensor decomposition! And as I noted, it's rank 1:
    # U[i,j,t] = 1, V[j,k,t] = 1, W[k,i,t] = 1 for t=1.
    #
    # But wait, that would mean rank 1 suffices for ANY matrix multiplication, which is absurd.
    # The issue is that the decomposition must be "valid" in the sense that it correctly
    # computes the matrix product for ALL input matrices, not just that the tensor entries match.
    #
    # Hmm, but if T[i,j,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = 1 for all i,j,k,
    # then C[i,k] = sum_j (sum_t U[i,j,t]*V[j,k,t]*W[k,i,t]) A[i,j] B[j,k]
    # = sum_j 1 * A[i,j] B[j,k] = sum_j A[i,j] B[j,k]
    # So it DOES work! The rank is indeed 1?!
    #
    # No wait, I think I'm making an error. Let me re-examine.
    #
    # The standard formulation: A rank-r decomposition of matrix multiplication
    # (n x m) * (m x p) -> (n x p) consists of r triples of matrices
    # (U_t, V_t, W_t) where U_t is n x m, V_t is m x p, W_t is p x n,
    # such that for all A, B:
    #   A*B = sum_t (U_t * A) * (V_t * B) * W_t  ... no, that's not right either.
    #
    # Let me look at this more carefully. The correct formulation:
    # We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # A rank-r decomposition gives r terms such that:
    # C[i,k] = sum_{t=1}^r (sum_j U_t[i,j] A[i,j]) * (sum_j V_t[j,k] B[j,k]) * W_t[k,i]
    #
    # No, that's not standard either. Let me think about this differently.
    #
    # The standard bilinear decomposition:
    # C[i,k] = sum_t (sum_j u_t[i,j] A[i,j]) * (sum_j v_t[j,k] B[j,k]) * w_t[i,k]
    #
    # Hmm, I think the correct formulation based on the interface is:
    # U has shape (n*m, r), so U[idx, t] where idx = i*m+j
    # V has shape (m*p, r), so V[idx, t] where idx = j*p+k  
    # W has shape (p*n, r), so W[idx, t] where idx = k*n+i
    #
    # The computation is:
    # C[i,k] = sum_t U[i*m + j, t] * V[j*p + k, t] * W[k*n + i, t] ... summed over j too?
    #
    # No. I think the correct interpretation is:
    # For each term t, we compute:
    #   intermediate_u = U[:,t] reshaped to (n,m) dotted with A: gives n-vector
    #   intermediate_v = V[:,t] reshaped to (m,p) dotted with B: gives p-vector
    #   Then C[i,k] += intermediate_u[i] * intermediate_v[k] * W[k*n+i, t]
    #
    # So: C[i,k] = sum_t (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k]) * W[k*n+i, t]
    #
    # For this to equal sum_j A[i,j] B[j,k] for all A, B:
    # We need: sum_t (sum_j U[i*m+j,t] A[i,j]) (sum_j' V[j'*p+k,t] B[j',k]) W[k*n+i,t] = sum_j A[i,j] B[j,k]
    #
    # Expanding: sum_t W[k*n+i,t] sum_j sum_j' U[i*m+j,t] V[j'*p+k,t] A[i,j] B[j',k]
    # = sum_j A[i,j] B[j,k]
    #
    # This must hold for all A, B, so:
    # sum_t W[k*n+i,t] U[i*m+j,t] V[j'*p+k,t] = delta_{j,j'}
    #
    # So the tensor equation is: sum_t U[i,j,t] V[j',k,t] W[k,i,t] = delta_{j,j'}
    # where I've used the trace convention W[k*n+i] for the (k,i) entry.
    #
    # This is DIFFERENT from the all-ones tensor! The condition is:
    # sum_t U[i,j,t] V[j',k,t] W[k,i,t] = delta_{j,j'}
    #
    # Not = 1. It's = delta_{j,j'}. So it's not the all-ones tensor.
    #
    # For the 3x3x3 case: sum_t U[i,j,t] V[j',k,t] W[k,i,t] = delta_{j,j'} for i,j,j',k in {0,1,2}
    #
    # This is a system of 3*3*3*3 = 81 equations in 3*3*3*r = 27r unknowns.
    #
    # OK so my earlier analysis was wrong. Let me just implement the schoolbook solution
    # which is guaranteed correct, and then try to find a better decomposition.
    
    # Schoolbook: rank = n*m*p = 3*3*7 = 63
    # For each (i,j,k), one term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    
    rank = n * m * p  # 63
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
