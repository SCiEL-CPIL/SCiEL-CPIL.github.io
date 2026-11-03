import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16
    
    # Strategy:
    # The multiplication tensor T_{i,j,k} = A_{i,j} B_{j,k} C_{k,i}
    # can be decomposed by summing over j.
    # Since A is 2x4 and B is 4x8, we have 4 values of j.
    # For a fixed j, the term is A_{i,j} B_{j,k}.
    # This looks like 4 independent "outer product" tensors?
    # No, because A_{i,j} depends on i and B_{j,k} depends on k.
    # The tensor is a sum of 4 rank-1 tensors if we fix i and k? No.
    # Let's look at the structure:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T is defined by T[i, j, k] = 1 if we are computing the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # Actually, the tensor T_{i,j,k} is 1 if the output entry (i,k) depends on A_{i,j} and B_{j,k}.
    # In standard matrix multiplication, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor has entries T[i, j, k] = 1 for all i, j, k?
    # No, the tensor T is defined such that sum_t U[i,j,t] V[j,k,t] W[k,i,t] = T[i,j,k].
    # For matrix multiplication, T[i,j,k] = 1 if the product A_{i,j} B_{j,k} contributes to C_{i,k}.
    # Yes, for every i, j, k, the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So T[i,j,k] = 1 for all i, j, k.
    
    # We want to decompose T into a sum of rank-1 tensors.
    # T = sum_t u_t ⊗ v_t ⊗ w_t
    # where u_t is a vector of length n*m, v_t length m*p, w_t length p*n.
    # The indices are flattened.
    # U[i*m+j, t], V[j*p+k, t], W[k*n+i, t].
    
    # Let's analyze the structure for a fixed j.
    # The terms involving A_{i,j} and B_{j,k} are for i in 0..1 and k in 0..7.
    # For a fixed j, the sub-tensor T_j has dimensions i (2) x k (8).
    # T_j[i, k] = 1.
    # This is a 2x8 matrix of ones.
    # We need to decompose the tensor T = sum_{j=0}^3 T_j (where T_j is supported on the j-th slice of the middle dimension).
    # Actually, the tensor T is 2x4x8.
    # T[i, j, k] = 1.
    # We can view this as 4 slices along the j-axis. Each slice is a 2x8 matrix of ones.
    # We need to decompose each 2x8 matrix of ones into a sum of rank-1 tensors (vectors u, v, w) such that
    # u is supported on { (i, j) }, v on { (j, k) }, w on { (k, i) }.
    # Wait, the rank-1 decomposition of a tensor T[i,j,k] is u[i,j] v[j,k] w[k,i].
    # For a fixed j, if we choose u[i,j] = delta_{j, j0} * u_i, v[j,k] = delta_{j, j0} * v_k, w[k,i] = w_ki.
    # Then the contribution to T[i, j0, k] is u_i v_k w_ki.
    # We want this to be 1 for all i, k.
    # So we need to decompose the 2x8 matrix of ones (M_{i,k} = 1) as a sum of rank-1 matrices u_i v_k?
    # No, w_ki is also a variable.
    # The term is u_i * v_k * w_ki.
    # If we fix u_i and v_k, we need w_ki to complete the decomposition.
    # Actually, we can just set w_ki = 1 / (u_i v_k) if u_i v_k != 0.
    # But we need the sum over t to equal 1.
    
    # Let's use the Karatsuba trick for the 2x8 matrix of ones.
    # We want to represent the all-ones matrix J_{2x8} as a sum of rank-1 matrices?
    # No, the tensor is 3D.
    # Let's look at the "scalar channel" idea from the prompt.
    # "Partition the 4-dimensional intermediate space into four independent scalar channels... apply a rank-3 decomposition to each"
    
    # Let's reconsider the decomposition of T[i,j,k]=1.
    # We can sum over j.
    # For each j, we need to decompose the 2x8 block of ones.
    # Let's try to decompose a single 2x8 block of ones (fixed j) into rank-1 tensors.
    # A rank-1 tensor is u_i * v_k * w_ki.
    # If we choose u_i = 1, v_k = 1, then we need w_ki = 1.
    # This gives one rank-1 tensor: u = 1 (for all i), v = 1 (for all k), w = 1 (for all k,i).
    # Does this work?
    # Contribution to T[i, j, k] is 1 * 1 * 1 = 1.
    # So a single rank-1 tensor covers the entire 2x8 block!
    # If so, the total rank would be 4 (one for each j).
    # Let's check if this is valid.
    # U[i*m+j, t] = 1 for all i, for t=j.
    # V[j*p+k, t] = 1 for all k, for t=j.
    # W[k*n+i, t] = 1 for all k, i, for t=j.
    # Sum over t:
    # For a given (i, j, k), the term is non-zero only if t=j.
    # Value = U[i*m+j, j] * V[j*p+k, j] * W[k*n+i, j] = 1 * 1 * 1 = 1.
    # For t != j, U[i*m+j, t] = 0 (since U is zero except for column j? No, U has columns t=0..3).
    # If we define:
    # For t in 0..3:
    #   U[i*m+t, t] = 1 for all i.
    #   V[t*p+k, t] = 1 for all k.
    #   W[k*n+i, t] = 1 for all k, i.
    # Then for a specific (i, j, k):
    # Sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t]
    # The term is non-zero only if t=j (because U[i*m+j, t] is 0 if t!=j? No, U[i*m+j, t] is defined as 1 if t==j, 0 otherwise?
    # My definition above: U[i*m+t, t] = 1. This means U[row, col] is 1 only if row % m == col.
    # So U[i*m+j, t] is 1 if (i*m+j) % m == t => j == t.
    # So yes, U[i*m+j, t] is 1 if t=j, 0 otherwise.
    # Similarly V[j*p+k, t] is 1 if (j*p+k) % p == t => k % p == t? No.
    # V index is j*p+k. V[row, col] is 1 if row % p == col?
    # In my definition: V[t*p+k, t] = 1. So V[row, col] is 1 if row % p == col.
    # For V[j*p+k, t], row = j*p+k. row % p = k.
    # So V[j*p+k, t] is 1 if k == t.
    # But we need t=j for the U term to be non-zero.
    # So if t=j, we need V[j*p+k, j] to be 1.
    # V[j*p+k, j] is 1 if (j*p+k) % p == j => k % p == j.
    # This is not always true. k is 0..7, j is 0..3.
    # So this simple construction fails because the V and W factors don't align correctly for all k.
    
    # The "scalar channel" idea suggests treating each (i,k) pair independently?
    # Or each j independently?
    
    # Let's look at the rank-3 decomposition for 2x1 * 1x2.
    # Matrix A is 2x1 (vector a), B is 1x2 (vector b).
    # Product C is 2x2. C_{i,k} = a_i b_k.
    # Tensor T_{i, 0, k} = 1 (since j=0 is the only intermediate).
    # We want to decompose T_{i, 0, k} = 1.
    # This is a 2x1x2 tensor of ones.
    # Can we decompose it into 3 rank-1 tensors?
    # T = sum_{t=0}^2 u_t[i] v_t[0] w_t[k,i].
    # Note: v_t[0] is a scalar. Let v_t = v_t[0].
    # u_t[i] is a vector of length 2.
    # w_t[k,i] is a matrix 2x2.
    # We need sum_t u_t[i] v_t w_t[k,i] = 1.
    
    # Let's try to construct such a decomposition.
    # This is equivalent to decomposing the all-ones 2x2 matrix into a sum of 3 rank-1 matrices?
    # No, w_t[k,i] is not rank-1 necessarily.
    # But the term is u_t[i] * v_t * w_t[k,i].
    # Let x_t[i] = u_t[i] v_t. Then sum_t x_t[i] w_t[k,i] = 1.
    # This looks like we can choose w_t[k,i] to be simple.
    
    # Actually, there is a known result: The tensor for 2x1 * 1x2 multiplication has rank 3.
    # Why?
    # C_{00} = a_0 b_0
    # C_{01} = a_0 b_1
    # C_{10} = a_1 b_0
    # C_{11} = a_1 b_1
    # This is just the outer product of a and b.
    # The tensor is T_{i, 0, k} = 1.
    # Rank of this tensor?
    # It is the Kronecker product of vector of ones (len 2) and vector of ones (len 2)?
    # No, T_{i,k} = 1.
    # This tensor is separable: T_{i,k} = 1 * 1.
    # So it is rank 1!
    # u[0]=1, u[1]=1. v[0]=1. w[0,0]=1, w[0,1]=1, w[1,0]=1, w[1,1]=1.
    # Check: sum_t u[i] v[0] w[k,i] = 1 * 1 * 1 = 1.
    # So the rank of 2x1 * 1x2 is 1.
    
    # If the rank of each "channel" is 1, and we have 4 channels (j=0,1,2,3), the total rank is 4.
    # Let's verify if we can construct a rank-4 decomposition for the whole 2x4*4x8 problem.
    
    # Proposed Rank 4 Construction:
    # For each j in 0..3, we have a rank-1 tensor.
    # t = j.
    # U[i*m+j, t] = 1 for all i.
    # V[j*p+k, t] = 1 for all k.
    # W[k*n+i, t] = 1 for all k, i.
    
    # Let's re-evaluate the condition for V.
    # We need V[j*p+k, t] to be 1 when t=j.
    # In the proposed construction, V[row, col] is 1 if row % p == col.
    # row = j*p+k. row % p = k.
    # So V[j*p+k, j] is 1 if k == j.
    # This is only true for specific k.
    # We need V[j*p+k, j] = 1 for ALL k.
    # So we should define V differently.
    # We want V[j*p+k, t] = 1 if t=j, for all k.
    # This means V[row, col] = 1 if floor(row/p) == col.
    # i.e., V is 1 on the diagonal of the "j-blocks".
    # Similarly for U: U[i*m+j, t] = 1 if t=j.
    # row = i*m+j. floor(row/m) = i? No, row % m = j.
    # We want U[row, col] = 1 if row % m == col.
    # This matches the previous U definition.
    
    # Let's check W.
    # We need W[k*n+i, t] = 1 if t=j?
    # No, W depends on t.
    # For a fixed t=j, we need W[k*n+i, j] = 1 for all k, i.
    # So W[row, col] = 1 if col == j?
    # i.e., W is 1 in the columns corresponding to j.
    
    # Let's test this construction:
    # For t in 0..3:
    #   U[i*m+t, t] = 1 for all i. (0 elsewhere)
    #   V[t*p+k, t] = 1 for all k. (0 elsewhere)
    #   W[k*n+i, t] = 1 for all k, i. (0 elsewhere)
    
    # Check sum for T[i, j, k]:
    # Sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t]
    # Term t is non-zero only if:
    # 1. U[i*m+j, t] != 0 => t == j (since U[row, t] is 1 only if row % m == t, and (i*m+j)%m = j).
    # 2. V[j*p+k, t] != 0 => t == j (since V[row, t] is 1 only if floor(row/p) == t, and floor((j*p+k)/p) = j).
    # 3. W[k*n+i, t] != 0 => t == j (since W[row, t] is 1 only if col == t).
    # So only t=j contributes.
    # Value = U[i*m+j, j] * V[j*p+k, j] * W[k*n+i, j]
    # = 1 * 1 * 1 = 1.
    # This works!
    
    # So the rank is 4.
    
    # Wait, is this too good to be true?
    # Let's check the dimensions.
    # U is 8x4.
    # V is 32x4.
    # W is 16x4.
    
    # Let's double check the index mappings.
    # U index: i*m + j. m=4.
    # U[i*4+j, t].
    # Condition: U[row, t] = 1 if row % 4 == t.
    # For a specific (i, j), row = i*4+j. row % 4 = j.
    # So U[i*4+j, t] = 1 if t=j. Correct.
    
    # V index: j*p + k. p=8.
    # V[j*8+k, t].
    # Condition: V[row, t] = 1 if floor(row/8) == t.
    # For a specific (j, k), row = j*8+k. floor(row/8) = j.
    # So V[j*8+k, t] = 1 if t=j. Correct.
    
    # W index: k*n + i. n=2.
    # W[k*2+i, t].
    # Condition: W[row, t] = 1 if t is in the column?
    # I said "W is 1 in the columns corresponding to j".
    # So W[row, t] = 1 for all row if t=j.
    # i.e., W[row, t] = 1 if t is the column index.
    # So W[row, t] = 1 for all row, for a specific t.
    # This means W is a matrix where each column t is all 1s.
    # So W[k*2+i, t] = 1 for all k, i, t.
    # Wait, if W[k*2+i, t] = 1 for all t, then the condition "W is non-zero only if t=j" is false.
    # If W is all 1s, then W[k*2+i, t] = 1 for all t.
    # Then the sum becomes:
    # Sum_t U[i*m+j, t] V[j*p+k, t] * 1
    # = U[i*m+j, j] V[j*p+k, j] * 1 (since U and V are 0 for t!=j)
    # = 1 * 1 = 1.
    # So W can be all 1s?
    # Let's check if W can be all 1s.
    # If W is all 1s, then for any t, W is 1.
    # The product U*V is non-zero only when t=j.
    # So the sum is indeed 1.
    
    # So the construction is:
    # U[i*m+j, t] = 1 if t==j, else 0.
    # V[j*p+k, t] = 1 if t==j, else 0.
    # W[k*n+i, t] = 1 for all t.
    
    # Let's verify the shapes.
    # U: (8, 4).
    # V: (32, 4).
    # W: (16, 4).
    
    # This yields a rank of 4.
    
    # Is there any constraint I missed?
    # "integer/half-integer entries" - 1 is integer.
    # "exactly decompose" - Yes, 1=1.
    
    # Why did the previous program use rank 56?
    # Maybe they didn't realize this simple decomposition?
    # Or maybe I am misinterpreting the tensor definition.
    
    # Let's re-read the tensor definition.
    # "U, V, W factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "U[i*m+j, t] V[j*p+k, t] W[k*n+i, t]"
    # Sum over t.
    # Target: Matrix multiplication tensor.
    # T[i,j,k] = 1.
    
    # My construction gives T[i,j,k] = 1.
    
    # Let's try to verify with a small example in code logic.
    # n=2, m=4, p=8.
    # i=0, j=1, k=3.
    # U[0*4+1, t] = U[1, t]. Non-zero only if t=1. U[1,1]=1.
    # V[1*8+3, t] = V[11, t]. Non-zero only if t=1. V[11,1]=1.
    # W[3*2+0, t] = W[6, t]. W[6, t]=1 for all t.
    # Sum_t U[1,t] V[11,t] W[6,t]
    # t=0: 0 * 0 * 1 = 0
    # t=1: 1 * 1 * 1 = 1
    # t=2: 0 * 0 * 1 = 0
    # t=3: 0 * 0 * 1 = 0
    # Sum = 1. Correct.
    
    # What if i=1, j=2, k=5?
    # U[1*4+2, t] = U[6, t]. Non-zero only if t=2. U[6,2]=1.
    # V[2*8+5, t] = V[21, t]. Non-zero only if t=2. V[21,2]=1.
    # W[5*2+1, t] = W[11, t]. W[11, t]=1.
    # Sum = 1. Correct.
    
    # It seems the rank is indeed 4.
    
    # Let's construct the matrices.
    
    best_rank = 4
    best_U = np.zeros((U_dim, best_rank))
    best_V = np.zeros((V_dim, best_rank))
    best_W = np.zeros((W_dim, best_rank))
    
    for t in range(4):
        # U: column t is 1 at positions i*m+t for i in 0..1
        for i in range(n):
            best_U[i*m + t, t] = 1
            
        # V: column t is 1 at positions t*p+k for k in 0..7
        for k in range(p):
            best_V[t*p + k, t] = 1
            
        # W: column t is 1 at all positions
        for idx in range(W_dim):
            best_W[idx, t] = 1
            
    # Verify
    def verify_decomp(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(U.shape[1]):
                        val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    if not verify_decomp(best_U, best_V, best_W):
        # Fallback to schoolbook if verification fails (should not happen)
        best_rank = n * m * p
        best_U = np.zeros((U_dim, best_rank))
        best_V = np.zeros((V_dim, best_rank))
        best_W = np.zeros((W_dim, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j, col] = 1
                    best_V[j*p+k, col] = 1
                    best_W[k*n+i, col] = 1
                    col += 1

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

