import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Constructs a rank-50 decomposition for the 2x5 by 5x5 matrix multiplication tensor.
    
    This implementation uses a vectorized NumPy approach to generate the factor matrices 
    U, V, and W. It models the multiplication tensor T_{i,j,k} = \sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}.
    
    The 'schoolbook' decomposition assigns a unique rank index r to each (i, j, k) triple.
    This program constructs the factors by explicitly mapping the tensor indices to the 
    factor matrix entries, ensuring exact decomposition.
    """
    n, m, p = DIMENSIONS
    # Total dimensions for the matrix multiplication problem
    dim_a = n * m  # 10
    dim_b = m * p  # 25
    dim_c = p * n  # 10
    
    # Schoolbook rank is n*m*p
    rank = n * m * p
    
    # Initialize factor matrices
    U = np.zeros((dim_a, rank))
    V = np.zeros((dim_b, rank))
    W = np.zeros((dim_c, rank))
    
    # Generate the decomposition
    # For each (i, j, k), we create a rank slice that isolates the term A[i,j] * B[j,k]
    # This contributes to C[i,k]
    
    # We can vectorize the index generation
    # Indices i, j, k
    # i in [0, n), j in [0, m), k in [0, p)
    
    # Create meshgrid of indices
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten the indices to get the rank index for each combination
    # The rank index r corresponds to the linear index of (i, j, k) in the flattened tensor
    # We map each (i, j, k) to a unique column in U, V, W
    
    # Reshape indices to (rank, 1) to broadcast against the matrix dimensions
    # We need to assign 1s to specific positions in U, V, W for each r
    
    # U row index: i * m + j
    # V row index: j * p + k
    # W row index: k * n + i
    
    # Flatten the index arrays to match the rank dimension
    flat_i = i_idx.flatten()
    flat_j = j_idx.flatten()
    flat_k = k_idx.flatten()
    
    u_rows = (flat_i * m + flat_j)
    v_rows = (flat_j * p + flat_k)
    w_rows = (flat_k * n + flat_i)
    
    # The columns in U, V, W correspond to the linear index in the flattened (n,m,p) space
    cols = np.arange(rank)
    
    # Set the values to 1
    # U[u_rows, cols] = 1
    # V[v_rows, cols] = 1
    # W[w_rows, cols] = 1
    
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    # Optional: Self-verification to ensure correctness before returning
    # This helps debug and ensures the "research" aspect of verifying the decomposition
    # Construct the tensor from factors and compare with the known multiplication tensor
    
    # T_{i,j,k} is 1 if the term A[i,j]B[j,k] is in the sum for C[i,k]?
    # Actually, the matrix multiplication tensor T is defined such that 
    # T_{i,j,k} = 1 if the output element C[i,k] depends on A[i,j] and B[j,k].
    # In the standard bilinear decomposition, the tensor T_{i,j,k} represents the coefficient 
    # of A[i,j]B[j,k] in the expression for C[i,k].
    # For matrix multiplication, C[i,k] = \sum_j A[i,j] B[j,k].
    # So T_{i,j,k} = 1 if the indices match? No, T is a 3-tensor.
    # The decomposition is: C[i,k] = \sum_r W_{k*n+i, r} ( \sum_{i,j} U_{i*m+j, r} A[i,j] ) ( \sum_{j,k} V_{j*p+k, r} B[j,k] ) ?
    # No, the standard form is:
    # C[i,k] = \sum_r ( \sum_{j} U_{i,j,r} A[i,j] ) ( \sum_{j} V_{j,k,r} B[j,k] ) W_{i,k,r} ?
    # The interface specifies:
    # U shape (n*m, rank)
    # V shape (m*p, rank)
    # W shape (p*n, rank)
    
    # Let's check the contraction.
    # The product matrix C is p*n (5x2? No, 2x5 * 5x5 = 2x5. So C is 2x5. p*n = 5*2 = 10. W is 10xRank).
    # W[k*n+i, r]. Indexing W by k*n+i suggests output index (i,k) where i is row (0..n-1) and k is col (0..p-1).
    # Wait, n=2, p=5. Output is 2x5.
    # Row i, Col k.
    # W index: k*n + i.
    # If i=0, k=0 -> 0. i=1, k=0 -> 1. i=0, k=1 -> 2.
    # This is a column-major like ordering of the output matrix if we view it as (k, i).
    # Let's verify the decomposition logic.
    
    # Term r contributes to C[i,k] as:
    # val = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Summed over j? No, U and V are fixed vectors for a given r.
    # The standard rank-1 tensor is U_a V_b W_c.
    # Here, the tensor indices are (Input A index, Input B index, Output C index).
    # A index: i*m+j. B index: j*p+k. C index: k*n+i.
    
    # If we pick r corresponding to (i0, j0, k0):
    # U[i0*m+j0, r] = 1, others 0.
    # V[j0*p+k0, r] = 1, others 0.
    # W[k0*n+i0, r] = 1, others 0.
    
    # The term in the sum for C[i, k] from this r is:
    # (Sum over all A indices a of U[a, r] A[a]) * (Sum over all B indices b of V[b, r] B[b]) * W[c_out, r]
    # This form is NOT the standard bilinear form for matrix multiplication if U and V are just vectors.
    # The standard form is C = \sum_r (A \cdot U_r) (V_r \cdot B) W_r ?
    # Actually, the tensor decomposition T_{a,b,c} = \sum_r U_{a,r} V_{b,r} W_{c,r}.
    # The matrix multiplication tensor T has T_{i*m+j, j*p+k, k*n+i} = 1.
    # Wait, the index b (Input B) depends on j which is also in index a (Input A).
    # This is not a simple product of three separate vectors unless the j indices match.
    # In the schoolbook decomposition:
    # For a specific (i, j, k), we set:
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1
    # W[k*n+i, r] = 1
    # Then T_{a,b,c} = \sum_r U_{a,r} V_{b,r} W_{c,r}.
    # If a = i*m+j, b = j*p+k, c = k*n+i, then U[a,r]=1, V[b,r]=1, W[c,r]=1. Product is 1.
    # If a, b, c correspond to a different combination, say a' = i'*m+j', b'=j'*p+k', c'=k'*n+i'.
    # If r is fixed to (i,j,k), U[a',r] is 1 only if i'=i and j'=j.
    # V[b',r] is 1 only if j'=j and k'=k.
    # W[c',r] is 1 only if k'=k and i'=i.
    # So the product is 1 only if i'=i, j'=j, k'=k.
    # Thus, the sum reconstructs the tensor exactly.
    
    # The calculation above is valid.
    
    # Convert to lists of lists as required by the interface "integer/half-integer entries"
    # The prompt says "integer/half-integer entries". Integers are fine.
    # The interface expects a dictionary with keys 'U', 'V', 'W'.
    # The values should be lists (or arrays?) The seed uses lists.
    # "factor matrices of shapes... with integer/half-integer entries"
    # I will return lists of lists to be safe and match the seed structure.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Constructed rank 50 decomposition.")
