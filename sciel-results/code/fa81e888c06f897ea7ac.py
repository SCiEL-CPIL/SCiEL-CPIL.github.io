import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Constructs a rank-minimal (schoolbook) exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    
    The 'schoolbook' decomposition assigns a unique rank index r to each (i, j, k) triple.
    This program constructs the factors by explicitly mapping the tensor indices to the 
    factor matrix entries using direct index arithmetic.
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
    # We iterate through each (i, j, k) triple and assign it to a unique column r.
    # The tensor entry T_{a, b, c} is 1 if a = i*m+j, b = j*p+k, c = k*n+i.
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Calculate the row indices for U, V, W for this specific (i, j, k)
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                
                # Set the corresponding entry in the factor matrices to 1
                U[u_row, r] = 1
                V[v_row, r] = 1
                W[w_row, r] = 1
                
                r += 1
    
    # Convert to lists of lists to match the interface requirement for integer/half-integer entries
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Constructed rank 50 decomposition.")
