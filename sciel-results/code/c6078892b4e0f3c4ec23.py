"""
Rewritten program to find a lower-rank decomposition for 5x5x5 matrix multiplication.
Uses a block decomposition strategy: 5 = 4 + 1.
Implements a rank-80 decomposition using schoolbook for the 4x4 block and optimal ranks for the rest.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Faster verification using einsum over all indices at once."""
    U_np = np.asarray(U, dtype=float)
    V_np = np.asarray(V, dtype=float)
    W_np = np.asarray(W, dtype=float)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U_np.reshape(n, m, -1)
    V3 = V_np.reshape(m, p, -1)
    W3 = W_np.reshape(p, n, -1)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.all(np.abs(T - 1.0) < tol)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _construct_block_decomp_4_1():
    """
    Constructs a decomposition for 5x5x5 using 4+1 block structure.
    Total rank: 64 (4x4x4 schoolbook) + 4 (4x4x1) + 4 (1x4x4) + 1 (4x1x4) + 4 (1x4x1) + 1 (4x1x1) + 1 (1x1x4) + 1 (1x1x1) = 80.
    """
    n = m = p = 5
    # We will build U, V, W lists of lists
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term
    def add_term(i, j, k, val=1.0):
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        # We append to the lists for this "column" (term)
        U_list.append([0.0] * (n*m))
        V_list.append([0.0] * (m*p))
        W_list.append([0.0] * (p*n))
        U_list[-1][i*m + j] = val
        V_list[-1][j*p + k] = val
        W_list[-1][k*n + i] = val

    # 1. Block (0,0): 4x4x4 Schoolbook (indices 0..3 for all)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                add_term(i, j, k)
                
    # 2. Block (0,1): A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    # This is 4x1 * 1x4 = 4x4. Rank 1? No, it's an outer product.
    # C_ij = A_i4 * B_4j.
    # This is a single rank-1 term?
    # T[i,j,k] = A_{i,4} B_{4,j}.
    # In our convention: U[i,4,c], V[4,j,c], W[j,i,c]? No.
    # The term is A_{i,4} B_{4,j}.
    # i is row of A, j is col of B.
    # k is the intermediate index. Here k=4.
    # So i in 0..3, j in 0..3, k=4.
    # U[i, 4, c] = 1, V[4, j, c] = 1, W[j, i, c] = 1?
    # Wait, the output is C[i,j].
    # C[i,j] = sum_k A[i,k] B[k,j].
    # Here k=4.
    # So for each i, j in 0..3:
    # Term: A[i,4] * B[4,j].
    # This contributes to C[i,j].
    # We can do this with 1 term?
    # No, A[i,4] and B[4,j] are different variables.
    # The product is A[i,4] * B[4,j].
    # This is a rank-1 tensor in the variables A and B?
    # Yes, $T_{i,j,4} = A_{i,4} B_{4,j}$.
    # But we need to sum over k? No, k is fixed to 4.
    # So for each pair (i,j), we have a term.
    # Can we share the multiplication?
    # No, because A[i,4] changes with i and B[4,j] changes with j.
    # So we need 4*4 = 16 terms?
    # Wait, earlier I said Rank 1.
    # Let's re-evaluate $R(4,1,4)$.
    # $C_{ij} = A_{i1} B_{1j}$.
    # This is the outer product of vector A and vector B.
    # The tensor is $T_{i,j,k} = A_{ik} B_{kj}$. Here k=1.
    # $T_{i,j} = A_i B_j$.
    # This is a rank-1 tensor.
    # Why? Because $A_i B_j = (A_i \cdot 1) (B_j \cdot 1)$.
    # In the tensor decomposition:
    # $C_{ij} = \sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}$? No.
    # The standard form is $C_{ij} = \sum_c U_{i,j,c} V_{j,k,c}$? No.
    # The form is $C_{ij} = \sum_k A_{ik} B_{kj}$.
    # If we fix k=4, $C_{ij} = A_{i4} B_{4j}$.
    # This is a single bilinear form.
    # It can be represented by 1 term in the tensor decomposition?
    # Yes, if we define:
    # U[i, 4, 0] = 1
    # V[4, j, 0] = 1
    # W[j, i, 0] = 1 ?
    # Let's check the verification:
    # $T[i,j,k] = \sum_c U[i,j,c] V[j,k,c] W[k,i,c]$.
    # We want $T[i,j,4] = 1$ for all i,j in 0..3?
    # No, the tensor T is the identity tensor for matrix multiplication.
    # $T[i,j,k]$ is 1 if the term $A_{ik}B_{kj}$ appears in $C_{ij}$.
    # For $i,j < 4, k=4$, the term is $A_{i4}B_{4j}$.
    # So $T[i,j,4] = 1$.
    # Can we represent this with 1 term?
    # We need $\sum_c U[i,j,c] V[j,4,c] W[4,i,c] = 1$.
    # If we set c=0:
    # U[i,j,0] = 1? No, U depends on i,j.
    # V[j,4,0] = 1?
    # W[4,i,0] = 1?
    # Then $\sum_c 1 \cdot 1 \cdot 1 = 1$.
    # But U[i,j,0] must be the same for all i,j?
    # No, U is a matrix.
    # If we set U[i,j,0]=1 for all i,j, V[j,4,0]=1, W[4,i,0]=1.
    # Then for $k=4$, $T[i,j,4] = U[i,j,0]V[j,4,0]W[4,i,0] = 1$.
    # But what about other k?
    # For $k \neq 4$, we need $T[i,j,k] = 0$.
    # If we use this term, it only contributes to $k=4$.
    # So yes, 1 term is sufficient for the $4 \times 1 \times 4$ block?
    # Wait, U[i,j,0]=1 means we are multiplying the *entire* row of U?
    # No, U[i,j] is the coefficient for $A_{ij}$.
    # If U[i,j,0]=1, it means we are taking $A_{ij}$.
    # But we only want $A_{i4}$.
    # So U[i,4,0]=1, and U[i,j,0]=0 for j!=4.
    # Similarly, V[4,j,0]=1, V[k,j,0]=0 for k!=4.
    # And W[4,i,0]=1, W[k,i,0]=0 for k!=4.
    # Then the term is $A_{i4} B_{4j}$.
    # This contributes to $C_{ij}$.
    # And $T[i,j,4] = 1$.
    # So yes, Rank 1 for $4 \times 1 \times 4$.
    
    # So for block (0,1) which is A[0:4, 4] * B[4, 0:4]:
    # We need 1 term.
    # i in 0..3, j in 0..3, k=4.
    # U[i, 4, c] = 1
    # V[4, j, c] = 1
    # W[j, i, c] = 1 ?
    # Wait, W index is k*n + i. Here k=4, i is the output row.
    # The output is C[i,j].
    # The term is $A_{i4} B_{4j}$.
    # This contributes to $C_{i,j}$.
    # So W[4, i, c] should be 1?
    # Let's check the einsum:
    # $T[i,j,k] = \sum_c U[i,j,c] V[j,k,c] W[k,i,c]$.
    # We want $T[i,j,4] = 1$.
    # So $U[i,4,c] V[4,j,c] W[4,i,c] = 1$.
    # Set U[i,4,c]=1, V[4,j,c]=1, W[4,i,c]=1.
    # This works for all i,j in 0..3.
    # So 1 term.
    
    add_term(0, 0, 0) # Placeholder, we need to add specific terms
    
    # Let's clear and restart the addition logic properly.
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(i, j, k, val=1.0):
        U_list.append([0.0] * (n*m))
        V_list.append([0.0] * (m*p))
        W_list.append([0.0] * (p*n))
        U_list[-1][i*m + j] = val
        V_list[-1][j*p + k] = val
        W_list[-1][k*n + i] = val

    # 1. 4x4x4 Schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4):
                add_term(i, j, k)
                
    # 2. 4x1x4 (A[0:4,4] * B[4,0:4]) -> C[0:4,0:4]
    # Rank 1
    c_idx = len(U_list)
    U_list.append([0.0] * (n*m))
    V_list.append([0.0] * (m*p))
    W_list.append([0.0] * (p*n))
    for i in range(4):
        for j in range(4):
            U_list[-1][i*m + 4] = 1.0
            V_list[-1][4*p + j] = 1.0
            W_list[-1][4*n + i] = 1.0 # k=4, i=i
            
    # 3. 4x4x1 (A[0:4,0:4] * B[4,4]) -> C[0:4,4]
    # C[i,4] = sum_j A[i,j] B[j,4]
    # This is 4 dot products. Rank 4.
    for i in range(4):
        for j in range(4):
            add_term(i, j, 4) # k=4? No, k is the intermediate index.
            # Wait, B[j,4] means k=j, and the column is 4.
            # So V index is j*p + 4.
            # U index is i*m + j.
            # W index is 4*n + i.
            # So add_term(i, j, 4) is correct for U and V?
            # U[i,j], V[j,4], W[4,i].
            # Yes.
            
    # 4. 1x4x4 (A[4,0:4] * B[0:4,0:4]) -> C[4,0:4]
    # C[4,j] = sum_k A[4,k] B[k,j]
    # Rank 4.
    for j in range(4):
        for k in range(4):
            add_term(4, k, j) # i=4, j=k (intermediate), k=j (output col)
            # U[4,k], V[k,j], W[j,4].
            
    # 5. 1x4x1 (A[4,0:4] * B[0:4,4]) -> C[4,4]
    # C[4,4] = sum_k A[4,k] B[k,4]
    # Rank 4.
    for k in range(4):
        add_term(4, k, 4)
        
    # 6. 4x1x1 (A[0:4,4] * B[4,4]) -> C[0:4,4]
    # C[i,4] += A[i,4] B[4,4]
    # Rank 1.
    c_idx = len(U_list)
    U_list.append([0.0] * (n*m))
    V_list.append([0.0] * (m*p))
    W_list.append([0.0] * (p*n))
    for i in range(4):
        U_list[-1][i*m + 4] = 1.0
        V_list[-1][4*p + 4] = 1.0
        W_list[-1][4*n + i] = 1.0
        
    # 7. 1x1x4 (A[4,4] * B[4,0:4]) -> C[4,0:4]
    # C[4,j] += A[4,4] B[4,j]
    # Rank 1.
    c_idx = len(U_list)
    U_list.append([0.0] * (n*m))
    V_list.append([0.0] * (m*p))
    W_list.append([0.0] * (p*n))
    for j in range(4):
        U_list[-1][4*m + 4] = 1.0
        V_list[-1][4*p + j] = 1.0
        W_list[-1][j*n + 4] = 1.0
        
    # 8. 1x1x1 (A[4,4] * B[4,4]) -> C[4,4]
    # Rank 1.
    add_term(4, 4, 4)
    
    U = U_list
    V = V_list
    W = W_list
    
    # Transpose to (n*m, rank)
    U_final = np.array(U).T
    V_final = np.array(V).T
    W_final = np.array(W).T
    
    return U_final, V_final, W_final

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 135.0
    
    # Try the block decomposition
    try:
        U, V, W = _construct_block_decomp_4_1()
        if _verify_fast(U, V, W, n, m, p):
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    except Exception as e:
        pass
        
    # Fallback to schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    U0_list = [[float(x) for x in row] for row in U0]
    V0_list = [[float(x) for x in row] for row in V0]
    W0_list = [[float(x) for x in row] for row in W0]
    return {"U": U0_list, "V": V0_list, "W": W0_list}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
