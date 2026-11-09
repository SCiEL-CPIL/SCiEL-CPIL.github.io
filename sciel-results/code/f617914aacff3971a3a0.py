import time
import sys

DIMENSIONS = [3, 5, 6]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety (limit is 120s)

    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill based on schoolbook decomposition
    # T = sum_c U[:, c] * V[:, c] * W[:, c]
    # Indices:
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    # for c = i*m*p + j*p + k
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i*m+j
                U[i * m + j][col] = 1
                # V index: row j*p+k
                V[j * p + k][col] = 1
                # W index: row k*n+i
                W[k * n + i][col] = 1
                col += 1
                
                # Check deadline (unlikely to be hit)
                if time.time() > deadline:
                    # If we hit the deadline, return what we have so far
                    # But since we are just filling, we should complete it if possible
                    # or break and return partial? 
                    # The contract says "exactly decompose". A partial decomposition is invalid.
                    # So if we hit deadline, we should probably return the full schoolbook 
                    # if we haven't finished, but we are in the middle.
                    # Given n,m,p are small, this will never happen.
                    pass

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Self-test
    res = construct()
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Verify shapes
    assert len(U) == n * m
    assert len(V) == m * p
    assert len(W) == p * n
    assert len(U[0]) == rank
    assert len(V[0]) == rank
    assert len(W[0]) == rank
    
    # Verify correctness for a random case
    import random
    A = [[random.randint(0, 10) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(0, 10) for _ in range(p)] for _ in range(m)]
    
    # Compute C = A * B
    C = [[0] * p for _ in range(n)]
    for i in range(n):
        for k in range(p):
            s = 0
            for j in range(m):
                s += A[i][j] * B[j][k]
            C[i][k] = s
            
    # Compute C' using tensor decomposition
    C_prime = [[0] * p for _ in range(n)]
    for c in range(rank):
        u_col = [U[r][c] for r in range(n * m)]
        v_col = [V[r][c] for r in range(m * p)]
        w_col = [W[r][c] for r in range(p * n)]
        
        # u is vector of length n*m, v is m*p, w is p*n
        # The tensor term is u \otimes v \otimes w
        # We need to contract this with A and B?
        # Actually, the decomposition is for the multiplication tensor T_{ijk}
        # T_{ijk} = sum_c U_{ij,c} V_{jk,c} W_{ki,c}
        # Then C_{ik} = sum_j A_{ij} B_{jk} T_{ijk}? No.
        # The matrix multiplication tensor is defined such that 
        # <T, A \otimes B> = C.
        # Specifically, C_{ik} = sum_{j} A_{ij} B_{jk}.
        # The tensor T has indices i, j, k.
        # T_{ijk} = 1 if it corresponds to the term A_{ij}B_{jk} in the sum?
        # Standard definition: The multiplication map is a bilinear map.
        # The rank is the number of bilinear terms.
        # Each term is u_c \otimes v_c \otimes w_c.
        # The product is sum_c (u_c^T A) * (v_c^T B) * w_c? No.
        
        # Let's check the indices in the seed code:
        # U[i*m + j, c] = 1  -> u_c has 1 at (i,j)
        # V[j*p + k, c] = 1  -> v_c has 1 at (j,k)
        # W[k*n + i, c] = 1  -> w_c has 1 at (k,i)
        
        # The reconstruction formula for C_{ik} is:
        # C_{ik} = sum_c (sum_{j} U_{ij,c} A_{ij}) * (sum_{j} V_{jk,c} B_{jk}) * W_{ki,c} ?
        # No, that would be cubic.
        
        # The standard tensor decomposition for matrix multiplication:
        # T = sum_c u_c \otimes v_c \otimes w_c
        # Then C = sum_c (A \cdot u_c) \cdot (v_c \cdot B) \cdot w_c^T ?
        # Let's trace:
        # u_c is a vector of length n*m. Let's reshape to U_c (n x m).
        # v_c is a vector of length m*p. Let's reshape to V_c (m x p).
        # w_c is a vector of length p*n. Let's reshape to W_c (p x n).
        
        # The term is (A^T U_c) ? No.
        # The bilinear form is:
        # C_{ik} = sum_j A_{ij} B_{jk}
        # We want to represent this as sum_c alpha_c(A) beta_c(B) gamma_c(C) ?
        # Actually, the rank is the minimum number of multiplications.
        # Each multiplication is scalar * scalar.
        # The factors are linear combinations of A and B.
        
        # In the tensor representation T_{ijk}:
        # T_{ijk} = 1 for all i,j,k? No, T is the identity map on the space?
        # The multiplication tensor T has entries T_{ijk} = 1 if the index combination (i,j,k) is "active"?
        # Actually, for matrix multiplication, the tensor is:
        # T_{ijk} = 1 for all i,j,k.
        # Wait, no. The tensor maps A \otimes B to C.
        # C_{ik} = sum_j A_{ij} B_{jk}.
        # This can be written as T_{ijk} A_{ij} B_{jk} where T_{ijk} = 1.
        # So the tensor T is the all-ones tensor of shape (n,m,p)?
        # No, the indices are i (row of A), j (col of A / row of B), k (col of B).
        # The output index is (i,k).
        # The tensor T has indices (i, j, k) and maps to output (i, k).
        # So T_{ijk} contributes to C_{ik}.
        # Specifically, C_{ik} = sum_j T_{ijk} A_{ij} B_{jk}.
        # If T_{ijk} = 1 for all i,j,k, then C_{ik} = sum_j A_{ij} B_{jk}.
        # So the tensor T is indeed the all-ones tensor of size n x m x p.
        
        # The rank of this tensor is the matrix multiplication rank.
        # The schoolbook decomposition uses n*m*p terms.
        # Term c corresponds to a specific (i,j,k).
        # u_c has 1 at (i,j), 0 elsewhere.
        # v_c has 1 at (j,k), 0 elsewhere.
        # w_c has 1 at (k,i), 0 elsewhere.
        
        # Let's verify the reconstruction:
        # sum_c (sum_{i',j'} U_{i'j',c} A_{i'j'}) * (sum_{j',k'} V_{j'k',c} B_{j'k'}) * W_{ki,c} ?
        # If we define:
        # M_c = (sum_{i',j'} U_{i'j',c} A_{i'j'}) * (sum_{j',k'} V_{j'k',c} B_{j'k'})
        # Then C_{ik} = sum_c M_c W_{ki,c}.
        
        # Let's test this for the schoolbook term c = (i0, j0, k0).
        # U_{i'j',c} is 1 if i'=i0, j'=j0.
        # So sum U A = A_{i0, j0}.
        # V_{j'k',c} is 1 if j'=j0, k'=k0.
        # So sum V B = B_{j0, k0}.
        # W_{ki,c} is 1 if k=k0, i=i0.
        # So W_{ki,c} is 1 if (k,i) = (k0, i0).
        
        # C_{ik} = sum_{c} A_{i_c, j_c} B_{j_c, k_c} W_{ki, c}
        # For a fixed (i,k), the term is non-zero only if W_{ki,c} != 0.
        # W_{ki,c} is non-zero only if c corresponds to (i_c, j_c, k_c) such that k_c=k and i_c=i.
        # In schoolbook, for each c=(i,j,k), W_{ki,c}=1.
        # So for a fixed output (i,k), the only c that contributes is the one with i_c=i and k_c=k.
        # There is exactly one such c for each j.
        # So C_{ik} = sum_{j} A_{i,j} B_{j,k} * 1.
        # This is correct.
        
        # So the verification code should be:
        for c in range(rank):
            # Calculate scalar factors
            sum_u = 0
            for i_prime in range(n):
                for j_prime in range(m):
                    sum_u += U[i_prime * m + j_prime][c] * A[i_prime][j_prime]
            
            sum_v = 0
            for j_prime in range(m):
                for k_prime in range(p):
                    sum_v += V[j_prime * p + k_prime][c] * B[j_prime][k_prime]
            
            # Add to C_prime
            for k_prime in range(p):
                for i_prime in range(n):
                    C_prime[i_prime][k_prime] += sum_u * sum_v * W[k_prime * n + i_prime][c]
                    
    # Check if C and C_prime are equal
    for i in range(n):
        for k in range(p):
            if C[i][k] != C_prime[i][k]:
                print(f"Mismatch at ({i},{k}): {C[i][k]} vs {C_prime[i][k]}")
                assert False, "Decomposition failed"
                
    print("Decomposition verified.")
