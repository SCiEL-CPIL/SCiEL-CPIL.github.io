"""Matrix multiplication 2x5x5 - rank minimization via structured decomposition."""
DIMENSIONS = [2, 5, 5]

import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x5 by 5x5 matrix.
    
    The output follows W[k*n + i, r] convention (n=2, m=5, p=5).
    U has shape (n*m, rank) = (10, rank)
    V has shape (m*p, rank) = (25, rank)  
    W has shape (p*n, rank) = (10, rank)
    
    The tensor identity is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where T[i,j,k] = 1 if this corresponds to a valid multiplication term, else 0.
    
    For matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    So T[i,j,k] = 1 for all valid (i,j,k) indices.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school, "rank": rank_schoolbook}
    
    # Try to find lower rank decompositions
    # For 2x5x5, known lower bounds and constructions:
    # - Schoolbook: 50
    # - We can try various structured decompositions
    
    # Strategy: Use the fact that n=2 is small
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0..4}, j in {0..4}
    
    # Try rank 49 first (minimal possible improvement)
    # Then try lower ranks with various techniques
    
    for target_rank in range(49, 24, -1):
        if time.time() > deadline - 5.0:
            break
        
        # Try random search for low-rank decomposition
        # The tensor T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor for the multiplication structure)
        # Wait, no. For matrix multiplication, T[i,j,k] = delta_{structure}
        # Actually for C = A*B, the multiplication tensor has T[i,j,k] = 1 when we're computing C[i,k] from A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for ALL combinations of i in [0,n), j in [0,m), k in [0,p)
        
        # The tensor is simply the all-ones tensor of size n x m x p!
        # Wait, that's not right either. Let me reconsider.
        
        # The matrix multiplication tensor T is defined by:
        # T[i, j, k] = 1 if and only if this term contributes to C[i,k]
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        # Yes! The tensor is the all-ones tensor.
        # The rank of the all-ones tensor of size n x m x p is... 
        # It's the tensor rank of the all-ones tensor.
        
        # For the all-ones tensor of size n x m x p, the tensor rank is max(n,m,p) if we're over reals?
        # No, that's not right. The tensor rank of the all-ones tensor 1_{n x m x p} is...
        
        # Actually, the all-ones tensor can be written as:
        # T[i,j,k] = 1 = u_i * v_j * w_k where u_i = 1, v_j = 1, w_k = 1
        # So rank 1!
        
        # But wait, that would mean matrix multiplication has rank 1, which is absurd.
        # I'm confusing the multiplication tensor with something else.
        
        # Let me re-read the task: "exactly decompose the matrix multiplication tensor"
        # The matrix multiplication tensor for (n,m,p) is NOT the all-ones tensor.
        
        # The correct definition: The multiplication tensor T for computing C = A*B has:
        # T[i, j, k] = 1 if the term A[i,j]*B[j,k] appears in the computation of C[i,k]
        # But since C[i,k] = sum_j A[i,j]*B[j,k], every term A[i,j]*B[j,k] appears exactly once.
        # So T[i,j,k] = 1 for all i,j,k.
        
        # Hmm, but then the rank would be 1, which contradicts known results.
        
        # I think I'm misunderstanding the tensor structure. Let me reconsider.
        
        # In tensor decomposition for matrix multiplication, the tensor is:
        # T is a 3-tensor where T[i, j, k] corresponds to the coefficient of A[i,j]*B[j,k] in C[i,k]
        # For standard matrix multiplication, this coefficient is 1 for all valid i,j,k.
        
        # So the tensor IS the all-ones tensor. But then why isn't matrix multiplication rank 1?
        
        # Ah, I see the issue. The tensor decomposition must satisfy:
        # C[i,k] = sum_r U[i, r] * V[r, k] * W[r]  (approximately)
        # But the exact constraint is on the bilinear map.
        
        # Let me reconsider the interface:
        # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
        # The decomposition should satisfy:
        # For all i in [0,n), k in [0,p):
        #   C[i,k] = sum_j A[i,j] * B[j,k]
        #   = sum_r U[i*m + ?] * ... 
        
        # Actually, looking at the schoolbook:
        # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
        # col corresponds to (i,j,k)
        
        # The reconstruction would be:
        # C[i,k] = sum_r U[i*m+?, r] * V[?*p+k, r] * W[k*n+i, r]
        
        # I think the correct identity is:
        # sum_r U[a, r] * V[b, r] * W[c, r] = delta_{a,b,c} (Kronecker delta in 3D)
        # where a = i*m+j, b = j*p+k, c = k*n+i
        
        # For schoolbook: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # = 1 if (i,j,k) is the unique column where all three are 1
        # = 1 if the column index matches (i,j,k)
        
        # So the tensor is T[a, b, c] = 1 if a, b, c are consistent (share the same j, and i,k match)
        # Specifically, a = i*m+j, b = j*p+k, c = k*n+i must be consistent.
        
        # Given a = i*m+j, b = j*p+k, c = k*n+i:
        # From a: i = a // m, j = a % m
        # From b: j = b // p, k = b % p  
        # From c: i = c % n, k = c // n
        # Consistency: j from a = j from b, k from b = k from c, i from a = i from c
        
        # So T[a,b,c] = 1 if and only if:
        # (a%m == b//p) and (b%p == c//n) and (a//m == c%n)
        
        # This is the actual multiplication tensor!
        
        # For n=2, m=5, p=5:
        # a in [0,10), b in [0,25), c in [0,10)
        # T[a,b,c] = 1 iff (a%5 == b//5) and (b%5 == c//2) and (a//5 == c%2)
        
        # Now we need to find the minimum rank decomposition of this tensor.
        
        # Let's verify the schoolbook works:
        # For each (i,j,k), set column c_idx to have U[i*m+j, c_idx]=1, V[j*p+k, c_idx]=1, W[k*n+i, c_idx]=1
        # Then sum_r U[a,r]*V[b,r]*W[c,r] = 1 if there exists r where U[a,r]=V[b,r]=W[c,r]=1
        # In schoolbook, r corresponds to (i,j,k), so U[a,r]=1 iff a = i*m+j for that (i,j,k)
        # So the sum is 1 iff a = i*m+j, b = j*p+k, c = k*n+i for the same (i,j,k)
        # Which is exactly the consistency condition. Good.
        
        # Now, can we do better than rank 50?
        # The tensor has a special structure. Let's think about it differently.
        
        # T[a,b,c] depends on:
        # i = a//5, j = a%5, so a = 5*i + j
        # j = b//5, k = b%5, so b = 5*j + k  
        # i = c%2, k = c//2, so c = 2*k + i
        
        # T = 1 iff j_from_a = j_from_b, k_from_b = k_from_c, i_from_a = i_from_c
        # i_from_a = a//5, j_from_a = a%5
        # j_from_b = b//5, k_from_b = b%5
        # i_from_c = c%2, k_from_c = c//2
        
        # So T[a,b,c] = 1 iff:
        # a%5 == b//5  (j matches)
        # b%5 == c//2  (k matches)
        # a//5 == c%2  (i matches)
        
        # This is a constraint satisfaction problem on the indices.
        
        # The tensor can be seen as a 3D array with entries 0 or 1.
        # We want to write it as sum of rank-1 tensors: T = sum_r u_r * v_r * w_r
        # where u_r is a vector of length 10, v_r of length 25, w_r of length 10.
        
        # One approach: use the structure to find a low-rank decomposition.
        # Since n=2 is small, we can enumerate possibilities.
        
        # Let's try a structured approach:
        # Fix i and k. Then for each j, we need T[5i+j, 5j+k, 2k+i] = 1.
        # For fixed (i,k), there are m=5 values of j.
        # The sum over r of u_r[5i+j] * v_r[5j+k] * w_r[2k+i] should equal 1 for all j.
        
        # This is getting complex. Let's just try random searches with verification.
        
        def verify_decomposition(U, V, W, rank):
            """Verify that the decomposition exactly reproduces the tensor."""
            # Check all (a, b, c) triples
            for a in range(n * m):
                i_a = a // m
                j_a = a % m
                for b in range(m * p):
                    j_b = b // p
                    k_b = b % p
                    for c in range(p * n):
                        i_c = c % n
                        k_c = c // n
                        expected = 1 if (j_a == j_b and k_b == k_c and i_a == i_c) else 0
                        actual = 0
                        for r in range(rank):
                            actual += U[a][r] * V[b][r] * W[c][r]
                        if abs(actual - expected) > 1e-9:
                            return False
            return True
        
        # Try to find lower rank via random restarts
        import random
        random.seed(42)
        
        for attempt in range(20):
            if time.time() > deadline - 3.0:
                break
            
            target_rank = min(49, 40 - attempt * 2)
            if target_rank < 20:
                break
            
            # Random initialization
            U = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify_decomposition(U, V, W, target_rank):
                result = {"U": U, "V": V, "W": W, "rank": target_rank}
                if target_rank < best_result["rank"]:
                    best_result = result
                break
        
        if time.time() > deadline - 2.0:
            break
    
    # Remove the rank key from the final result
    final = {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
    return final
# EVOLVE-BLOCK-END
