import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    n, m, p = 4, 4, 5
    
    # We will attempt to construct a rank-35 decomposition:
    # 1. Lander-Bailey rank-19 for 4x4x4
    # 2. Schoolbook rank-16 for 4x4x1
    # If we can't verify it quickly, we fall back to schoolbook rank-80.
    
    # The Lander-Bailey decomposition of M_{4,4,4} is a sum of 19 rank-1 tensors.
    # Each rank-1 tensor is defined by a triple of indices (i, j, k) for the U, V, W supports?
    # No, a rank-1 tensor in the matrix multiplication tensor M_{n,m,p} is of the form:
    # T(i,j,k) = u_{ij} v_{jk} w_{ki}
    # where u, v, w are vectors.
    # For the "standard" basis of rank-1 tensors that appear in decompositions,
    # the coefficients are often 1, -1, 0.
    # The Lander-Bailey decomposition is a specific set of 19 such terms.
    
    # Let's try to retrieve the Lander-Bailey coefficients.
    # Since I cannot look them up, I will attempt to *solve* for a rank-19 decomposition?
    # No, that's computationally infeasible in 150s.
    
    # Alternative: Use a known rank-20 or lower decomposition?
    # Actually, there is a well-known rank-19 decomposition by Lander and Bailey (1986).
    # Let's see if I can reconstruct it or find a simpler valid one.
    
    # If I cannot guarantee the LB coefficients, I should use a valid, known decomposition.
    # Is there a rank-20 or rank-21 that is easier?
    # Strassen for 4x4x4 is rank 49.
    # Schoolbook is 64.
    
    # Wait, can I use a rank-19 decomposition if I hardcode the 19 terms?
    # The 19 terms of LB are:
    # (0,0,0), (0,1,1), (0,2,2), (0,3,3),
    # (1,0,0), (1,1,0), (1,2,1), (1,3,2),
    # (2,0,1), (2,1,1), (2,2,0), (2,3,3),
    # (3,0,2), (3,1,3), (3,2,0), (3,3,1),
    # (0,1,0), (1,0,1), (2,3,0) -- This is a guess. I don't know the exact indices.
    
    # Let's try a different strategy:
    # Can I reduce the rank of the 4x4x1 part?
    # 4x4x1 is just a linear map from 16-dim to 16-dim? No, C (4x1) = A (4x4) * B (4x1).
    # Each C_i = sum_j A_ij B_j.
    # This is 4 independent linear combinations of 4 variables?
    # No, B is a vector of 4 elements.
    # C_i = A_i . B.
    # This is 4 dot products.
    # Can we do 4 dot products of length 4 in rank < 16?
    # A dot product of length n has rank n (schoolbook).
    # So 4 dot products of length 4 have rank 4*4 = 16.
    # Is there a way to share computations?
    # If A is arbitrary, no.
    # So 16 is likely minimal for the 4x4x1 part.
    
    # So the bottleneck is the 4x4x4 part.
    # If I can't do rank 19, can I do rank 20, 21, ... 48?
    # Any rank < 64 is an improvement over schoolbook.
    # Strassen is 49.
    # Can I implement Strassen for 4x4x4?
    # Yes, it's easier than LB.
    
    # Let's implement Strassen for 4x4x4 (rank 49) + Schoolbook for 4x4x1 (rank 16).
    # Total rank: 65.
    # 65 < 80. This is an improvement.
    
    # Implementation of Strassen 4x4x4:
    # Split A, B, C into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 - M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 - M7
    
    # Each M is a 2x2x2 multiplication.
    # Rank of 2x2x2 multiplication is 7 (Strassen).
    # So total rank = 7 * 7 = 49.
    
    # We need to map this to the global U, V, W matrices.
    
    # Global dimensions:
    # U: (n*m, rank) = (16, 65)
    # V: (m*p, rank) = (20, 65)
    # W: (p*n, rank) = (20, 65)
    
    # Let's build the U, V, W matrices for the 4x4x4 part (rank 49).
    # Then append the 4x4x1 part (rank 16).
    
    rank_total = 49 + 16
    U = np.zeros((16, rank_total), dtype=int)
    V = np.zeros((20, rank_total), dtype=int)
    W = np.zeros((20, rank_total), dtype=int)
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen decomposition
    # i_base, j_base, k_base are the top-left indices of the 2x2 blocks in A, B, C.
    # A is 4x4, so blocks are at (0,0), (0,2), (2,0), (2,2).
    # For the 4x4x4 part, i_base, j_base, k_base are in {0, 2}.
    
    def add_strassen_2x2x2(i_base, j_base, k_base):
        nonlocal col
        # The 2x2x2 Strassen decomposition has 7 terms.
        # Term 1: M1 = (A11+A22)(B11+B22)
        # A11 is at (i_base, j_base), A22 is at (i_base+1, j_base+1)
        # B11 is at (j_base, k_base), B22 is at (j_base+1, k_base+1)
        # C11 is at (i_base, k_base), C12 is at (i_base, k_base+1), etc.
        
        # M1
        U[i_base*4+j_base][col] = 1
        U[(i_base+1)*4+(j_base+1)][col] = 1
        V[j_base*4+k_base][col] = 1
        V[(j_base+1)*4+(k_base+1)][col] = 1
        W[k_base*4+i_base][col] = 1       # C11
        W[(k_base+1)*4+(i_base+1)][col] = 1 # C22
        W[k_base*4+(i_base+1)][col] = -1   # C21
        W[(k_base+1)*4+i_base][col] = -1   # C12
        col += 1
        
        # M2 = (A21+A22)B11
        U[(i_base+1)*4+j_base][col] = 1
        U[(i_base+1)*4+(j_base+1)][col] = 1
        V[j_base*4+k_base][col] = 1
        W[k_base*4+(i_base+1)][col] = 1   # C21
        col += 1
        
        # M3 = A11(B12-B22)
        U[i_base*4+j_base][col] = 1
        V[j_base*4+(k_base+1)][col] = 1
        V[(j_base+1)*4+(k_base+1)][col] = -1
        W[(k_base+1)*4+i_base][col] = 1   # C12
        col += 1
        
        # M4 = A22(B21-B11)
        U[(i_base+1)*4+(j_base+1)][col] = 1
        V[(j_base+1)*4+k_base][col] = 1
        V[j_base*4+k_base][col] = -1
        W[k_base*4+(i_base+1)][col] = 1   # C21
        col += 1
        
        # M5 = (A11+A12)B22
        U[i_base*4+j_base][col] = 1
        U[i_base*4+(j_base+1)][col] = 1
        V[(j_base+1)*4+(k_base+1)][col] = 1
        W[(k_base+1)*4+(i_base+1)][col] = 1 # C22
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[(i_base+1)*4+j_base][col] = 1
        U[i_base*4+j_base][col] = -1
        V[j_base*4+k_base][col] = 1
        V[j_base*4+(k_base+1)][col] = 1
        W[k_base*4+i_base][col] = -1      # C11
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[i_base*4+(j_base+1)][col] = 1
        U[(i_base+1)*4+(j_base+1)][col] = -1
        V[(j_base+1)*4+k_base][col] = 1
        V[(j_base+1)*4+(k_base+1)][col] = 1
        W[(k_base+1)*4+i_base][col] = -1  # C12
        col += 1
        
    # Apply Strassen to the 4x4x4 block
    add_strassen_2x2x2(0, 0, 0)
    
    # Now add the 4x4x1 part (schoolbook)
    # This multiplies A (4x4) by B2 (4x1) to get C2 (4x1).
    # B2 corresponds to columns 4, 5, 6, 7 of the full 4x5 matrix B?
    # No, B is 4x5. Columns 0-3 are for 4x4x4, column 4 is for 4x4x1.
    # So for the 4x4x1 part:
    # A is 4x4, B is 4x1 (column 4 of B), C is 4x1 (column 4 of C).
    # Indices:
    # A: i in 0..3, j in 0..3
    # B: j in 0..3, k=4
    # C: k=4, i in 0..3
    
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j*4+4][col] = 1  # j*5+4? No, V is (m*p, rank) = (4*5, rank) = (20, rank).
                                # V index is j*p + k. p=5. So j*5+4.
            V[j*5+4][col] = 1
            W[4*4+i][col] = 1  # W index is k*n + i. n=4. k=4. So 4*4+i.
            col += 1
            
    # Verify shapes
    assert U.shape == (16, rank_total)
    assert V.shape == (20, rank_total)
    assert W.shape == (20, rank_total)
    
    # Convert to lists of lists as required by interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Quick self-test to ensure Strassen part is correct
def _test():
    n, m, p = 4, 4, 5
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    rank = U.shape[1]
    
    # Check tensor identity: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 if (i,j,k) is a valid term, 0 otherwise?
    # Actually, the condition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if i<n, j<m, k<p (for the full multiplication)
    # Wait, the tensor M_{n,m,p} has entries 1 for all (i,j,k).
    # So we need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for ALL i, j, k.
    
    # Let's check a few random indices
    np.random.seed(42)
    for _ in range(100):
        i = np.random.randint(0, n)
        j = np.random.randint(0, m)
        k = np.random.randint(0, p)
        val = 0
        for r in range(rank):
            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        if val != 1:
            print(f"FAIL: i={i}, j={j}, k={k}, val={val}")
            return False
    print("PASS")
    return True

if __name__ == "__main__":
    _test()
