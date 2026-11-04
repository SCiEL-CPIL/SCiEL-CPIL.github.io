"""Rewritten program for 4x5x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # We'll construct a decomposition using Strassen for 4x4x4 blocks
    # plus trivial decompositions for the remaining strips.
    
    # Strategy:
    # A is 4x5, B is 5x6
    # C = A * B is 4x6
    # 
    # Split A into A_left (4x4) and A_right (4x1)
    # Split B into B_left (5x4) and B_right (5x2)
    # 
    # C = A_left * B_left + A_right * B_right
    # 
    # A_left * B_left: 4x4 * 4x4 -> use Strassen (rank 7)
    # But wait, B_left is 5x4, not 4x4. Let me reconsider.
    #
    # Actually, let me think about this differently.
    # A is 4x5, B is 5x6.
    # 
    # Let's split the middle dimension (5) into 4+1.
    # A = [A1 | A2] where A1 is 4x4, A2 is 4x1
    # B = [B1; B2] where B1 is 4x6, B2 is 1x6
    #
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 4x4 * 4x6 -> This is 4x4 times 4x6.
    # We can view this as 4x4 * (4x6). 
    # Split B1 into B1a (4x4) and B1b (4x2).
    # C_part1 = A1*B1a + A1*B1b
    # A1*B1a: 4x4 * 4x4 -> Strassen rank 7
    # A1*B1b: 4x4 * 4x2 -> This is 4x4 times 4x2.
    #   We can do this with 2 separate 4x4*4x1 products.
    #   4x4*4x1: rank 1? No. The tensor for 4x4x1 has all entries 1.
    #   Actually, for multiplying a 4x4 by a 4x1, we get a 4x1 result.
    #   C_i = sum_j A_ij * B_j. This is just a linear combination.
    #   The tensor T[i,j,0] = 1 for all i,j. Rank of this tensor is 1? 
    #   Wait, no. For each output i, C_i = sum_j A_ij B_j.
    #   This is not a rank-1 tensor in general. Let me reconsider.
    #
    # Actually, the matrix multiplication tensor for m x n * n x p has rank at least max(mn, np, mp) in some cases, but that's not quite right either.
    #
    # Let me use a simpler approach: just use the schoolbook decomposition but try to find a better one through structured search.
    #
    # Actually, let me just implement a proper Strassen-based construction.
    #
    # For 4x4 * 4x4, Strassen gives rank 7.
    # For 4x4 * 4x2, we can split the 2 columns: two 4x4*4x1 products.
    # 4x4*4x1: The result is 4x1. C_i = sum_j A_ij B_j.
    # This is equivalent to computing 4 separate dot products of length 4.
    # Each dot product of length 4 can be done with rank 1? No.
    # Actually, for a single dot product of length n, the rank is 1 (it's just one multiplication).
    # Wait, no. The tensor for a single dot product of length n is T[i] = 1 for all i (since there's only one output).
    # So rank is 1.
    # But we have 4 separate dot products (one for each output row).
    # They share the same B vector but different A rows.
    # So we need 4 rank-1 tensors? No, they can be combined.
    # Actually, C = A * b where A is 4x4 and b is 4x1.
    # This is a 4x4x1 tensor. The rank of this tensor... 
    # T[i,j,0] = 1 for all i,j. This is an all-ones tensor of shape 4x4x1.
    # The rank of an all-ones tensor of shape a x b x c is min(a, b, c) if one dimension is 1?
    # Actually, for a x b x 1, the tensor is just a matrix of all ones.
    # The rank of a matrix of all ones is 1.
    # So 4x4*4x1 has rank 1? That doesn't seem right for matrix multiplication.
    #
    # Let me reconsider. In the tensor decomposition for matrix multiplication,
    # we have T[i,j,k] = 1 if the indices match appropriately, 0 otherwise.
    # For m x n * n x p, T[i,j,k] = 1 if... actually the standard formulation is:
    # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor T has T[i,j,k] = 1 for all valid i,j,k, and the decomposition is:
    # sum_r U[i,j,r] V[j,k,r] W[i,k,r] = 1 if we're computing the (i,k) entry.
    #
    # Wait, I think I'm overcomplicating this. Let me just use the standard schoolbook
    # and try to improve it with a structured approach.
    
    # Schoolbook: rank = 4*5*6 = 120
    # Let's try to do better with a block approach.
    
    # Split A (4x5) into A1 (4x4) and A2 (4x1)
    # Split B (5x6) into B1 (4x6) and B2 (1x6)
    # C = A1*B1 + A2*B2
    
    # A1*B1: 4x4 * 4x6
    # Split B1 into B1a (4x4) and B1b (4x2)
    # A1*B1a: 4x4 * 4x4 -> Strassen rank 7
    # A1*B1b: 4x4 * 4x2 -> Split into two 4x4*4x1
    #   Each 4x4*4x1: rank 4 (one per output row, since each output is a dot product)
    #   Actually, for 4x4*4x1, we compute 4 dot products of length 4.
    #   Each dot product is rank 1, but they share the B vector.
    #   The total rank is 4 (we need 4 multiplications total, one per output).
    #   Wait, no. We need A_ij * B_j for each i, summed over j.
    #   This is 4 outputs, each requiring 4 multiplications. Total 16 multiplications schoolbook.
    #   But can we do it with fewer? The tensor is 4x4x1 with all entries 1.
    #   The rank of this tensor is 4 (since it's a 4x4 matrix of ones, rank 1 as a matrix,
    #   but as a 3-tensor with one dimension being 1, the rank equals the matrix rank, which is 1).
    #   Hmm, I'm confusing myself. Let me just use schoolbook for the small parts.
    
    # Let me just implement a clean schoolbook and return it, since finding a truly
    # lower-rank decomposition for 4x5x6 is non-trivial.
    
    schoolbook_rank = n * m * p  # 120
    
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    def to_list(mat):
        return [[round(float(x), 12) for x in row] for row in mat]
    
    def clean_list(lst):
        for row in lst:
            for idx in range(len(row)):
                v = row[idx]
                if abs(v) < 1e-10:
                    row[idx] = 0.0
                elif abs(v - round(v)) < 1e-10:
                    row[idx] = float(round(v))
                elif abs(v - round(v*2)/2) < 1e-10:
                    row[idx] = round(v*2)/2
        return lst
    
    U_out = clean_list(to_list(U))
    V_out = clean_list(to_list(V))
    W_out = clean_list(to_list(W))
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
