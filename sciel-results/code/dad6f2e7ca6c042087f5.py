"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    # Schoolbook decomposition: rank 48
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

def _verify(U, V, W, n, m, p):
    """Check if the factorization exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # The decomposition must satisfy: for all i,j,k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    # And for all non-tensor entries, the sum must be 0.
    # Actually, the standard decomposition for matrix multiplication tensor T[i,j,k] = delta is:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We need T[i,j,k] = 1 for all i,j,k (since A*B[i,k] = sum_j A[i,j]*B[j,k], the tensor has 1s everywhere).
    # Wait, the matrix multiplication tensor has entries T[i,j,k] = 1 for all i,j,k.
    # And the "orthogonality" condition is that for any (i,j,k) not a valid triple... 
    # Actually, the tensor for matrix mult is defined on indices (i,j,k) where i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = 1 for all such triples.
    # The decomposition is valid if:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] == 1  for all i,j,k
    # That's it. There are no "other" indices because the tensor is only defined on these indices.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _try_rank_reduction(n, m, p, target_rank, timeout=30):
    """
    Try to find a decomposition of rank < n*m*p.
    Uses a simple iterative approach: start from schoolbook, try to merge terms.
    """
    start_time = time.time()
    
    # Try a specific known structure: Strassen-like for sub-blocks
    # For 3x4 * 4x4, we can try to use Strassen on 2x2 blocks.
    # Split A (3x4) into A1 (2x4), A2 (1x4)
    # Split B (4x4) into B1 (4x2), B2 (4x2)
    # C1 (2x2) = A1 * B1, C2 (2x2) = A1 * B2
    # C3 (1x2) = A2 * B1, C4 (1x2) = A2 * B2
    
    # Strassen on 2x2 * 2x2 gives rank 7 instead of 8.
    # So C1 and C2 each save 1 rank. Total saving: 2.
    # Schoolbook: 48. New rank: 48 - 2 = 46?
    # Let's check: 
    # A1 is 2x4, B1 is 4x2. Product is 2x2. Strassen needs 7 multiplications for 2x2*2x2.
    # But A1 is 2x4, not 2x2. We need to split A1 into A11, A12 (2x2 each) and B1 into B11, B12 (2x2 each).
    # Then C1 = A11*B11 + A12*B12. This is exactly the 2x2*2x2 case.
    # So yes, we can use Strassen for C1 and C2.
    # C3 and C4 are 1x2 * 2x2? No, A2 is 1x4, B1 is 4x2. Product is 1x2.
    # 1x4 * 4x2: schoolbook is 1*4*2 = 8. Can we do better?
    # It's just two dot products of length 4. Rank is 8 (each output needs 4 mults).
    # So total rank = 7 (for C1) + 7 (for C2) + 8 (for C3) + 8 (for C4) = 30?
    # Wait, that seems too good. Let me re-evaluate.
    # The tensor is for the whole 3x4 * 4x4 multiplication.
    # The decomposition must satisfy the tensor equation for all i,j,k.
    # If I build separate decompositions for each output block, they are independent.
    # So the total rank is the sum of the ranks of each block.
    # C1 (2x2): rank 7 (Strassen)
    # C2 (2x2): rank 7 (Strassen)
    # C3 (1x2): rank 8 (schoolbook, since 1x4*4x2 is just 2 dot products of length 4)
    # C4 (1x2): rank 8 (schoolbook)
    # Total: 7 + 7 + 8 + 8 = 30.
    
    # Let's verify this is correct. The tensor for C3 (1x2) has indices (i=2, j=0..3, k=0..1).
    # T[2,j,k] = 1 for all j,k.
    # This is equivalent to multiplying a 1x4 matrix by a 4x2 matrix.
    # The rank of this tensor is indeed 8 (it's the "dot product" tensor for two vectors of length 4, repeated for 2 outputs, but the outputs are independent).
    # Actually, the tensor for 1xm * mxp has rank m*p. Here m=4, p=2, so rank 8.
    
    # So rank 30 is achievable? That's a huge improvement from 48.
    # Let's double-check the Strassen part.
    # For C1 (2x2 output): indices (i=0,1; j=0..3; k=0,1)
    # T[i,j,k] = 1 for all i,j,k in these ranges.
    # This is the tensor for 2x4 * 4x2.
    # Split j into two blocks of 2: j1 in {0,1}, j2 in {2,3}.
    # Then C1[i,k] = sum_{j1} A[i,j1]B[j1,k] + sum_{j2} A[i,j2]B[j2,k]
    # = A1 * B1 + A2 * B2
    # where A1 is 2x2, B1 is 2x2, A2 is 2x2, B2 is 2x2.
    # The tensor for A1*B1 + A2*B2 is the sum of two 2x2*2x2 tensors.
    # But they share the same i and k indices!
    # So the tensor is T[i,j,k] = 1 for all i,k in {0,1} and j in {0,1,2,3}.
    # This is exactly the tensor for 2x4 * 4x2.
    # Can we decompose this with rank 7?
    # No! Strassen's 7-multiplication algorithm works for computing C = A*B where A,B are 2x2.
    # Here, we have C = A1*B1 + A2*B2.
    # This is NOT the same as a single 2x2 matrix multiplication.
    # It's a sum of two 2x2 matrix multiplications.
    # The rank of the sum is at most 8+8=16, but can it be 7?
    # Actually, the tensor for 2x4*4x2 is isomorphic to the tensor for (2x2*2x2) + (2x2*2x2).
    # The rank of a sum of two tensors is at most the sum of their ranks, but it can be less.
    # However, it's known that the rank of 2x4*4x2 is 8.
    # Let me check: the tensor has 2*4*2 = 16 entries, all 1.
    # The border rank or rank?
    # Actually, I think the rank is 8.
    # Because you need to compute 2*2=4 outputs, each requiring a sum of 4 products.
    # But the products can be shared?
    # For C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0] + A[0,3]B[3,0]
    # For C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1] + A[0,3]B[3,1]
    # The A factors are shared between C[0,0] and C[0,1].
    # This is similar to the 1x4 * 4x2 case, which has rank 8.
    # So 2x4 * 4x2 likely has rank 16? No, that's schoolbook.
    # Let me think. The tensor is T[i,j,k] = 1 for i in {0,1}, j in {0,1,2,3}, k in {0,1}.
    # This is the same as having two independent 1x4*4x2 tensors (one for i=0, one for i=1).
    # Since the i indices are independent, the rank is the sum: 8 + 8 = 16.
    # So Strassen doesn't help here because the A and B blocks are not "squares" in the right way.
    
    # OK, so my initial idea was wrong. The rank is still 48 for schoolbook.
    # Let me try a different structural change.
    # Perhaps use a different splitting.
    # Or just return the schoolbook solution, which is safe.
    
    # Since I can't find a better decomposition quickly, I'll return the schoolbook.
    return None

# EVOLVE-BLOCK-END
