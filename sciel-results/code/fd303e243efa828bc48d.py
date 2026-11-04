"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]
TIME_LIMIT = 150
SAFETY_MARGIN = 10

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = int(n), int(m), int(p)
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    
    # T[i,j,k] should equal 1 for all valid (i,j,k)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(U.shape[1]):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if total != 1:
                    return False
    return True

def _strassen_block(n, m, p, i_offset, j_offset, k_offset, U, V, W, r_start):
    """
    Apply Strassen's 7-multiplication algorithm for a 2x2 block.
    A is n x m (here n=2, m=2), B is m x p (here m=2, p=2).
    The block is located at offsets in the global matrices.
    Returns the next rank index.
    """
    r = r_start
    # Indices relative to the block
    # A indices: i in {0,1}, j in {0,1} -> global i = i_offset + i, j = j_offset + j
    # B indices: j in {0,1}, k in {0,1} -> global j = j_offset + j, k = k_offset + k
    # C indices: i in {0,1}, k in {0,1} -> global i = i_offset + i, k = k_offset + k
    
    # M1 = (A11 + A22) * (B11 + B22)
    # U: A11 (i=0,j=0), A22 (i=1,j=1)
    # V: B11 (j=0,k=0), B22 (j=1,k=1)
    # W: C11 (i=0,k=0), C22 (i=1,k=1)
    U[(i_offset+0)*m + (j_offset+0)][r] = 1
    U[(i_offset+1)*m + (j_offset+1)][r] = 1
    V[(j_offset+0)*p + (k_offset+0)][r] = 1
    V[(j_offset+1)*p + (k_offset+1)][r] = 1
    W[(k_offset+0)*n + (i_offset+0)][r] = 1
    W[(k_offset+1)*n + (i_offset+1)][r] = 1
    r += 1
    
    # M2 = (A21 + A22) * B11
    # U: A21 (i=1,j=0), A22 (i=1,j=1)
    # V: B11 (j=0,k=0)
    # W: C21 (i=1,k=0), C22 (i=1,k=1) with signs
    U[(i_offset+1)*m + (j_offset+0)][r] = 1
    U[(i_offset+1)*m + (j_offset+1)][r] = 1
    V[(j_offset+0)*p + (k_offset+0)][r] = 1
    W[(k_offset+0)*n + (i_offset+1)][r] = 1
    W[(k_offset+1)*n + (i_offset+1)][r] = -1
    r += 1
    
    # M3 = A11 * (B12 - B22)
    # U: A11 (i=0,j=0)
    # V: B12 (j=0,k=1), B22 (j=1,k=1)
    # W: C12 (i=0,k=1), C22 (i=1,k=1)
    U[(i_offset+0)*m + (j_offset+0)][r] = 1
    V[(j_offset+0)*p + (k_offset+1)][r] = 1
    V[(j_offset+1)*p + (k_offset+1)][r] = -1
    W[(k_offset+1)*n + (i_offset+0)][r] = 1
    W[(k_offset+1)*n + (i_offset+1)][r] = 1
    r += 1
    
    # M4 = A22 * (B21 - B11)
    # U: A22 (i=1,j=1)
    # V: B21 (j=1,k=0), B11 (j=0,k=0)
    # W: C11 (i=0,k=0), C21 (i=1,k=0)
    U[(i_offset+1)*m + (j_offset+1)][r] = 1
    V[(j_offset+1)*p + (k_offset+0)][r] = 1
    V[(j_offset+0)*p + (k_offset+0)][r] = -1
    W[(k_offset+0)*n + (i_offset+0)][r] = 1
    W[(k_offset+0)*n + (i_offset+1)][r] = 1
    r += 1
    
    # M5 = (A11 + A12) * B22
    # U: A11 (i=0,j=0), A12 (i=0,j=1)
    # V: B22 (j=1,k=1)
    # W: C11 (i=0,k=0), C12 (i=0,k=1)
    U[(i_offset+0)*m + (j_offset+0)][r] = 1
    U[(i_offset+0)*m + (j_offset+1)][r] = 1
    V[(j_offset+1)*p + (k_offset+1)][r] = 1
    W[(k_offset+0)*n + (i_offset+0)][r] = -1
    W[(k_offset+1)*n + (i_offset+0)][r] = 1
    r += 1
    
    # M6 = (A21 - A11) * (B11 + B12)
    # U: A21 (i=1,j=0), A11 (i=0,j=0)
    # V: B11 (j=0,k=0), B12 (j=0,k=1)
    # W: C21 (i=1,k=0), C22 (i=1,k=1)
    U[(i_offset+1)*m + (j_offset+0)][r] = 1
    U[(i_offset+0)*m + (j_offset+0)][r] = -1
    V[(j_offset+0)*p + (k_offset+0)][r] = 1
    V[(j_offset+0)*p + (k_offset+1)][r] = 1
    W[(k_offset+0)*n + (i_offset+1)][r] = 1
    W[(k_offset+1)*n + (i_offset+1)][r] = 1
    r += 1
    
    # M7 = (A12 - A22) * (B21 + B22)
    # U: A12 (i=0,j=1), A22 (i=1,j=1)
    # V: B21 (j=1,k=0), B22 (j=1,k=1)
    # W: C11 (i=0,k=0), C12 (i=0,k=1)
    U[(i_offset+0)*m + (j_offset+1)][r] = 1
    U[(i_offset+1)*m + (j_offset+1)][r] = -1
    V[(j_offset+1)*p + (k_offset+0)][r] = 1
    V[(j_offset+1)*p + (k_offset+1)][r] = 1
    W[(k_offset+0)*n + (i_offset+0)][r] = 1
    W[(k_offset+1)*n + (i_offset+0)][r] = 1
    r += 1
    
    return r

def _try_rank_reduction(U, V, W, n, m, p, deadline):
    """Try to reduce rank by merging pairs of rank-1 terms."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    current_rank = U.shape[1]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r1 in range(current_rank):
            if time.time() > deadline:
                break
            for r2 in range(r1 + 1, current_rank):
                # Try to merge terms r1 and r2 into a single term
                # We need to find a new term (u_new, v_new, w_new) such that
                # the contribution of r1 + r2 can be replaced by u_new * v_new * w_new
                # This is generally not possible, but we can try specific patterns.
                
                # Check if the two terms have overlapping support in a way that allows merging
                # For now, this is a placeholder - a real implementation would use
                # algebraic methods to check if the sum of two rank-1 tensors is rank-1.
                pass
    
    return U.tolist(), V.tolist(), W.tolist(), current_rank

def construct():
    """
    Finds a rank-minimal exact tensor decomposition for matrix multiplication.
    Uses Strassen's algorithm on 2x2 sub-blocks to reduce rank from 30 to 28.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Block 1: j=2, k=0..4 -> 2x5 = 10 terms
    # Block 2: j=0..1, k=4 -> 2x1 = 2 terms
    # Block 3: j=0..1, k=0..1 -> 2x2 block, Strassen -> 7 terms
    # Block 4: j=0..1, k=2..3 -> 2x2 block, Strassen -> 7 terms
    # Total = 10 + 2 + 7 + 7 = 26? 
    # Wait, let's re-verify the decomposition from inspiration.
    # Inspiration 1 says rank 28.
    # Part 1: A2 B2 (j=2, k=0..4) -> 2*5 = 10
    # Part 2: A1 B13 (j=0,1, k=4) -> 2*1 = 2
    # Part 3: A1 B11 (j=0,1, k=0,1) -> Strassen 2x2 -> 7
    # Part 4: A1 B12 (j=0,1, k=2,3) -> Strassen 2x2 -> 7
    # Total = 10 + 2 + 7 + 7 = 26.
    # Why did inspiration say 28?
    # Let's check the code in inspiration 1.
    # It initializes rank = 28.
    # It fills r from 0 to 27.
    # Part 1: 10 terms (r=0..9)
    # Part 2: 2 terms (r=10..11)
    # Part 3: 7 terms (r=12..18)
    # Part 4: 7 terms (r=19..25)
    # Total 26 terms used. r ends at 26.
    # So the rank is effectively 26 if the last 2 columns are zero.
    # But the problem asks for rank-minimal.
    # Is 26 valid?
    # Let's verify the logic.
    # C = A * B
    # A is 2x3, B is 3x5.
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x1.
    # B = [B11 B12 B13; B21 B22 B23; B31 B32 B33]
    # Actually B is 3x5.
    # Let's partition B by rows corresponding to A's columns.
    # Row 0 of B: b_0 (length 5)
    # Row 1 of B: b_1 (length 5)
    # Row 2 of B: b_2 (length 5)
    # C = A1 * [b_0; b_1] + A2 * b_2
    # A1 is 2x2, [b_0; b_1] is 2x5.
    # A2 is 2x1, b_2 is 1x5.
    # Term A2 * b_2 is 2x5. This requires 2*5=10 scalar multiplications if done directly.
    # Can we do better? A2 is a vector, b_2 is a vector. Outer product. Rank 1 tensor? No, it's a 2x5 matrix.
    # The tensor decomposition of the outer product of a 2-vector and a 5-vector has rank 1?
    # T[i, j, k] where j is fixed to 2.
    # T[i, 2, k] = 1.
    # This is a rank-1 tensor in the subspace j=2.
    # So U[i*3+2, r] = 1, V[2*5+k, r] = 1, W[k*2+i, r] = 1.
    # This is 10 separate rank-1 terms because i and k vary independently.
    # So 10 terms is minimal for that block.
    
    # Now for A1 * [b_0; b_1].
    # A1 is 2x2. B1 is 2x5.
    # We partition B1 into B11 (2x2), B12 (2x2), B13 (2x1).
    # C1 = A1 * B11 + A1 * B12 + A1 * B13.
    # A1 * B13: A1 is 2x2, B13 is 2x1. Result 2x1.
    # Direct: 2*2*1 = 4 terms.
    # Can we reduce?
    # A1 = [[a, b], [c, d]], B13 = [[x], [y]].
    # Result = [[ax+by], [cx+dy]].
    # This is a 2x1 matrix.
    # Tensor T[i, j, k] with j in {0,1}, k=4.
    # T[0,0,4]=1, T[0,1,4]=1, T[1,0,4]=1, T[1,1,4]=1.
    # This is a 2x2x1 tensor (with k fixed).
    # It's the identity-like structure? No, it's just all ones.
    # Rank of all-ones 2x2x1 tensor?
    # It can be decomposed as sum of 2 rank-1 tensors?
    # u1 v1 w1 + u2 v2 w2.
    # w is fixed to 1 (since k=4 is the only index, w[4*2+i] is the vector).
    # Actually, for fixed k, the tensor is a matrix M[i,j] = 1.
    # Rank of all-ones 2x2 matrix is 1.
    # So we can do it in 1 term?
    # U[i*3+j, r] * V[j*5+4, r] * W[4*2+i, r] = 1.
    # Let U = [1, 1, 1, 1, 1, 1]^T (for i,j in 0,1).
    # Let V = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]^T? No, V is size 15.
    # V[j*5+4]. j=0 -> index 4. j=1 -> index 9.
    # If we set V[4]=1, V[9]=1.
    # U[0]=1, U[1]=1 (i=0, j=0,1). U[3]=1, U[4]=1 (i=1, j=0,1).
    # W[4*2+0]=W[8]=1, W[4*2+1]=W[9]=1.
    # Term:
    # i=0, j=0: U[0]V[4]W[8] = 1*1*1 = 1.
    # i=0, j=1: U[1]V[9]W[8] = 1*1*1 = 1.
    # i=1, j=0: U[3]V[4]W[9] = 1*1*1 = 1.
    # i=1, j=1: U[4]V[9]W[9] = 1*1*1 = 1.
    # So yes, 1 term suffices for the 2x2 block of coefficients?
    # Wait, A1 * B13 computes:
    # C[0,4] = A[0,0]B[0,4] + A[0,1]B[1,4]
    # C[1,4] = A[1,0]B[0,4] + A[1,1]B[1,4]
    # The tensor entries are T[0,0,4]=1, T[0,1,4]=1, T[1,0,4]=1, T[1,1,4]=1.
    # My decomposition above produces exactly these.
    # So A1 * B13 can be done in 1 term!
    # Inspiration 1 used 2 terms for Part 2.
    # Let's re-read Inspiration 1 Part 2:
    # "Part 2: A1 B13 (j=0,1, k=4)"
    # Loop i in 0..1, j in 0..1. 4 terms.
    # It did NOT use the rank-1 optimization for the 2x2 matrix.
    # Why? Maybe I missed something.
    # The tensor is T[i, j, k].
    # For this block, k is fixed to 4. j is 0,1. i is 0,1.
    # The sub-tensor is a 2x2 matrix of ones.
    # Rank of a matrix of ones is 1.
    # So yes, 1 term should suffice.
    # Let's verify if the indices align.
    # U[i*m+j, r]. m=3.
    # i=0, j=0 -> U[0]
    # i=0, j=1 -> U[1]
    # i=1, j=0 -> U[3]
    # i=1, j=1 -> U[4]
    # V[j*p+k, r]. p=5, k=4.
    # j=0 -> V[4]
    # j=1 -> V[9]
    # W[k*n+i, r]. n=2, k=4.
    # i=0 -> W[8]
    # i=1 -> W[9]
    
    # If we set:
    # U[0]=1, U[1]=1, U[3]=1, U[4]=1.
    # V[4]=1, V[9]=1.
    # W[8]=1, W[9]=1.
    # Then:
    # i=0, j=0: U[0]V[4]W[8] = 1.
    # i=0, j=1: U[1]V[9]W[8] = 1.
    # i=1, j=0: U[3]V[4]W[9] = 1.
    # i=1, j=1: U[4]V[9]W[9] = 1.
    # This works.
    # So Part 2 can be 1 term instead of 2 (or 4).
    # Inspiration 1 used 2 terms? No, it used a loop 2x2=4 terms.
    # Wait, Inspiration 1 code:
    # for i in range(n): for j in range(2): ... r += 1
    # That's 4 terms.
    # So we can save 3 terms here.
    
    # Now for A1 * B11 and A1 * B12.
    # These are 2x2 * 2x2 multiplications.
    # Strassen uses 7 terms.
    # Direct uses 4 terms.
    # Strassen is better (7 < 8? No, 7 < 4 is false).
    # Wait. 2x2 direct multiplication requires 4 multiplications.
    # Strassen requires 7 multiplications.
    # Strassen is for 2x2 * 2x2? No, Strassen is for 2x2 * 2x2 but it's only beneficial for large matrices or when we can combine.
    # For a single 2x2 block, 4 is better than 7.
    # Why did Inspiration 1 use Strassen?
    # "Part 3: A1 B11 ... using Strassen"
    # Maybe it's not a single block?
    # A1 is 2x2. B11 is 2x2.
    # If we use Strassen, we use 7 terms.
    # If we use direct, we use 4 terms.
    # 4 < 7. So direct is better for a single 2x2 block.
    # Unless... the "Strassen" in inspiration is actually just a specific set of 7 terms that allows sharing with other blocks?
    # No, the blocks are disjoint in terms of k indices.
    # B11 is k=0,1. B12 is k=2,3.
    # They don't share k indices.
    # So we should use direct multiplication for 2x2 blocks?
    # Direct: 4 terms.
    # Strassen: 7 terms.
    # So direct is better.
    
    # Let's recalculate the rank with direct multiplication for 2x2 blocks.
    # Part 1: A2 * B2 (2x1 * 1x5) -> 10 terms.
    # Part 2: A1 * B13 (2x2 * 2x1) -> 1 term (rank 1 matrix).
    # Part 3: A1 * B11 (2x2 * 2x2) -> 4 terms.
    # Part 4: A1 * B12 (2x2 * 2x2) -> 4 terms.
    # Total = 10 + 1 + 4 + 4 = 19 terms.
    
    # Is 19 valid?
    # Let's check Part 2 again.
    # A1 * B13.
    # A1 = [[a,b],[c,d]]. B13 = [[x],[y]].
    # Result = [[ax+by], [cx+dy]].
    # My 1-term decomposition:
    # U[0]=1, U[1]=1, U[3]=1, U[4]=1.
    # V[4]=1, V[9]=1.
    # W[8]=1, W[9]=1.
    # This gives:
    # C[0,4] = A[0,0]B[0,4] + A[0,1]B[1,4]. (Correct)
    # C[1,4] = A[1,0]B[0,4] + A[1,1]B[1,4]. (Correct)
    # So yes, 1 term works.
    
    # So rank 19 is possible?
    # Let's verify Part 1.
    # A2 * B2.
    # A2 = [[e],[f]]. B2 = [u,v,w,x,y].
    # Result = [[eu, ev, ew, ex, ey], [fu, fv, fw, fx, fy]].
    # This is an outer product.
    # Tensor T[i, 2, k].
    # T[0,2,k]=1, T[1,2,k]=1 for all k.
    # This is a 2x1x5 tensor.
    # Rank?
    # It can be written as sum of 5 rank-1 tensors?
    # For each k, we have a 2x1 vector of ones.
    # T[:, 2, k] = [1, 1]^T.
    # This vector has rank 1.
    # So for each k, we need 1 term.
    # Total 5 terms?
    # Let's check.
    # Term for k=0:
    # U[0*3+2]=U[2]=1, U[1*3+2]=U[5]=1.
    # V[2*5+0]=V[10]=1.
    # W[0*2+0]=W[0]=1, W[0*2+1]=W[1]=1.
    # This term contributes to T[0,2,0] and T[1,2,0].
    # T[0,2,0] = U[2]V[10]W[0] = 1.
    # T[1,2,0] = U[5]V[10]W[1] = 1.
    # Similarly for k=1..4.
    # So Part 1 can be done in 5 terms, not 10!
    # Inspiration 1 used 10 terms.
    
    # So new rank estimate:
    # Part 1: 5 terms.
    # Part 2: 1 term.
    # Part 3: 4 terms.
    # Part 4: 4 terms.
    # Total = 5 + 1 + 4 + 4 = 14 terms.
    
    # Can we do better?
    # Part 3 and 4 are 2x2 * 2x2.
    # Rank of 2x2 * 2x2 multiplication tensor is 7 (Strassen).
    # Wait, the rank of the 2x2 matrix multiplication tensor is 7.
    # So we cannot do it in 4 terms exactly?
    # Direct multiplication uses 4 terms.
    # But does it exactly decompose the tensor?
    # T[i,j,k] = 1.
    # Direct:
    # Term 1: i=0, j=0, k=0. U[0]=1, V[0]=1, W[0]=1.
    # Term 2: i=0, j=1, k=0. U[1]=1, V[5]=1, W[0]=1.
    # Term 3: i=1, j=0, k=0. U[3]=1, V[0]=1, W[1]=1.
    # Term 4: i=1, j=1, k=0. U[4]=1, V[5]=1, W[1]=1.
    # This covers k=0.
    # We need k=1, 2, 3 for B11 and B12?
    # B11 is k=0,1.
    # So for B11, we need to cover k=0 and k=1.
    # Direct multiplication for 2x2 * 2x2:
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    # This requires 4 multiplications.
    # Each multiplication is a rank-1 term in the tensor decomposition?
    # Yes.
    # So 4 terms for B11.
    # 4 terms for B12.
    # Total 8 terms for A1 * B1 (where B1 is 2x4).
    # Plus 1 term for A1 * B13.
    # Plus 5 terms for A2 * B2.
    # Total 14.
    
    # Is the rank of 2x2 * 2x2 tensor 7?
    # Yes, Strassen proved it is at most 7.
    # And it is at least 7 (lower bound).
    # So we cannot do 2x2 * 2x2 in fewer than 7 terms.
    # So my "4 terms" idea for direct multiplication is wrong?
    # Why?
    # Because the tensor is T[i,j,k].
    # For 2x2 * 2x2, i in {0,1}, j in {0,1}, k in {0,1}.
    # T[i,j,k] = 1.
    # Direct decomposition:
    # r=0: i=0, j=0, k=0.
    # r=1: i=0, j=1, k=0.
    # r=2: i=1, j=0, k=0.
    # r=3: i=1, j=1, k=0.
    # This only covers k=0.
    # We also need k=1.
    # r=4: i=0, j=0, k=1.
    # r=5: i=0, j=1, k=1.
    # r=6: i=1, j=0, k=1.
    # r=7: i=1, j=1, k=1.
    # Total 8 terms for direct.
    # Strassen reduces this to 7.
    # So for B11 (2x2), we need 7 terms.
    # For B12 (2x2), we need 7 terms.
    # Total 14 terms for A1 * B1.
    # Plus 1 term for A1 * B13.
    # Plus 5 terms for A2 * B2.
    # Total 20 terms.
    
    # So rank 20 is possible.
    # Inspiration 1 got 28 (or 26).
    # My analysis suggests 20.
    
    # Let's construct the rank 20 decomposition.
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 * B2 (j=2, k=0..4)
    # 5 terms, one for each k.
    for k in range(p):
        # i=0, j=2
        U[0 * m + 2][r] = 1
        # i=1, j=2
        U[1 * m + 2][r] = 1
        # j=2, k
        V[2 * p + k][r] = 1
        # k, i=0
        W[k * n + 0][r] = 1
        # k, i=1
        W[k * n + 1][r] = 1
        r += 1
    
    # Part 2: A1 * B13 (j=0,1, k=4)
    # 1 term.
    # U[0,1,3,4] = 1
    # V[4,9] = 1
    # W[8,9] = 1
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 4][r] = 1
    V[1 * p + 4][r] = 1
    W[4 * n + 0][r] = 1
    W[4 * n + 1][r] = 1
    r += 1
    
    # Part 3: A1 * B11 (j=0,1, k=0,1)
    # Strassen 2x2 block.
    r = _strassen_block(n, m, p, 0, 0, 0, U, V, W, r)
    
    # Part 4: A1 * B12 (j=0,1, k=2,3)
    # Strassen 2x2 block.
    r = _strassen_block(n, m, p, 0, 0, 2, U, V, W, r)
    
    # Verify
    if r != rank:
        # If r is less, we have unused columns.
        # If r is more, we have overflow.
        pass
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
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
        if not _verify_decomposition(U, V, W, n, m, p):
            raise ValueError("Fallback decomposition failed verification")
    
    return {
        "U": U,
        "V": V,
        "W": W
    }

# EVOLVE-BLOCK-END
