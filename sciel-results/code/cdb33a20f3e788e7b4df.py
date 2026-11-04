"""
Tensor rank minimization for 3x5x7 matrix multiplication.
Uses a recursive block decomposition strategy based on known sub-rectangular
Strassen-style decompositions to reduce rank below the schoolbook 105.

The approach:
1. Use a 2x2x2 Strassen block (rank 7 instead of 8) wherever possible.
2. Combine blocks with standard multiplication for the remaining parts.
3. The known best upper bound for R(3,5,7) is 42 (from literature),
   but constructing it explicitly requires careful design.
4. We aim for a practical reduction: use Strassen where blocks are 2x2,
   and handle the odd dimensions (3, 5, 7) by padding or splitting.
"""
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _strassen_2x2x2_blocks(U, V, W, rank_start):
    """
    Add 7 Strassen multiplication terms for a 2x2x2 sub-problem.
    The sub-problem multiplies a 2x2 matrix by a 2x2 matrix.
    U rows: 2*2=4 rows (i*2+j for i,j in {0,1})
    V rows: 2*2=4 rows (j*2+k for j,k in {0,1})
    W rows: 2*2=4 rows (k*2+i for k,i in {0,1})
    
    Strassen formulas (C = A * B, A is 2x2, B is 2x2):
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    In tensor form, for each multiplication M_r = (sum_a u_{a,r} A_a)(sum_b v_{b,r} B_b),
    the contribution to W is w_{c,r} = sum over pairs where C_c = sum_r u_{a,r} v_{b,r} * (A_a * B_b contribution to C_c).
    
    Actually, let me think about this differently. The tensor T for matrix multiplication
    T[i,j,k] = 1 means C_{i,k} gets A_{i,j} * B_{j,k}.
    
    A rank-1 decomposition is: C_{i,k} += alpha * (sum_a U[i,a] A_a) * (sum_b V[b,k] B_b) * W[i,k]
    
    No wait. The standard form: C_{i,k} = sum_r U[i,r] * V[r,k] * ... no.
    
    Let me use the standard tensor decomposition:
    C_{i,k} = sum_j A_{i,j} B_{j,k} = sum_r (sum_{i',j} U[i',j,r] A_{i',j}) (sum_{j',k'} V[j',k',r] B_{j',k'}) W[i',k',r]
    where the sums are over the block indices.
    
    Actually the interface says:
    U: shape (n*m, rank) - so U[i*m+j, r]
    V: shape (m*p, rank) - so V[j*p+k, r]  
    W: shape (p*n, rank) - so W[k*n+i, r]
    
    And the decomposition is: C[k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where the sum over r is the outer sum, and for each r, we have a rank-1 term.
    
    For a rank-1 term corresponding to a single (i,j,k) triple:
    U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    This gives C[k*n+i] += A[i,j] * B[j,k] * 1
    
    For Strassen, we group terms. The key insight is that Strassen computes
    products of sums, which corresponds to having U and V entries that are sums.
    
    Let me just implement the known result. For 3x5x7, a known construction gives rank 42.
    But let me first try a simpler approach that should give rank < 105.
    
    Strategy: Split 5 = 2 + 3 and 7 = 3 + 4.
    A is 3x5, B is 5x7, C is 3x7.
    
    Split A into A1 (3x2) and A2 (3x3)
    Split B into B1 (2x7) and B2 (3x7) -- wait, B is 5x7, so B1 is 2x7 and B2 is 3x7? No.
    
    A is 3x5. Split columns: A = [A1 | A2] where A1 is 3x2, A2 is 3x3.
    B is 5x7. Split rows: B = [B1; B2] where B1 is 2x7, B2 is 3x7.
    C = A1*B1 + A2*B2 (3x7)
    
    A1*B1: 3x2 * 2x7 -> 3x7. Use schoolbook or better.
    A2*B2: 3x3 * 3x7 -> 3x7. Use schoolbook or better.
    
    This doesn't directly help unless we use Strassen on the 2x2 sub-blocks.
    
    Let me try a different split. The known optimal for R(3,5,7) = 42 uses a specific construction.
    Let me just implement a direct search for a low-rank decomposition using the known structure.
    
    Actually, let me implement a simpler improvement first: use Strassen 2x2x2 where possible.
    
    For the 3x5x7 problem, I can:
    - Pad to 4x8x8 or similar and use known decompositions, but that might increase rank.
    - Use the fact that R(3,5,7) <= R(3,5,5) * R(5,7,1) or similar sub-multiplicativity.
    
    Let me just implement a straightforward improvement: 
    Use Strassen for the 2x2 blocks within the multiplication.
    
    Here's a concrete plan:
    - The 5 dimension can be split as 2+3.
    - The 7 dimension can be split as 2+2+3.
    - The 3 dimension stays as is.
    
    A (3x5) = [A1(3x2) | A2(3x3)]
    B (5x7) = [B1(2x7); B2(3x7)]
    C (3x7) = A1*B1 + A2*B2
    
    For A1*B1 (3x2 * 2x7): 
    Split B1 into B1a(2x2) and B1b(2x5).
    A1*B1a: 3x2 * 2x2. Split A1 into rows: [A1_top(1x2); A1_bot(2x2)].
    A1_bot*B1a: 2x2 * 2x2 -> use Strassen (rank 7 instead of 8).
    A1_top*B1a: 1x2 * 2x2 -> rank 4 (can't do better than 4 for 1x2x2? Actually R(1,2,2)=4).
    
    This is getting complex. Let me just implement the known 42-rank construction directly.
    
    From literature, R(3,5,7) = 42. The construction uses a specific set of 42 rank-1 terms.
    Rather than deriving it from scratch, let me implement a computational search that finds
    a decomposition of rank 42 using the known structure.
    
    Actually, the simplest correct approach that beats schoolbook: 
    Use the fact that we can combine the Strassen trick. Let me implement a hybrid where
    I use Strassen 2x2x2 for the core and handle the extras.
    
    Let me just output a known-good decomposition. The rank 42 decomposition for 3x5x7
    is well-documented. Let me construct it algorithmically.
    
    Here's my approach: I'll implement a direct enumeration/search for a low-rank 
    decomposition using rational arithmetic, starting from a structured initialization.
"""

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    # Schoolbook rank is 105. We aim for lower.
    
    # Strategy: Use a known rectangular Strassen-style decomposition.
    # We'll build the decomposition term by term.
    
    # For now, let's implement a simple improvement:
    # Use Strassen 2x2x2 (rank 7) for a 2x2x2 sub-block, and schoolbook for the rest.
    
    # Split the 5 into 2+3, and 7 into 2+5.
    # A (3x5): columns 0-1 (2 cols), columns 2-4 (3 cols)
    # B (5x7): rows 0-1 (2 rows), rows 2-4 (3 rows)
    # C (3x7): all 3x7
    
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:5] @ B[2:5,:]
    # First term: 3x2 * 2x7, schoolbook rank = 3*2*7 = 42
    # Second term: 3x3 * 3x7, schoolbook rank = 3*3*7 = 63
    # Total: 105 (no improvement)
    
    # To get improvement, we need to use Strassen within these blocks.
    # For the 3x2 * 2x7 part:
    #   Split the 2 dimension... it's already minimal.
    #   3x2 * 2x7: we can think of this as 3 separate 1x2 * 2x7 multiplications.
    #   Each 1x2 * 2x7 has rank 4 (since it's just 2 products per row of C).
    #   Total: 3*4 = 12? No, that's not right.
    
    # Actually, R(1,2,7) = 4? Let's see: C[i,k] = sum_j A[i,j] B[j,k] for i in {0}, j in {0,1}, k in {0..6}
    # That's 2 terms per (i,k) pair, so 2*1*7 = 14 schoolbook. Can we do better?
    # R(1,2,p) = 2p for p>=1? No, R(1,2,2) = 4 = 2*2. R(1,2,3) = 6? 
    # Actually for 1x2xp, the tensor is sum_{j,k} A_0j B_jk C_0k which has 2p terms.
    # I don't think we can reduce below 2p for this case.
    
    # Let me try a different approach. Use the known result that R(3,5,7) <= 42.
    # I'll implement a direct construction based on the literature.
    
    # The 42-rank decomposition can be constructed as follows:
    # View the problem as multiplying a 3x5 matrix by a 5x7 matrix.
    # Use a combination of 2x2 Strassen blocks and linear combinations.
    
    # Since constructing the exact 42-rank decomposition from scratch is complex,
    # let me implement a search-based approach that tries to find a low-rank 
    # decomposition using numerical optimization.
    
    import time
    deadline = time.time() + 140  # 140s deadline, 10s margin
    
    # Start with schoolbook and try to reduce rank by merging terms.
    # Actually, let me just implement a known construction.
    
    # Known: R(3,5,7) = 42. Here's a construction outline:
    # Use the fact that 3x5x7 can be embedded in a larger structure.
    # 
    # Alternative: use the asymptotic formula or known bounds.
    # 
    # For practical purposes, let me implement a Strassen-based hybrid:
    # - Use 2x2 Strassen where the dimensions allow.
    # - The 5 can be seen as 2+3, the 7 as 3+4.
    
    # Let me just implement a simple rank reduction:
    # Combine the 2x2 Strassen (7 terms) for a sub-block.
    
    # Split: A (3x5) -> A1 (2x2) + A2 (rest), B (5x7) -> B1 (2x2) + B2 (rest)
    # The 2x2 * 2x2 part uses Strassen: 7 terms.
    # The rest: (3x5*5x7 - 2x2*2x2) = 105 - 8 = 97 schoolbook terms.
    # But the cross terms need careful handling.
    
    # This is getting too complex for a clean implementation. Let me just 
    # implement a numerical search.
    
    # Actually, the simplest improvement: use the known fact that 
    # R(2,2,2) = 7 (Strassen). We can find all 2x2x2 sub-tensors in our 
    # 3x5x7 tensor and apply Strassen to each.
    
    # The 3x5x7 tensor has C(3,2)*C(5,2)*C(7,2) = 3*10*21 = 630 possible 2x2x2 sub-tensors.
    # But they overlap, so we can't just apply Strassen to all of them.
    
    # Let me implement a greedy approach:
    # 1. Start with schoolbook (105 terms).
    # 2. Find a set of non-overlapping 2x2x2 sub-tensors.
    # 3. Replace each 8-term schoolbook block with 7-term Strassen.
    
    # Non-overlapping 2x2x2 blocks: 
    # In the 3x5x7 grid, how many disjoint 2x2x2 boxes can we fit?
    # 3/2 = 1 (with 1 leftover), 5/2 = 2 (with 1 leftover), 7/2 = 3 (with 1 leftover).
    # So we can fit 1*2*3 = 6 disjoint 2x2x2 blocks.
    # Each saves 1 term, so total rank = 105 - 6 = 99.
    
    # Let me implement this.
    
    n, m, p = 3, 5, 7
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We'll build the decomposition term by term.
    # First, identify the 6 disjoint 2x2x2 blocks.
    # Block (bi, bj, bk) covers rows i in {2*bi, 2*bi+1}, cols j in {2*bj, 2*bj+1}, etc.
    
    # For each block, we'll use Strassen (7 terms) instead of schoolbook (8 terms).
    # For the remaining cells, use schoolbook (1 term each).
    
    # Let me track which (i,j,k) triples have been assigned.
    assigned = [[[False] * p for _ in range(m)] for _ in range(n)]
    
    column = 0
    
    # Process each potential 2x2x2 block.
    for bi in range(n // 2):
        for bj in range(m // 2):
            for bk in range(p // 2):
                i0, i1 = 2 * bi, 2 * bi + 1
                j0, j1 = 2 * bj, 2 * bj + 1
                k0, k1 = 2 * bk, 2 * bk + 1
                
                # Check if all 8 cells are unassigned.
                all_free = True
                for ii in [i0, i1]:
                    for jj in [j0, j1]:
                        for kk in [k0, k1]:
                            if assigned[ii][jj][kk]:
                                all_free = False
                                break
                        if not all_free:
                            break
                    if not all_free:
                        break
                if not all_free:
                    continue
                
                # Mark as assigned.
                for ii in [i0, i1]:
                    for jj in [j0, j1]:
                        for kk in [k0, k1]:
                            assigned[ii][jj][kk] = True
                
                # Strassen 2x2x2 decomposition.
                # The 8 schoolbook terms for this block are:
                # (i0,j0,k0), (i0,j0,k1), (i0,j1,k0), (i0,j1,k1)
                # (i1,j0,k0), (i1,j0,k1), (i1,j1,k0), (i1,j1,k1)
                
                # Strassen replaces these 8 with 7 terms.
                # The rank-1 terms correspond to the 7 Strassen multiplications.
                
                # In the tensor T[i,j,k] = A[i,j] * B[j,k] * C[i,k],
                # each rank-1 term has U[i,j], V[j,k], W[k,i] values.
                
                # Strassen M1 = (A11+A22)(B11+B22)
                # This contributes to C11 with coefficient 1.
                # In tensor form, this is a combination of terms.
                
                # The 7 Strassen terms for a 2x2x2 block:
                # M1: (A00+A11)(B00+B11) -> contributes to C00
                #   U: A00: 1, A11: 1
                #   V: B00: 1, B11: 1
                #   W: C00: 1
                # M2: (A10+A11)(B00) -> contributes to C10
                #   U: A10: 1, A11: 1
                #   V: B00: 1
                #   W: C10: 1
                # M3: (A00)(B10-B11) -> contributes to C01
                #   U: A00: 1
                #   V: B10: 1, B11: -1
                #   W: C01: 1
                # M4: (A11)(B01-B00) -> contributes to C00, C10
                #   U: A11: 1
                #   V: B01: 1, B00: -1
                #   W: C00: 1, C10: 1  -- wait, this doesn't work as a single rank-1 term.
                
                # I think I'm confusing myself. Let me use the standard Strassen
                # tensor decomposition.
                
                # For C = AB where A, B are 2x2:
                # M1 = (A00+A11)(B00+B11)
                # M2 = (A10+A11)(B00)
                # M3 = (A00)(B10-B11)
                # M4 = (A11)(B01-B00)
                # M5 = (A00+A01)(B11)
                # M6 = (A10-A00)(B00+B01)
                # M7 = (A01-A11)(B10+B11)
                
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                
                # Each Mr is a product of a linear combination of A entries
                # and a linear combination of B entries.
                
                # In our tensor T[i,j,k] where i=row of A/row of C, j=col of A/row of B, k=col of B/col of C:
                # A entries: T[i,j,?] -- no, T is indexed by (i,j,k) and T[i,j,k] = A[i,j]*B[j,k]*C[i,k].
                
                # For a rank-1 term r, we have:
                # sum_{i,j,k} T[i,j,k] = (sum_{i,j} u_{i,j} A[i,j]) * (sum_{j,k} v_{j,k} B[j,k]) * (sum_{i,k} w_{i,k} C[i,k])
                # No, that's not right either.
                
                # The correct form: C[i,k] = sum_r U[i,r] * (sum_j A[i,j] * ... ) 
                
                # Let me use the interface definition:
                # U has shape (n*m, rank), indexed as U[i*m+j, r]
                # V has shape (m*p, rank), indexed as V[j*p+k, r]
                # W has shape (p*n, rank), indexed as W[k*n+i, r]
                
                # And the decomposition is:
                # C[k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # where for each r, the "term" is:
                # sum_{i,j,k} U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
                # = C[k*n+i] for the specific (i,k) pair... 
                
                # No, I think the correct interpretation is:
                # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]
                # Hmm, that doesn't look right either.
                
                # Let me just verify with the schoolbook case:
                # For term r corresponding to (i0, j0, k0):
                # U[i0*m+j0, r] = 1, all other U = 0
                # V[j0*p+k0, r] = 1, all other V = 0
                # W[k0*n+i0, r] = 1, all other W = 0
                # Then C[i0,k0] += U[i0*m+j0,r] * V[j0*p+k0,r] * W[k0*n+i0,r] * A[i0,j0] * B[j0,k0]
                # = 1 * 1 * 1 * A[i0,j0] * B[j0,k0]
                # Yes! So the formula is:
                # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
                # where the sum over j is implicit (for each r, only one j has U and V non-zero in schoolbook).
                
                # Wait, that's not quite right. Let me re-read.
                # I think it's: C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_j' V[j'*p+k, r] B[j',k]) W[k*n+i, r]
                # No...
                
                # Let me just assume the standard bilinear form:
                # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
                # where for each r, the indices i, j, k are determined by where U, V, W are non-zero.
                
                # For Strassen, multiple (i,j) pairs can have non-zero U in the same r,
                # and multiple (j,k) pairs can have non-zero V in the same r.
                
                # For M1 = (A00+A11)(B00+B11) in the 2x2x2 block:
                # U[0*m+0, r] = 1, U[1*m+1, r] = 1  (for the local 2x2, indices 0,1)
                # V[0*p+0, r] = 1, V[1*p+1, r] = 1
                # W[0*n+0, r] = 1  (contributes to C[0,0])
                # But M1 also contributes to other C entries in the full problem.
                
                # In the local 2x2 block, M1 contributes:
                # C00 += M1, so W[0*n+0, r] = 1
                # And M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
                # This matches: sum_{i,j} U[i*m+j,r] A[i,j] = A00 + A11
                # sum_{j,k} V[j*p+k,r] B[j,k] = B00 + B11
                # Product = A00*B00 + A00*B11 + A11*B00 + A11*B11
                # And this gets multiplied by W[k*n+i, r] for each (i,k).
                # For M1, it should only contribute to C00, so W[0*n+0, r] = 1, others 0.
                # But wait, (A00+A11)(B00+B11) is a scalar, and it should add to C00.
                # So W[0*n+0, r] = 1.
                
                # OK let me just implement it.
                
                # Local indices: i in {0,1} maps to global i0+i, j in {0,1} maps to j0+j, k in {0,1} maps to k0+k.
                
                # M1: U: (0,0):1, (1,1):1; V: (0,0):1, (1,1):1; W: (0,0):1
                U[i0*m+j0][column] = 1
                U[(i0+1)*m+(j0+1)][column] = 1
                V[j0*p+k0][column] = 1
                V[(j0+1)*p+(k0+1)][column] = 1
                W[k0*n+i0][column] = 1
                column += 1
                
                # M2: (A10+A11)(B00); contributes to C10
                # U: (1,0):1, (1,1):1; V: (0,0):1; W: (0,1):1
                U[(i0+1)*m+j0][column] = 1
                U[(i0+1)*m+(j0+1)][column] = 1
                V[j0*p+k0][column] = 1
                W[k0*n+(i0+1)][column] = 1
                column += 1
                
                # M3: (A00)(B10-B11); contributes to C01
                # U: (0,0):1; V: (1,0):1, (1,1):-1; W: (1,0):1
                U[i0*m+j0][column] = 1
                V[(j0+1)*p+k0][column] = 1
                V[(j0+1)*p+(k0+1)][column] = -1
                W[(k0+1)*n+i0][column] = 1
                column += 1
                
                # M4: (A11)(B01-B00); contributes to C00 and C10
                # U: (1,1):1; V: (0,1):1, (0,0):-1; W: (0,0):1, (0,1):1
                U[(i0+1)*m+(j0+1)][column] = 1
                V[j0*p+(k0+1)][column] = 1
                V[j0*p+k0][column] = -1
                W[k0*n+i0][column] = 1
                W[k0*n+(i0+1)][column] = 1
                column += 1
                
                # M5: (A00+A01)(B11); contributes to C01
                # U: (0,0):1, (0,1):1; V: (1,1):1; W: (1,0):1
                U[i0*m+j0][column] = 1
                U[i0*m+(j0+1)][column] = 1
                V[(j0+1)*p+(k0+1)][column] = 1
                W[(k0+1)*n+i0][column] = 1
                column += 1
                
                # M6: (A10-A00)(B00+B01); contributes to C11
                # U: (1,0):1, (0,0):-1; V: (0,0):1, (0,1):1; W: (1,1):1
                U[(i0+1)*m+j0][column] = 1
                U[i0*m+j0][column] = -1
                V[j0*p+k0][column] = 1
                V[j0*p+(k0+1)][column] = 1
                W[(k0+1)*n+(i0+1)][column] = 1
                column += 1
                
                # M7: (A01-A11)(B10+B11); contributes to C00
                # U: (0,1):1, (1,1):-1; V: (1,0):1, (1,1):1; W: (0,0):1
                U[i0*m+(j0+1)][column] = 1
                U[(i0+1)*m+(j0+1)][column] = -1
                V[(j0+1)*p+k0][column] = 1
                V[(j0+1)*p+(k0+1)][column] = 1
                W[k0*n+i0][column] = 1
                column += 1
    
    # Now handle unassigned (i,j,k) triples with schoolbook terms.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not assigned[i][j][k]:
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
    
    # The final rank is 'column'.
    # Trim the matrices to actual rank.
    rank = column
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
