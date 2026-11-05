"""Rewritten program for 2x5x5 matrix multiplication tensor rank minimization."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

DEADLINE = time.time() + 135  # 15s safety margin before 150s kill

def _schoolbook(n=2, m=5, p=5):
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=2, m=5, p=5):
    """Verify exact tensor decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _strassen_222(U, V, W, i_off, j_off, k_off, n, m, p, col_start):
    """Add Strassen 2x2x2x2 decomposition for sub-blocks.
    A sub-block: rows i_off..i_off+1 of A (2 rows), cols j_off..j_off+1 (2 cols)
    B sub-block: rows j_off..j_off+1 (2 rows), cols k_off..k_off+1 (2 cols)
    C sub-block: rows i_off..i_off+1 (2 rows), cols k_off..k_off+1 (2 cols)
    
    Strassen's 7 products for 2x2 matrix mult:
    Let X = A[i_off:i_off+2, j_off:j_off+2], Y = B[j_off:j_off+2, k_off:k_off+2]
    X11=X[0,0], X12=X[0,1], X21=X[1,0], X22=X[1,1]
    Y11=Y[0,0], Y12=Y[0,1], Y21=Y[1,0], Y22=Y[1,1]
    
    M1 = (X11+X22)*(Y11+Y22)
    M2 = (X21+X22)*Y11
    M3 = X11*(Y12-Y22)
    M4 = X22*(Y21-Y11)
    M5 = (X11+X12)*Y22
    M6 = (X21-X11)*(Y11+Y12)
    M7 = (X12-X22)*(Y21+Y22)
    
    Z11 = M1+M4-M5+M7
    Z12 = M3+M5
    Z21 = M2+M4
    Z22 = M1-M2+M3+M6
    """
    # Each Mi is a product of a linear form in U entries and a linear form in V entries
    # For rank-1 decomposition, each Mi contributes one column.
    # Mi = (linear combo of U entries) * (linear combo of V entries) * (linear combo of W entries)
    
    # For M1 = (X11+X22)*(Y11+Y22):
    # X11 is A[i_off, j_off], X22 is A[i_off+1, j_off+1]
    # Y11 is B[j_off, k_off], Y22 is B[j_off+1, k_off+1]
    # The product contributes to Z11 and Z22
    # U entries: A[i_off,j_off]=1, A[i_off+1,j_off+1]=1
    # V entries: B[j_off,k_off]=1, B[j_off+1,k_off+1]=1
    # W entries: need to distribute so that sum gives correct Z
    
    # Actually, for a rank-1 term: u[i,j]*v[j,k]*w[k,i]
    # For M1 to contribute to Z11 = C[i_off, k_off]:
    #   We need u[i_off,j_off]*v[j_off,k_off]*w[k_off,i_off] to appear
    #   And u[i_off+1,j_off+1]*v[j_off+1,k_off+1]*w[k_off+1,i_off+1] to appear
    # But these are different (i,j,k) triples!
    
    # The issue: a single rank-1 column can only have one (i,j) for U, one (j,k) for V, one (k,i) for W.
    # So M1 as written doesn't directly give a rank-1 term.
    
    # Instead, Strassen works by defining new intermediate matrices and then
    # the products are scalar. In the tensor decomposition context, we need
    # to think of it differently.
    
    # For the full 2x2x2x2 multiplication tensor, Strassen gives rank 7.
    # The 7 rank-1 terms are:
    # Term 1: U has entries at (i_off,j_off) and (i_off+1,j_off+1) with coeff 1
    #          V has entries at (j_off,k_off) and (j_off+1,k_off+1) with coeff 1
    #          W has entries at (k_off,i_off) and (k_off+1,i_off+1) with coeff 1
    #          But this gives products at (i_off,j_off,k_off), (i_off,j_off+1,k_off+1), etc.
    
    # Let me think about this more carefully using the standard Strassen decomposition.
    # The 7 terms for multiplying 2x2 matrices X*Y=Z:
    # Each term t is: t = (linear form in X) * (linear form in Y)
    # In tensor form, each term contributes to specific (i,j,k) positions.
    
    # For our purposes, let's just enumerate the 7 rank-1 contributions:
    # Term c: U_c[i,j], V_c[j,k], W_c[k,i]
    
    # Standard Strassen rank-1 decomposition for 2x2x2x2:
    # Using the 7 products, each product corresponds to a rank-1 term where
    # the U, V, W entries are the coefficients from the linear forms.
    
    # Let's define the 7 terms explicitly:
    # Let i0=i_off, i1=i_off+1, j0=j_off, j1=j_off+1, k0=k_off, k1=k_off+1
    
    i0, i1 = i_off, i_off + 1
    j0, j1 = j_off, j_off + 1
    k0, k1 = k_off, k_off + 1
    
    # Term 1: M1 = (X11+X22)(Y11+Y22)
    # U: (i0,j0)=1, (i1,j1)=1
    # V: (j0,k0)=1, (j1,k1)=1
    # W: (k0,i0)=1, (k1,i1)=1
    # This gives: 1*1*1 at (i0,j0,k0), 1*1*1 at (i0,j1,k1), 1*1*1 at (i1,j0,k0), 1*1*1 at (i1,j1,k1)
    # Wait, that's not right. The product u[i,j]*v[j,k]*w[k,i] for a single column:
    # If U has nonzero at (i0,j0) and (i1,j1), V at (j0,k0) and (j1,k1), W at (k0,i0) and (k1,i1):
    # (i0,j0,k0): U[i0,j0]*V[j0,k0]*W[k0,i0] = 1*1*1 = 1
    # (i0,j0,k1): U[i0,j0]*V[j0,k1]*W[k1,i0] = 1*0*0 = 0 (V[j0,k1]=0, W[k1,i0]=0)
    # (i0,j1,k0): U[i0,j1]*V[j1,k0]*W[k0,i0] = 0*0*1 = 0
    # (i0,j1,k1): U[i0,j1]*V[j1,k1]*W[k1,i0] = 0*1*0 = 0
    # (i1,j0,k0): U[i1,j0]*V[j0,k0]*W[k0,i1] = 0*1*0 = 0
    # (i1,j0,k1): U[i1,j0]*V[j0,k1]*W[k1,i1] = 0*0*1 = 0
    # (i1,j1,k0): U[i1,j1]*V[j1,k0]*W[k0,i1] = 1*0*0 = 0
    # (i1,j1,k1): U[i1,j1]*V[j1,k1]*W[k1,i1] = 1*1*1 = 1
    # So this term contributes 1 at (i0,j0,k0) and (i1,j1,k1), 0 elsewhere.
    # That's not M1!
    
    # I see the issue. The rank-1 decomposition of the 2x2x2x2 tensor via Strassen
    # is more subtle. Each of the 7 "products" in Strassen corresponds to a
    # bilinear form, but in the tensor decomposition, each rank-1 term is
    # u[i,j]*v[j,k]*w[k,i] which is a trilinear form.
    
    # The correct way: For 2x2x2x2, the tensor T[i,j,k] = 1 for all i,j,k in {0,1}.
    # Strassen shows rank <= 7. The 7 rank-1 terms are:
    
    # Let me look up the explicit Strassen rank-1 decomposition for 2x2x2x2.
    # The 7 terms (using 0-indexed):
    # t1: U=(1,1;1,1) pattern... 
    
    # Actually, let me just use the known result. For 2x2x2x2 matrix multiplication,
    # the Strassen decomposition has 7 terms. Each term c has:
    # U_c is a 2x2 matrix, V_c is a 2x2 matrix, W_c is a 2x2 matrix
    # such that for all i,j,k: sum_c U_c[i,j]*V_c[j,k]*W_c[k,i] = 1
    
    # The 7 terms (one common formulation):
    # c=0: U=[[1,0],[0,1]], V=[[1,1],[1,1]], W=[[1,1],[1,1]] -> check: (0,0,0):1*1*1=1, (0,0,1):1*1*1=1, ... not all 1s
    # This is getting complicated. Let me just use a known explicit decomposition.
    
    # Known Strassen decomposition for 2x2x2x2 (from literature):
    # Using the 7 products, the rank-1 terms are:
    
    # I'll use a different approach: directly construct the 7 terms.
    # The tensor T[i,j,k]=1 for i,j,k in {0,1}.
    # We need sum_c u_c[i,j] v_c[j,k] w_c[k,i] = 1 for all 8 triples.
    
    # One valid set of 7 terms (verified):
    # c0: u=[[1,0],[0,0]], v=[[1,0],[0,0]], w=[[1,0],[0,0]] -> (0,0,0)=1, others 0
    # This is just schoolbook. For Strassen, we need overlapping support.
    
    # Let me just use the fact that the known rank for 2x2x2x2 is 7 and
    # construct it via the standard Strassen formulas, mapping each product
    # to a rank-1 term appropriately.
    
    # For product M1 = (X11+X22)(Y11+Y22):
    # This is a scalar product. In the tensor decomposition, this corresponds to
    # a term where the U part is (X11+X22) = A[i0,j0]+A[i1,j1], 
    # the V part is (Y11+Y22) = B[j0,k0]+B[j1,k1],
    # and the W part distributes the result to Z11 and Z22.
    # But W can only have one (k,i) pair per term...
    
    # I think the correct interpretation is:
    # Each Strassen product Mi = Li(X) * Mi(Y) where Li and Mi are linear forms.
    # In the tensor decomposition, this becomes a term where:
    # u[i,j] = coefficient of A[i,j] in Li
    # v[j,k] = coefficient of B[j,k] in Mi  
    # w[k,i] = determined by how Mi contributes to Z entries
    # But w[k,i] must be such that the product gives the right Z entries.
    
    # This is inherently a 3-way factorization, not a 2-way one.
    # The standard Strassen algorithm gives a bilinear decomposition, not trilinear.
    # For the tensor (trilinear) decomposition, we need to "split" each product.
    
    # Actually, I recall that the matrix multiplication tensor of format (n,m,p)
    # has rank equal to the number of scalar multiplications needed.
    # Strassen's algorithm for 2x2x2x2 uses 7 scalar multiplications, so the
    # rank of the 2x2x2x2 tensor is at most 7.
    
    # The 7 rank-1 terms can be constructed as follows. Each scalar multiplication
    # in Strassen corresponds to one rank-1 term in the tensor decomposition.
    # For Mi = Li(X) * Mi(Y), the rank-1 term has:
    # u_c[i,j] = coefficient of X[i,j] in Li
    # v_c[j,k] = coefficient of Y[j,k] in Mi
    # w_c[k,i] = 1 if Mi contributes to Z[k,i], with appropriate coefficient
    # But this doesn't quite work because one Mi contributes to multiple Z entries.
    
    # I think the correct way is that each of the 7 products corresponds to
    # a rank-1 term in a DIFFERENT sense. Let me just look at this differently.
    
    # For the full problem, let me just use schoolbook for the 2x2 blocks
    # and see if I can get rank 46 as mentioned in the knowledge.
    # The knowledge says "reducing total rank from 50 to 46 by replacing
    # 8-multiplication blocks with 7-multiplication Strassen blocks"
    # So 4 Strassen blocks: 4*(8-7) = 4 saved, giving 50-4=46.
    
    # Let me implement Strassen for 2x2x2x2 properly.
    # After research, the 7 rank-1 terms for 2x2x2x2 tensor are:
    
    # Using a known decomposition:
    # Let the 8 tensor entries be T[i,j,k] for i,j,k in {0,1}.
    # We need sum_c u_c[i,j] v_c[j,k] w_c[k,i] = 1 for all 8.
    
    # One valid decomposition (from algebraic complexity theory):
    # c=0: u00=1,u01=0,u10=0,u11=0; v00=1,v01=0,v10=0,v11=0; w00=1,w01=0,w10=0,w11=0
    # This is just one schoolbook term.
    
    # I'll take a different approach: implement Strassen by tracking the
    # intermediate linear forms and creating one rank-1 term per product,
    # where the W matrix encodes which output entries receive the contribution.
    
    # For M1 = (X11+X22)(Y11+Y22), which contributes to Z11 and Z22:
    # Z11 += M1, Z22 += M1
    # The rank-1 term: u[i,j] from (X11+X22), v[j,k] from (Y11+Y22)
    # u: (i0,j0)=1, (i1,j1)=1
    # v: (j0,k0)=1, (j1,k1)=1
    # For this to contribute to Z11=C[i0,k0] and Z22=C[i1,k1]:
    # w[k0,i0] and w[k1,i1] need to be set.
    # But the product u[i,j]*v[j,k]*w[k,i] gives:
    # (i0,j0,k0): 1*1*w[k0,i0]
    # (i0,j1,k1): 0*1*w[k1,i0] = 0
    # (i1,j0,k0): 0*1*w[k0,i1] = 0  
    # (i1,j1,k1): 1*1*w[k1,i1]
    # So we get contributions at (i0,j0,k0) and (i1,j1,k1) only.
    # But M1 should contribute to Z11=C[i0,k0] (which is (i0,*,k0) summed over j)
    # and Z22=C[i1,k1] (which is (i1,*,k1) summed over j).
    # The tensor entry T[i,j,k]=1 means C[i,k] = sum_j A[i,j]*B[j,k].
    # So the contribution to C[i0,k0] from this term is sum_j u[i0,j]*v[j,k0]*w[k0,i0]
    # = u[i0,j0]*v[j0,k0]*w[k0,i0] + u[i0,j1]*v[j1,k0]*w[k0,i0]
    # = 1*1*w[k0,i0] + 0*0*w[k0,i0] = w[k0,i0]
    # Similarly for C[i1,k1]: = u[i1,j1]*v[j1,k1]*w[k1,i1] = 1*1*w[k1,i1] = w[k1,i1]
    # We need both to equal 1 (since M1 contributes 1 to both Z11 and Z22).
    # So w[k0,i0]=1, w[k1,i1]=1.
    # But we also need to check no spurious contributions:
    # C[i0,k1] = sum_j u[i0,j]*v[j,k1]*w[k1,i0] = u[i0,j0]*v[j0,k1]*w[k1,i0] + u[i0,j1]*v[j1,k1]*w[k1,i0]
    # = 1*0*w[k1,i0] + 0*1*w[k1,i0] = 0. Good.
    # C[i1,k0] = sum_j u[i1,j]*v[j,k0]*w[k0,i1] = u[i1,j0]*v[j0,k0]*w[k0,i1] + u[i1,j1]*v[j1,k0]*w[k0,i1]
    # = 0*1*w[k0,i1] + 1*0*w[k0,i1] = 0. Good.
    
    # So for M1: u=(i0,j0)=1,(i1,j1)=1; v=(j0,k0)=1,(j1,k1)=1; w=(k0,i0)=1,(k1,i1)=1.
    # This works! Let me verify all 8 tensor entries for this single term:
    # T[i,j,k] = u[i,j]*v[j,k]*w[k,i]
    # (0,0,0): 1*1*1=1
    # (0,0,1): 1*0*0=0
    # (0,1,0): 0*0*1=0
    # (0,1,1): 0*1*0=0
    # (1,0,0): 0*1*0=0
    # (1,0,1): 0*0*1=0
    # (1,1,0): 1*0*0=0
    # (1,1,1): 1*1*1=1
    # So this term gives T[0,0,0]=1, T[1,1,1]=1, others 0.
    # But we need ALL T[i,j,k]=1. So we need 7 such terms that together give all 1s.
    
    # This means each of the 7 Strassen terms gives a pattern of 1s, and the
    # sum over all 7 terms gives 1 at every position.
    
    # Let me construct all 7 terms:
    
    # M1 = (X11+X22)(Y11+Y22): contributes to Z11, Z22
    # u: (i0,j0)=1, (i1,j1)=1; v: (j0,k0)=1, (j1,k1)=1; w: (k0,i0)=1, (k1,i1)=1
    # Pattern: (i0,j0,k0)=1, (i1,j1,k1)=1
    
    # M2 = (X21+X22)Y11: contributes to Z21
    # u: (i1,j0)=1, (i1,j1)=1; v: (j0,k0)=1; w: (k0,i1)=1
    # Pattern: (i1,j0,k0)=1, (i1,j1,k0)=1
    
    # M3 = X11(Y12-Y22): contributes to Z12
    # u: (i0,j0)=1; v: (j0,k1)=1, (j1,k1)=-1; w: (k1,i0)=1
    # Pattern: (i0,j0,k1)=1, (i0,j1,k1)=-1
    
    # M4 = X22(Y21-Y11): contributes to Z11, Z21
    # u: (i1,j1)=1; v: (j1,k0)=1, (j0,k0)=-1; w: (k0,i0)=1, (k0,i1)=1
    # Wait, M4 contributes to both Z11 and Z21? Let me recheck.
    # Z11 = M1+M4-M5+M7, Z21 = M2+M4
    # So M4 contributes to Z11 and Z21.
    # u: (i1,j1)=1; v: (j1,k0)=1, (j0,k0)=-1; w: (k0,i0)=1, (k0,i1)=1
    # Pattern: (i1,j1,k0)*w[k0,i1]=1*1*1=1 at (i1,j1,k0) for Z21 contribution
    #          (i1,j1,k0)*w[k0,i0]=1*1*1=1 at (i1,j1,k0) for Z11 contribution
    # But these are the same (i,j,k) triple! The tensor entry is T[i1,j1,k0].
    # The contribution to C[i0,k0] (Z11) from this term: sum_j u[i0,j]*v[j,k0]*w[k0,i0] = 0 (u[i0,*]=0)
    # The contribution to C[i1,k0] (Z21) from this term: sum_j u[i1,j]*v[j,k0]*w[k0,i1] = u[i1,j1]*v[j1,k0]*w[k0,i1] = 1*1*1=1
    # But M4 should contribute to Z11=C[i0,k0] too! 
    # C[i0,k0] contribution: sum_j u[i0,j]*v[j,k0]*w[k0,i0] = 0 since u[i0,j]=0 for all j.
    # That's wrong. M4 = X22*(Y21-Y11) should contribute to Z11.
    # Z11 = C[i0,k0] = sum_j A[i0,j]*B[j,k0]. M4 involves X22=A[i1,j1], so it doesn't
    # directly contribute to C[i0,k0]. 
    # I think I'm confusing myself. In Strassen, the intermediate results are combined
    # to form the final Z. The "contribution to Z11" means that after all 7 products,
    # Z11 = M1+M4-M5+M7. Each Mi is a scalar, and they are added to form Z11.
    # In the tensor decomposition, each Mi corresponds to a rank-1 term, and the
    # W matrix determines how the scalar Mi is distributed to the Z entries.
    
    # For Mi, the W matrix should have w[k,i] = 1 for each (k,i) where Mi contributes
    # to Z[i,k] (with the appropriate sign). But w[k,i] is a single value, and if Mi
    # contributes to multiple Z entries, w needs to encode that.
    
    # The key insight: for a rank-1 term, the contribution to C[i,k] is
    # sum_j u[i,j]*v[j,k]*w[k,i]. If w[k,i] is the same for all j, then the
    # contribution is w[k,i] * (sum_j u[i,j]*v[j,k]).
    # So for Mi to contribute to Z[i0,k0] with coefficient 1, we need
    # w[k0,i0] * (sum_j u[i0,j]*v[j,k0]) = 1.
    
    # For M4 = X22*(Y21-Y11):
    # u[i1,j1]=1, all other u=0
    # v[j1,k0]=1, v[j0,k0]=-1, all other v=0
    # sum_j u[i0,j]*v[j,k0] = 0 (since u[i0,j]=0)
    # sum_j u[i1,j]*v[j,k0] = u[i1,j1]*v[j1,k0] = 1*1 = 1
    # So C[i1,k0] gets w[k0,i1]*1. We want this to be 1 (M4 contributes to Z21=C[i1,k0]).
    # And C[i0,k0] gets w[k0,i0]*0 = 0. But M4 should contribute to Z11=C[i0,k0]!
    # Contradiction. 
    
    # I think the issue is that in Strassen, M4 does NOT directly contribute to Z11
    # in the way I'm thinking. Let me re-examine:
    # Z11 = M1+M4-M5+M7
    # Z12 = M3+M5
    # Z21 = M2+M4
    # Z22 = M1-M2+M3+M6
    # 
    # M4 appears in Z11 and Z21. But M4 = X22*(Y21-Y11).
    # X22 = A[i1,j1], Y21-Y11 = B[j1,k0]-B[j0,k0].
    # The product M4 is a scalar. In the tensor decomposition, this scalar is
    # "placed" into Z11 and Z21 via the W matrix.
    # 
    # For the rank-1 term corresponding to M4:
    # u[i,j] = coefficient in X22 = (i1,j1) has coeff 1
    # v[j,k] = coefficients in (Y21-Y11) = (j1,k0) has coeff 1, (j0,k0) has coeff -1
    # w[k,i] = ? 
    # The contribution to C[i,k] is sum_j u[i,j]*v[j,k]*w[k,i].
    # For i=i1: sum_j u[i1,j]*v[j,k]*w[k,i1] = u[i1,j1]*v[j1,k]*w[k,i1] = v[j1,k]*w[k,i1]
    # For k=k0: = v[j1,k0]*w[k0,i1] = 1*w[k0,i1]
    # For k!=k0: = 0
    # For i!=i1: = 0
    # So this term only contributes to C[i1,k0] with value w[k0,i1].
    # We need M4 to contribute 1 to both Z11=C[i0,k0] and Z21=C[i1,k0].
    # But this term can only contribute to C[i1,k0]. So we need a SEPARATE mechanism
    # for the Z11 contribution.
    
    # I think the resolution is that each Strassen "product" actually corresponds to
    # MULTIPLE rank-1 terms in the tensor decomposition, or the decomposition is
    # not one-to-one with Strassen products.
    
    # Given the complexity, let me just use schoolbook for all blocks and accept
    # rank 50, OR use a simpler block decomposition that I can verify.
    
    # Actually, let me try a different partition: split into 2+3 or other sizes.
    # Or just return schoolbook since the current program already does that.
    
    # For now, let me just return schoolbook to ensure correctness.
    return col_start

def _strassen_222_add(U, V, W, i_off, j_off, k_off, n, m, p, col_start):
    """Add Strassen 2x2x2x2 decomposition. Returns new col_start."""
    i0, i1 = i_off, i_off + 1
    j0, j1 = j_off, j_off + 1
    k0, k1 = k_off, k_off + 1
    
    # After careful analysis, the 7 Strassen products map to 7 rank-1 terms
    # as follows. Each term c has u_c[i,j], v_c[j,k], w_c[k,i].
    # The sum over all 7 terms must give T[i,j,k]=1 for all i,j,k in the 2x2x2 block.
    
    # Term 0 (M1): u=(i0,j0)=1,(i1,j1)=1; v=(j0,k0)=1,(j1,k1)=1; w=(k0,i0)=1,(k1,i1)=1
    # Gives: (i0,j0,k0)=1, (i1,j1,k1)=1
    
    # Term 1 (M2): u=(i1,j0)=1,(i1,j1)=1; v=(j0,k0)=1; w=(k0,i1)=1
    # Gives: (i1,j0,k0)=1, (i1,j1,k0)=1
    
    # Term 2 (M3): u=(i0,j0)=1; v=(j0,k1)=1,(j1,k1)=-1; w=(k1,i0)=1
    # Gives: (i0,j0,k1)=1, (i0,j1,k1)=-1
    
    # Term 3 (M4): u=(i1,j1)=1; v=(j1,k0)=1,(j0,k0)=-1; w=(k0,i1)=1
    # Gives: (i1,j1,k0)=1, (i1,j0,k0)=-1
    
    # Term 4 (M5): u=(i0,j0)=1,(i0,j1)=1; v=(j1,k1)=1; w=(k1,i0)=1
    # Gives: (i0,j1,k1)=1
    
    # Term 5 (M6): u=(i1,j0)=1,(i0,j0)=-1; v=(j0,k0)=1,(j0,k1)=1; w=(k0,i1)=1,(k1,i1)=1
    # Gives: (i1,j0,k0)=1, (i1,j0,k1)=1, (i0,j0,k0)=-1, (i0,j0,k1)=-1
    
    # Term 6 (M7): u=(i0,j1)=1,(i1,j1)=-1; v=(j1,k0)=1,(j1,k1)=1; w=(k0,i0)=1,(k1,i0)=1
    # Gives: (i0,j1,k0)=1, (i0,j1,k1)=1, (i1,j1,k0)=-1, (i1,j1,k1)=-1
    
    # Let me sum up all contributions to each (i,j,k):
    # (i0,j0,k0): T0=1, T5=-1 -> 0. Need 1. WRONG.
    
    # My assignment is wrong. Let me try a different approach and just
    # verify by computation.
    
    # For now, let me just add schoolbook terms for this 2x2x2 block (8 terms)
    # to ensure correctness, and skip Strassen for now.
    col = col_start
    for i in [i0, i1]:
        for j in [j0, j1]:
            for k in [k0, k1]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return col

def construct():
    start = time.time()
    return _schoolbook()
# EVOLVE-BLOCK-END
