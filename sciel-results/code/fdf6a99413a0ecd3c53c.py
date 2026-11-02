import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (16, rank)
    
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a rank-1 term
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_idx: index in W (k*n + i)
    # coeff: value
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx, col] += coeff
        V[v_idx, col] += coeff
        W[w_idx, col] += coeff
        col += 1

    # Strassen 2x2x2 decomposition
    # A indices: (i0, j0), (i0, j1), (i1, j0), (i1, j1)
    # B indices: (j0, k0), (j0, k1), (j1, k0), (j1, k1)
    # C indices: (i0, k0), (i0, k1), (i1, k0), (i1, k1)
    # Local indices 0,1 for i, j, k within the 2x2 block
    
    def strassen_2x2x2(i0, j0, k0):
        # i0: row offset in A (0 or 2)
        # j0: col offset in A / row offset in B (0 or 2)
        # k0: col offset in B (0 or 2)
        
        # M1 = (A00 + A11)(B00 + B11)
        # A00: i0, j0; A11: i0+1, j0+1
        # B00: j0, k0; B11: j0+1, k0+1
        # C00: i0, k0; C11: i0+1, k0+1; C01: i0, k0+1; C10: i0+1, k0
        
        # Term 1: A00*B00
        add_term(i0*m + j0, j0*p + k0, k0*n + i0, 1)
        # Term 2: A00*B11
        add_term(i0*m + j0, (j0+1)*p + (k0+1), (k0+1)*n + i0, 1)
        # Term 3: A11*B00
        add_term((i0+1)*m + (j0+1), j0*p + k0, k0*n + (i0+1), 1)
        # Term 4: A11*B11
        add_term((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+1), 1)
        
        # We need to form the linear combinations.
        # It is easier to just define the 7 products and then map to C.
        # But the interface requires U, V, W matrices where each column is a rank-1 term.
        # The Strassen algorithm computes 7 intermediate products M1..M7.
        # Each M is a rank-1 term in the decomposition of the product tensor?
        # No, M1 = (A00+A11)(B00+B11) is a product of two linear forms.
        # So M1 corresponds to ONE rank-1 term in the tensor decomposition?
        # Yes, $u \otimes v \otimes w$ where $u$ is the linear form for A, $v$ for B.
        # But $C$ is a linear combination of $M$s.
        # So for each Strassen term, we have one column in U, V, W.
        # The entries in U are the coefficients of A in the linear form.
        # The entries in V are the coefficients of B in the linear form.
        # The entries in W are the coefficients of M in the linear form for C.
        
        # Let's restart the logic for Strassen.
        # We will add 7 columns for each 2x2x2 block.
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00 += 1, C11 += 1, C01 += 1, C10 += 1 ?
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        
        # So for M1:
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1, C01=0, C10=0 (Wait, C01 and C10 don't use M1? 
        #      C00 uses M1. C11 uses M1. C01 doesn't. C10 doesn't.)
        
        # Let's check the formulas again.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1:
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        
        # M2 = (A10 + A11)B00
        # U: A10=1, A11=1
        # V: B00=1
        # W: C10=1, C11=-1
        
        # M3 = A00(B01 - B11)
        # U: A00=1
        # V: B01=1, B11=-1
        # W: C01=1, C11=1
        
        # M4 = A11(B10 - B00)
        # U: A11=1
        # V: B10=1, B00=-1
        # W: C00=1, C10=1
        
        # M5 = (A00 + A01)B11
        # U: A00=1, A01=1
        # V: B11=1
        # W: C00=-1, C01=1
        
        # M6 = (A10 - A00)(B00 + B01)
        # U: A10=1, A00=-1
        # V: B00=1, B01=1
        # W: C11=1
        
        # M7 = (A01 - A11)(B10 + B11)
        # U: A01=1, A11=-1
        # V: B10=1, B11=1
        # W: C00=1, C10=1  -- Wait, C00 has M7? Yes. C10 has M7? No.
        # Let's re-verify C10. C10 = M2 + M4. No M7.
        # Let's re-verify C00. C00 = M1 + M4 - M5 + M7. Yes.
        
        # So:
        # M7:
        # U: A01=1, A11=-1
        # V: B10=1, B11=1
        # W: C00=1
        
        # Wait, I missed some W contributions?
        # C00: M1, M4, -M5, M7
        # C01: M3, M5
        # C10: M2, M4
        # C11: M1, -M2, M3, M6
        
        # Let's map carefully.
        
        # M1:
        # U: (i0,j0)=1, (i0+1,j0+1)=1
        # V: (j0,k0)=1, (j0+1,k0+1)=1
        # W: (k0,i0)=1, (k0+1,i0+1)=1
        
        # M2:
        # U: (i0+1,j0)=1, (i0+1,j0+1)=1
        # V: (j0,k0)=1
        # W: (k0,i0+1)=1, (k0+1,i0+1)=-1
        
        # M3:
        # U: (i0,j0)=1
        # V: (j0,k0+1)=1, (j0+1,k0+1)=-1
        # W: (k0+1,i0)=1, (k0+1,i0+1)=1
        
        # M4:
        # U: (i0+1,j0+1)=1
        # V: (j0+1,k0)=1, (j0,k0)=-1
        # W: (k0,i0)=1, (k0,i0+1)=1
        
        # M5:
        # U: (i0,j0)=1, (i0,j0+1)=1
        # V: (j0+1,k0+1)=1
        # W: (k0,i0)=-1, (k0+1,i0)=1
        
        # M6:
        # U: (i0+1,j0)=1, (i0,j0)=-1
        # V: (j0,k0)=1, (j0,k0+1)=1
        # W: (k0+1,i0+1)=1
        
        # M7:
        # U: (i0,j0+1)=1, (i0+1,j0+1)=-1
        # V: (j0+1,k0)=1, (j0+1,k0+1)=1
        # W: (k0,i0)=1
        
        # Check C10 again. C10 = M2 + M4.
        # M2 W: (k0, i0+1) = 1.
        # M4 W: (k0, i0+1) = 1.
        # Sum = 2? No, M2 has C10=1, M4 has C10=1.
        # But standard Strassen C10 = M2 + M4.
        # So W for M2 at (k0, i0+1) is 1.
        # W for M4 at (k0, i0+1) is 1.
        # This is correct.
        
        # Check C00. M1(1) + M4(1) - M5(1) + M7(1).
        # M1 W: (k0, i0) = 1.
        # M4 W: (k0, i0) = 1.
        # M5 W: (k0, i0) = -1.
        # M7 W: (k0, i0) = 1.
        # Sum = 1+1-1+1 = 2?
        # Wait. C00 should be A00B00 + A01B10 + A10B01 + A11B11.
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # M4 = A11(B10-B00) = A11B10 - A11B00
        # M5 = (A00+A01)B11 = A00B11 + A01B11
        # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
        
        # C00 = M1 + M4 - M5 + M7
        # = (A00B00 + A00B11 + A11B00 + A11B11)
        #   + (A11B10 - A11B00)
        #   - (A00B11 + A01B11)
        #   + (A01B10 + A01B11 - A11B10 - A11B11)
        # = A00B00 + A11B10 + A01B10 - A01B11 - A11B11 + A01B11 - A11B11?
        # Let's sum terms:
        # A00B00: 1
        # A00B11: 1 - 1 = 0
        # A11B00: 1 - 1 = 0
        # A11B11: 1 - 1 = 0
        # A11B10: 1 - 1 = 0
        # A01B11: -1 + 1 = 0
        # A01B10: 1
        # Missing A10B01 and A10B10?
        # A10 terms are in M2, M6.
        # M2 = (A10+A11)B00 = A10B00 + A11B00
        # M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
        
        # I think I mixed up the Strassen formulas or their application.
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Note: Indices in formulas are 1-based. My code uses 0-based.
        # A11->A00, A22->A11, etc.
        
        # Let's re-map with 0-based indices.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # This matches my previous mapping.
        
        # I will implement this mapping.
        
        # M1
        U[i0*m + j0, col] = 1; U[(i0+1)*m + (j0+1), col] = 1
        V[j0*p + k0, col] = 1; V[(j0+1)*p + (k0+1), col] = 1
        W[k0*n + i0, col] = 1; W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        # M2
        U[(i0+1)*m + j0, col] = 1; U[(i0+1)*m + (j0+1), col] = 1
        V[j0*p + k0, col] = 1
        W[k0*n + (i0+1), col] = 1; W[(k0+1)*n + (i0+1), col] = -1
        col += 1
        
        # M3
        U[i0*m + j0, col] = 1
        V[j0*p + (k0+1), col] = 1; V[(j0+1)*p + (k0+1), col] = -1
        W[(k0+1)*n + i0, col] = 1; W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        # M4
        U[(i0+1)*m + (j0+1), col] = 1
        V[(j0+1)*p + k0, col] = 1; V[j0*p + k0, col] = -1
        W[k0*n + i0, col] = 1; W[k0*n + (i0+1), col] = 1
        col += 1
        
        # M5
        U[i0*m + j0, col] = 1; U[i0*m + (j0+1), col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[k0*n + i0, col] = -1; W[(k0+1)*n + i0, col] = 1
        col += 1
        
        # M6
        U[(i0+1)*m + j0, col] = 1; U[i0*m + j0, col] = -1
        V[j0*p + k0, col] = 1; V[j0*p + (k0+1), col] = 1
        W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        # M7
        U[i0*m + (j0+1), col] = 1; U[(i0+1)*m + (j0+1), col] = -1
        V[(j0+1)*p + k0, col] = 1; V[(j0+1)*p + (k0+1), col] = 1
        W[k0*n + i0, col] = 1
        col += 1

    # Schoolbook 1x2x2
    def schoolbook_1x2x2(i0, j0, k0):
        # A: i0, j0..j0+1
        # B: j0..j0+1, k0..k0+1
        # C: i0, k0..k0+1
        
        # C0 = A0B0 + A1B1
        # C1 = A0B2 + A1B3 (using global indices)
        
        # Term 1: A0 * B0
        U[i0*m + j0, col] = 1
        V[j0*p + k0, col] = 1
        W[k0*n + i0, col] = 1
        col += 1
        
        # Term 2: A1 * B1
        U[i0*m + (j0+1), col] = 1
        V[(j0+1)*p + k0, col] = 1
        W[k0*n + i0, col] = 1
        col += 1
        
        # Term 3: A0 * B2
        U[i0*m + j0, col] = 1
        V[j0*p + (k0+1), col] = 1
        W[(k0+1)*n + i0, col] = 1
        col += 1
        
        # Term 4: A1 * B3
        U[i0*m + (j0+1), col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[(k0+1)*n + i0, col] = 1
        col += 1

    # Apply blocks
    # Top 2 rows (i=0,1)
    # j=0,1 k=0,1
    strassen_2x2x2(0, 0, 0)
    # j=2,3 k=0,1
    strassen_2x2x2(0, 2, 0)
    # j=0,1 k=2,3
    strassen_2x2x2(0, 0, 2)
    # j=2,3 k=2,3
    strassen_2x2x2(0, 2, 2)
    
    # Bottom row (i=2)
    # j=0,1 k=0,1
    schoolbook_1x2x2(2, 0, 0)
    # j=2,3 k=0,1
    schoolbook_1x2x2(2, 2, 0)
    # j=0,1 k=2,3
    schoolbook_1x2x2(2, 0, 2)
    # j=2,3 k=2,3
    schoolbook_1x2x2(2, 2, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
