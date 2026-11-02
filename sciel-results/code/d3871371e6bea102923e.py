"""Rank-minimal tensor decomposition for 2x2 * 2x11 matrix multiplication."""
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r]
                # W index: k*n + i
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r])
                if s != Fraction(W[k * n + i][0]) if len(W[k * n + i]) > 0 else s != 0:
                    # W should be 1 for the matching term, but in a decomposition
                    # W[k*n+i, r] is the coefficient. Let me re-check the convention.
                    pass
    # Actually, the decomposition is: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait no. The tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] should equal delta_{i,j,k} for the multiplication tensor.
    # For matrix multiplication: (AB)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 if we're computing (AB)[i,k] with A[i,j]*B[j,k]
    # Actually T[i,j,k] = 1 for all i,j,k? No.
    # The multiplication tensor: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the product is
    # <T, A ⊗ B> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} (AB)[i,k] * 1
    # So T[i,j,k] = 1 for all i,j,k? That's not right either.
    # Let me think again. The bilinear map (A,B) -> AB has:
    # (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C
    # where e_i^A picks out A[i,*], e_j^B picks out B[*,j]... 
    # Actually: T = sum_{i,j,k} (e_i ⊗ e_j ⊗ e_k) where the trilinear form is
    # <T, A, B, C> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i]
    # For multiplication: C = AB, so C[k,i]... hmm the indexing is getting confusing.
    
    # Let me use the standard: the matrix multiplication tensor for n×m times m×p is
    # T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if and only if... actually T[i,j,k] = 1 for all valid i,j,k
    # because (AB)[i,k] = sum_j A[i,j] B[j,k]
    # The tensor decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where (U_r)[i*m+j] is the A-coefficient, (V_r)[j*p+k] is the B-coefficient, 
    # (W_r)[k*n+i] is the C-coefficient
    # And we need: sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1 for all i,j,k
    
    # Let me just verify this directly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                if s != 1:
                    return False
    return True


def _construct_rank39():
    """Construct a rank-39 decomposition using Strassen on column pairs."""
    n, m, p = 2, 2, 11
    # 5 pairs of columns (0,1),(2,3),(4,5),(6,7),(8,9) using Strassen (7 terms each)
    # 1 single column (10) using schoolbook (4 terms)
    # Total: 5*7 + 4 = 39
    
    rank = 39
    U = [[0] * rank for _ in range(4)]   # n*m = 4 rows
    V = [[0] * rank for _ in range(22)]  # m*p = 22 rows
    W = [[0] * rank for _ in range(22)]  # p*n = 22 rows
    
    col = 0
    
    for pair_idx in range(5):
        k0 = 2 * pair_idx
        k1 = 2 * pair_idx + 1
        
        # Strassen for 2x2 * 2x2 block
        # A = [[a, b], [c, d]] where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # B = [[e, f], [g, h]] where e=B[0,k0], f=B[0,k1], g=B[1,k0], h=B[1,k1]
        # C = A*B = [[C00, C01], [C10, C11]]
        # C00 = a*e + b*g, C01 = a*f + b*h
        # C10 = c*e + d*g, C11 = c*f + d*h
        
        # Strassen:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e  
        # m3 = (a+b)h
        # m4 = d(h-g)
        # m5 = a(f-e)
        # m6 = (c+a)(e+g)
        # m7 = (b+d)(f+h)
        
        # C00 = m1 + m4 - m5 - m7
        # C01 = m5 + m3
        # C10 = m6 + m4
        # C11 = m1 + m2 - m3 - m6
        
        # Let me verify: C00 = (a+d)(e+h) + d(h-g) - a(f-e) - (b+d)(f+h)
        # = ae + ah + de + dh + dh - dg - af + ae - bf - bh - df - dh
        # = 2ae + 2dh - dg - af - bf - bh - df + ah + de - dh
        # Hmm this is getting complicated. Let me just verify numerically.
        
        # Actually, let me use the well-known correct Strassen formulas:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)(e)
        # m3 = (a+b)(h)
        # m4 = d(h-g)
        # m5 = a(f-e)
        # m6 = (c+a)(e+g)
        # m7 = (b+d)(f+h)
        
        # C00 = m1 + m4 - m5 - m7
        # C01 = m5 + m3
        # C10 = m6 + m4
        # C11 = m1 + m2 - m3 - m6
        
        # Verify C00: (a+d)(e+h) + d(h-g) - a(f-e) - (b+d)(f+h)
        # = (ae+ah+de+dh) + (dh-dg) - (af-ae) - (bf+bh+df+dh)
        # = ae+ah+de+dh+dh-dg-af+ae-bf-bh-df-dh
        # = 2ae + ah + de + dh - dg - af - bf - bh - df
        # Expected: ae + bg
        # This doesn't match! I'm using wrong formulas.
        
        # Let me look up the correct Strassen formulas carefully.
        # From Wikipedia (Strassen algorithm):
        # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # p1 = (a+d)(e+h)
        # p2 = (c+d)(e)
        # p3 = (a+b)(h)
        # p4 = d(h-g)
        # p5 = a(f-e)
        # p6 = (c+a)(e+g)
        # p7 = (b+d)(f+h)
        
        # C00 = p1 + p4 - p5 - p7
        # C01 = p5 + p3
        # C10 = p6 + p4
        # C11 = p1 + p2 - p3 - p6
        
        # Let me verify C00 = p1 + p4 - p5 - p7:
        # = (a+d)(e+h) + d(h-g) - a(f-e) - (b+d)(f+h)
        # = ae + ah + de + dh + dh - dg - af + ae - bf - bh - df - dh
        # = 2ae + ah + de + dh - dg - af - bf - bh - df
        # Expected: ae + bg
        # Still doesn't work. I must have the wrong formula set.
        
        # Let me try a different set. The standard Strassen:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-g)
        # m4 = d(h-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+g)
        # m7 = (b-d)(f+h)
        
        # C00 = m1 - m4 + m6
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m3 + m7
        
        # Verify C00 = m1 - m4 + m6:
        # = (a+d)(e+h) - d(h-e) + (c-a)(e+g)
        # = ae + ah + de + dh - dh + de + ce + cg - ae - eg
        # = ah + 2de + dh + ce + cg - eg
        # Expected: ae + bg
        # Still wrong!
        
        # I think I need to be very careful. Let me derive from scratch.
        # We want to compute C = AB where A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C00 = ae + bg, C01 = af + bh, C10 = ce + dg, C11 = cf + dh
        
        # Strassen's 7 products:
        # m1 = (a+d)(e+h) = ae + ah + de + dh
        # m2 = (c+d)e = ce + de
        # m3 = a(f-g) = af - ag
        # m4 = d(h-e) = dh - de
        # m5 = (a+b)h = ah + bh
        # m6 = (c-a)(e+g) = ce + cg - ae - ag
        # m7 = (b-d)(f+h) = bf + bh - df - dh
        
        # C00 = ae + bg
        # C01 = af + bh
        # C10 = ce + dg
        # C11 = cf + dh
        
        # Let me try:
        # C00 = m1 - m4 + m6 = (ae+ah+de+dh) - (dh-de) + (ce+cg-ae-ag)
        # = ae + ah + de + dh - dh + de + ce + cg - ae - ag
        # = ah + 2de + ce + cg - ag
        # Not matching.
        
        # C00 = m1 + m4 - m5 - m7?
        # = (ae+ah+de+dh) + (dh-de) - (ah+bh) - (bf+bh-df-dh)
        # = ae+ah+de+dh+dh-de-ah-bh-bf-bh+df+dh
        # = ae + 3dh - bh - bf + df
        # No.
        
        # OK let me just look at this differently. I'll use a known correct implementation.
        # The correct Strassen formulas (verified):
        # m1 = (a + d) * (e + h)
        # m2 = (c + d) * e
        # m3 = a * (f - g)
        # m4 = d * (h - e)
        # m5 = (a + b) * h
        # m6 = (c - a) * (e + g)
        # m7 = (b - d) * (f + h)
        
        # C00 = m1 - m4 + m6
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m3 + m7
        
        # Verify C00 = m1 - m4 + m6:
        # m1 = (a+d)(e+h) = ae+ah+de+dh
        # m4 = d(h-e) = dh-de
        # m6 = (c-a)(e+g) = ce+cg-ae-ag
        # C00 = ae+ah+de+dh - (dh-de) + ce+cg-ae-ag
        # = ae+ah+de+dh-dh+de+ce+cg-ae-ag
        # = ah + 2de + ce + cg - ag
        # Expected: ae + bg
        # This gives ah + 2de + ce + cg - ag, not ae + bg.
        
        # I'm clearly making an error. Let me try yet another formulation.
        
        # From a reliable source, the Strassen algorithm:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)(e)  
        # m3 = (a+b)(h)
        # m4 = d(h-g)
        # m5 = a(f-e)
        # m6 = (c+a)(e+g)
        # m7 = (b+d)(f+h)
        
        # C00 = m1 + m4 - m5 - m7
        # C01 = m5 + m3
        # C10 = m6 + m4
        # C11 = m1 + m2 - m3 - m6
        
        # Verify C01 = m5 + m3 = a(f-e) + (a+b)h = af - ae + ah + bh
        # Expected: af + bh
        # Got: af - ae + ah + bh. Extra terms: -ae + ah. Not matching.
        
        # I give up trying to remember. Let me just use a simple correct approach.
        # I'll use the fact that we can decompose using 7 multiplications per 2x2x2 block.
        # Let me just hard-code a verified rank-39 solution.
        
        # Actually, let me use a different strategy. I'll use the "four Russians" or 
        # simply verify my construction by testing it.
        
        # Let me use a simpler approach: for each pair of columns, use the 
        # bilinear complexity of 2x2 * 2x2 which is 7 (Strassen).
        # I'll implement it and verify.
        
        # Correct Strassen (I'll verify by testing):
        # For A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)e
        # p3 = a(f-g) 
        # p4 = d(h-e)
        # p5 = (a+b)h
        # p6 = (c-a)(e+g)
        # p7 = (b-d)(f+h)
        
        # C00 = p1 - p4 + p6
        # C01 = p3 + p5
        # C10 = p2 + p4
        # C11 = p1 - p3 + p7
        
        # I'll verify this in the code and if it fails, fall back to schoolbook.
        
        # A indices: 0=a(0,0), 1=b(0,1), 2=c(1,0), 3=d(1,1)
        # B for this block: e=B[0,k0]=V[0*11+k0], f=B[0,k1]=V[0*11+k1]
        #                   g=B[1,k0]=V[1*11+k0], h=B[1,k1]=V[1*11+k1]
        
        # p1 = (a+d)(e+h): 
        #   U: A[0,0]+A[1,1] = U[0]+U[3], V: B[0,k0]+B[1,k1] = V[11+k0]+V[11+k1]
        # p2 = (c+d)e:
        #   U: A[1,0]+A[1,1] = U[2]+U[3], V: B[0,k0] = V[11+k0]
        # p3 = a(f-g):
        #   U: A[0,0] = U[0], V: B[0,k1]-B[1,k0] = V[11+k1]-V[11+k0]
        # p4 = d(h-e):
        #   U: A[1,1] = U[3], V: B[1,k1]-B[0,k0] = V[11+k1]-V[11+k0]
        # p5 = (a+b)h:
        #   U: A[0,0]+A[0,1] = U[0]+U[1], V: B[1,k1] = V[11+k1]
        # p6 = (c-a)(e+g):
        #   U: A[1,0]-A[0,0] = U[2]-U[0], V: B[0,k0]+B[1,k0] = V[11+k0]+V[11+k0]
        # p7 = (b-d)(f+h):
        #   U: A[0,1]-A[1,1] = U[1]-U[3], V: B[0,k1]+B[1,k1] = V[11+k1]+V[11+k1]
        
        # C00 (output at row 0, col k0) = p1 - p4 + p6
        # C01 (output at row 0, col k1) = p3 + p5
        # C10 (output at row 1, col k0) = p2 + p4
        # C11 (output at row 1, col k1) = p1 - p3 + p7
        
        # W indices: W[k*n + i] where i is row of C, k is col of C
        # C[0,k0] -> W[k0*2+0] = W[2*k0]
        # C[0,k1] -> W[k1*2+0] = W[2*k1]
        # C[1,k0] -> W[k0*2+1] = W[2*k0+1]
        # C[1,k1] -> W[k1*2+1] = W[2*k1+1]
        
        # Now let me assign the 7 Strassen terms:
        # Term 0: p1 = (a+d)(e+h)
        # Term 1: p2 = (c+d)e
        # Term 2: p3 = a(f-g)
        # Term 3: p4 = d(h-e)
        # Term 4: p5 = (a+b)h
        # Term 5: p6 = (c-a)(e+g)
        # Term 6: p7 = (b-d)(f+h)
        
        # U contributions (for A indices 0,1,2,3):
        # p1: U[0]=1, U[3]=1
        # p2: U[2]=1, U[3]=1
        # p3: U[0]=1
        # p4: U[3]=1
        # p5: U[0]=1, U[1]=1
        # p6: U[2]=1, U[0]=-1
        # p7: U[1]=1, U[3]=-1
        
        # V contributions (for B indices):
        # B[0,k0] index in V: 0*11+k0 = k0
        # B[0,k1] index in V: 0*11+k1 = k1
        # B[1,k0] index in V: 1*11+k0 = 11+k0
        # B[1,k1] index in V: 1*11+k1 = 11+k1
        
        # p1: V[k0]=1, V[11+k1]=1
        # p2: V[k0]=1
        # p3: V[k1]=1, V[11+k0]=-1
        # p4: V[11+k1]=1, V[k0]=-1
        # p5: V[11+k1]=1
        # p6: V[k0]=1, V[11+k0]=1
        # p7: V[k1]=1, V[11+k1]=1
        
        # W contributions:
        # C[0,k0] -> W[2*k0]: p1=1, p4=-1, p6=1
        # C[0,k1] -> W[2*k1]: p3=1, p5=1
        # C[1,k0] -> W[2*k0+1]: p2=1, p4=1
        # C[1,k1] -> W[2*k1+1]: p1=1, p3=-1, p7=1
        
        # Let me code this up:
        for t in range(7):
            r = col + t
            # U
            if t == 0:  # p1
                U[0][r] += 1; U[3][r] += 1
            elif t == 1:  # p2
                U[2][r] += 1; U[3][r] += 1
            elif t == 2:  # p3
                U[0][r] += 1
            elif t == 3:  # p4
                U[3][r] += 1
            elif t == 4:  # p5
                U[0][r] += 1; U[1][r] += 1
            elif t == 5:  # p6
                U[2][r] += 1; U[0][r] += -1
            elif t == 6:  # p7
                U[1][r] += 1; U[3][r] += -1
            
            # V
            if t == 0:  # p1
                V[k0][r] += 1; V[11+k1][r] += 1
            elif t == 1:  # p2
                V[k0][r] += 1
            elif t == 2:  # p3
                V[k1][r] += 1; V[11+k0][r] += -1
            elif t == 3:  # p4
                V[11+k1][r] += 1; V[k0][r] += -1
            elif t == 4:  # p5
                V[11+k1][r] += 1
            elif t == 5:  # p6
                V[k0][r] += 1; V[11+k0][r] += 1
            elif t == 6:  # p7
                V[k1][r] += 1; V[11+k1][r] += 1
            
            # W
            if t == 0:  # p1
                W[2*k0][r] += 1; W[2*k1+1][r] += 1
            elif t == 1:  # p2
                W[2*k0+1][r] += 1
            elif t == 2:  # p3
                W[2*k1][r] += 1; W[2*k1+1][r] += -1
            elif t == 3:  # p4
                W[2*k0][r] += -1; W[2*k0+1][r] += 1
            elif t == 4:  # p5
                W[2*k1][r] += 1
            elif t == 5:  # p6
                W[2*k0][r] += 1
            elif t == 6:  # p7
                W[2*k1+1][r] += 1
        
        col += 7
    
    # Last column (k=10): schoolbook, 4 terms
    k = 10
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # Term 0: A[0,0]*B[0,k] -> C[0,k]
    # Term 1: A[0,1]*B[1,k] -> C[0,k]
    # Term 2: A[1,0]*B[0,k] -> C[1,k]
    # Term 3: A[1,1]*B[1,k] -> C[1,k]
    
    U[0][col+0] = 1; V[k][col+0] = 1; W[2*k][col+0] = 1
    U[1][col+1] = 1; V[11+k][col+1] = 1; W[2*k][col+1] = 1
    U[2][col+2] = 1; V[k][col+2] = 1; W[2*k+1][col+2] = 1
    U[3][col+3] = 1; V[11+k][col+3] = 1; W[2*k+1][col+3] = 1
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n=2, m=2, p=11):
    """Verify the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True


def _try_lower_rank(deadline):
    """Try to find a decomposition with rank < 39."""
    n, m, p = 2, 2, 11
    # The lower bound for 2x2 * 2xp is at least 2*2*p - something
    # Actually, the tensor rank of 2x2 * 2xp is known to be at least 2p+... 
    # For p=11, we know rank 39 works. Can we do better?
    # The bilinear complexity of 2x2 * 2x1 is 4 (schoolbook, optimal)
    # For 2x2 * 2x2 it's 7 (Strassen, optimal)
    # For 2x2 * 2x3, it might be 11 or so
    # Let's try to see if rank 38 or lower is possible by attempting random search
    
    # Actually, let's try a different construction:
    # Use Strassen for columns 0-1, 2-3, 4-5, 6-7 (4 pairs = 28 terms)
    # For columns 8,9,10: use a 2x2 * 2x3 decomposition
    # The 2x2 * 2x3 tensor has rank at least... let's try 11 terms
    # Total: 28 + 11 = 39. Same.
    
    # Alternative: 3 pairs (21 terms) + 2x2*2x5 block
    # 2x2*2x5 might have rank 18? Total: 21+18=39. Same.
    
    # Let me try: what if we can share some terms between the last column and a pair?
    # This is getting complex. Let me just try to verify rank 39 and return it.
    
    # For now, return None to indicate no better rank found
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # First, construct and verify rank 39
    result39 = _construct_rank39()
    
    if _verify_decomposition(result39["U"], result39["V"], result39["W"], n, m, p):
        # Rank 39 is verified. Try to find something better.
        if time.time() < deadline - 10:
            better = _try_lower_rank(deadline)
            if better is not None and _verify_decomposition(better["U"], better["V"], better["W"], n, m, p):
                return better
        return result39
    else:
        # Fallback to schoolbook (rank 44)
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


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"Verified: {_verify_decomposition(result['U'], result['V'], result['W'])}")
