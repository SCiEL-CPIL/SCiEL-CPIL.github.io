"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 8]
DEADLINE_SECONDS = 130.0

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # We want to decompose the tensor of rank n*m*p = 2*4*8 = 64.
    # Using 8 blocks of 2x2x2 matrices, each of rank 7 gives 8*7 = 56.
    # But 2x4x8 doesn't naturally decompose into 2x2x2 blocks without mixing.
    # The 2x4x8 tensor is a 2x4x8 3-tensor.
    # Actually, the tensor has dimensions (n, m, p) = (2, 4, 8).
    # We can think of it as a 2x(4x8) matrix or a (2x4)x8 matrix.
    # The standard way: the tensor T[i, j, k] = 1 if and only if ... 
    # Actually the matrix multiplication tensor is T[i,j,k] = 1 for all (i,j,k) in the 
    # support... no. The tensor for matrix mult is T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    # So the tensor T has T[i, j, k] = 1 for all valid (i,j,k). Wait no.
    # The tensor is T[i, j, k] = 1. The matrix mult is sum_{j,k} A[i,j] B[j,k] = C[i,k].
    # The tensor T[i,j,k] is 1 for all i, j, k. That's the trivial tensor.
    # No, the tensor is T[i, j, k] = 1. The bilinear form is sum T[i,j,k] U[i,j, c] V[j,k, c] W[k,i, c].
    # For this to equal C[i,k], we need sum_{j, c} U[i,j,c] V[j,k,c] W[k,i,c] ... 
    # Actually the tensor T[i,j,k] is just 1. The decomposition is:
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # where T[i,j,k] = 1 for all i,j,k. That's not right either.
    
    # Let me re-read the problem. The tensor is the matrix multiplication tensor.
    # T[i, j, k] = 1 if we need A[i,j] * B[j,k] for C[i,k]. So T[i,j,k] = 1 for all (i,j,k).
    # The schoolbook decomposition has rank n*m*p, with U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # Wait, U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # The tensor T[i,j,k] is indexed by i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = 1.
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # For schoolbook: each c corresponds to a unique (i,j,k). U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # So T[i,j,k] = 1 for the matching (i,j,k). Yes, that works.
    
    # Now for 2x4x8: n=2, m=4, p=8.
    # The tensor is 2x4x8. We can decompose it into 4 blocks of 2x2x2 (since 4=2*2 and 8=2*4... no).
    # Actually, 2x4x8 can be seen as 2 * (2x2x2) * 4 blocks? No.
    # Let's think: n=2, m=4, p=8.
    # We can write m=4 as 2*2, p=8 as 2*4.
    # So we have 2 blocks in m and 4 blocks in p, giving 8 blocks of 2x2x2.
    # Each 2x2x2 block can be done with Strassen rank 7.
    # So total rank = 8 * 7 = 56.
    
    # Let's construct this.
    # The 2x4x8 tensor: i in [0,2), j in [0,4), k in [0,8).
    # We partition j into 2 groups of 2, and k into 4 groups of 2.
    # So j = 2*jb + jb2, where jb in [0,2), jb2 in [0,2).
    # k = 2*kb + kb2, where kb in [0,4), kb2 in [0,2).
    # i = i.
    # So we have 2*2*4 = 16 blocks? No, 2 (i) * 2 (jb) * 4 (kb) = 16 blocks of size 1x1x1? No.
    
    # Let me reconsider. The tensor T[i,j,k] = 1.
    # We want to decompose it as sum_c U[i, j, c] V[j, k, c] W[k, i, c].
    # But U is (n*m, rank) = (8, rank), V is (m*p, rank) = (32, rank), W is (p*n, rank) = (16, rank).
    # So U is indexed by (i, j), V by (j, k), W by (k, i).
    
    # For Strassen on 2x2x2: we have 7 terms.
    # Let's apply Strassen to each 2x2x2 sub-tensor.
    # The 2x4x8 tensor can be decomposed into 8 sub-tensors of size 2x2x2?
    # n=2, m=4, p=8.
    # If we fix i, j, k, we have a 2x4x8 array.
    # We can think of it as 4 blocks of 2x2x2 (by partitioning m into 2 groups of 2 and p into 2 groups of 4? No).
    
    # Let's just use the known Strassen decomposition for 2x2x2 and tile it.
    # Actually, the tensor is separable in a way that allows tiling.
    # T[i,j,k] = 1 for all i,j,k.
    # We can write T = sum_{jb=0}^{1} sum_{kb=0}^{3} T_{jb,kb} where T_{jb,kb}[i, j, k] = 1 if j in [2*jb, 2*jb+2) and k in [2*kb, 2*kb+2), else 0.
    # So we have 8 sub-tensors, each of size 2x2x2 (with i from 0 to 1, j from 2*jb to 2*jb+1, k from 2*kb to 2*kb+1).
    # Each sub-tensor is the standard 2x2x2 matrix multiplication tensor.
    # So we can use Strassen for each, giving rank 7 per sub-tensor, total 56.
    
    # Strassen for 2x2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[C11,C12],[C21,C22]]
    # C11 = ae + bg
    # C12 = af + bh
    # C21 = ce + dg
    # C22 = cf + dh
    # Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C11 = p1 - p4 + p6
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p3 + p7
    
    # Let's define the Strassen factors for a single 2x2x2 block.
    # U[i, j, c], V[j, k, c], W[k, i, c] for i,j,k in {0,1}.
    # a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
    # e = B[0,0], f = B[0,1], g = B[1,0], h = B[1,1]
    # C11 = C[0,0], C12 = C[0,1], C21 = C[1,0], C22 = C[1,1]
    
    # p1 = (a+d)(e+h) = a*e + a*h + d*e + d*h
    # U: i=0,j=0: 1; i=1,j=1: 1
    # V: j=0,k=0: 1; j=1,k=1: 1
    # W: k=0,i=0: 1; k=1,i=1: 1
    # Wait, W is indexed by (k, i). So W[k*n+i, c] for n=2.
    # W[0*2+0, c] = W[0, c] for (k=0, i=0)
    # W[1*2+1, c] = W[3, c] for (k=1, i=1)
    
    # Let me carefully define the factors.
    # For p1: U[0,0] = 1, U[1,1] = 1; V[0,0] = 1, V[1,1] = 1; W[0,0] = 1, W[1,1] = 1
    # Here U is indexed by (i,j), so U[0,0] means i=0,j=0; U[1,1] means i=1,j=1.
    # V is indexed by (j,k), so V[0,0] means j=0,k=0; V[1,1] means j=1,k=1.
    # W is indexed by (k,i), so W[0,0] means k=0,i=0; W[1,1] means k=1,i=1.
    
    # p2 = (c+d)e = c*e + d*e
    # U[1,0] = 1, U[1,1] = 1; V[0,0] = 1; W[0,1] = 1
    # W[0,1] means k=0, i=1.
    
    # p3 = a(f-h) = a*f - a*h
    # U[0,0] = 1; V[0,1] = 1, V[1,1] = -1; W[1,0] = 1
    # W[1,0] means k=1, i=0.
    
    # p4 = d(g-e) = d*g - d*e
    # U[1,1] = 1; V[1,0] = 1, V[0,0] = -1; W[0,1] = 1
    # Wait, W[0,1] for p4? C21 = p2 + p4, so W should contribute to C[1,0].
    # C[1,0] = C21. W is indexed by (k,i). For C[1,0], i=1, k=0. So W[0,1] = 1.
    # But p2 also contributes to C[1,0]. So W[0,1] = 1 for both p2 and p4? That would give 2.
    # Let me re-check.
    
    # C21 = ce + dg = p2 + p4.
    # p2 = (c+d)e = ce + de.
    # p4 = d(g-e) = dg - de.
    # p2 + p4 = ce + dg. Yes.
    # So W for p2: W[0,1] = 1 (k=0, i=1).
    # W for p4: W[0,1] = 1 (k=0, i=1).
    # So total W[0,1] = 2? No, the sum is over c. So if p2 and p4 both have W[0,1] = 1, then the sum is 2.
    # But we need C21 = ce + dg, so the coefficient should be 1.
    # Let me re-examine.
    
    # Actually, the decomposition is T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # For C21 = T[1,0,0] (i=1, j=0, k=0)? No, T is the tensor, not C.
    # T[i,j,k] = 1.
    # So T[1,0,0] = 1.
    # sum_c U[1,0,c] V[0,0,c] W[0,1,c] should equal 1.
    
    # Let me list all T[i,j,k] and what they should equal.
    # T[0,0,0] = 1 (C11 from a*e)
    # T[0,0,1] = 1 (C12 from a*f)
    # T[0,1,0] = 1 (C11 from b*g)
    # T[0,1,1] = 1 (C12 from b*h)
    # T[1,0,0] = 1 (C21 from c*e)
    # T[1,0,1] = 1 (C22 from c*f)
    # T[1,1,0] = 1 (C21 from d*g)
    # T[1,1,1] = 1 (C22 from d*h)
    
    # Now let's check the Strassen factors for each T[i,j,k].
    # I'll define the factors as lists of (i,j), (j,k), (k,i) for each c.
    
    # c=0: p1 = (a+d)(e+h)
    # U: (0,0), (1,1)
    # V: (0,0), (1,1)
    # W: (0,0), (1,1)
    # Contributes to:
    # T[0,0,0]: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # T[0,0,1]: U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0 (V[0,1] is not set, so 0)
    # Wait, V is indexed by (j,k). For T[0,0,1], j=0, k=1. So V[0,1].
    # In p1, V has (0,0) and (1,1). So V[0,1] = 0.
    # So T[0,0,1] gets 0 from p1.
    # T[0,1,0]: U[0,1]*V[1,0]*W[0,0] = 0*0*1 = 0
    # T[0,1,1]: U[0,1]*V[1,1]*W[1,0] = 0*1*0 = 0
    # T[1,0,0]: U[1,0]*V[0,0]*W[0,1] = 0*1*0 = 0
    # T[1,0,1]: U[1,0]*V[0,1]*W[1,1] = 0*0*1 = 0
    # T[1,1,0]: U[1,1]*V[1,0]*W[0,1] = 1*0*0 = 0
    # T[1,1,1]: U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1
    # So p1 contributes to T[0,0,0] and T[1,1,1].
    
    # c=1: p2 = (c+d)e
    # U: (1,0), (1,1)
    # V: (0,0)
    # W: (0,1)
    # T[1,0,0]: U[1,0]*V[0,0]*W[0,1] = 1*1*1 = 1
    # T[1,1,0]: U[1,1]*V[1,0]*W[0,1] = 1*0*1 = 0 (V[1,0] not set)
    # Others 0.
    
    # c=2: p3 = a(f-h)
    # U: (0,0)
    # V: (0,1), (1,1)
    # W: (1,0)
    # T[0,0,1]: U[0,0]*V[0,1]*W[1,0] = 1*1*1 = 1
    # T[0,1,1]: U[0,1]*V[1,1]*W[1,0] = 0*1*1 = 0
    # Others 0.
    
    # c=3: p4 = d(g-e)
    # U: (1,1)
    # V: (1,0), (0,0)
    # W: (0,1)
    # T[1,1,0]: U[1,1]*V[1,0]*W[0,1] = 1*1*1 = 1
    # T[1,0,0]: U[1,0]*V[0,0]*W[0,1] = 0*1*1 = 0
    # Others 0.
    
    # c=4: p5 = (a+b)h
    # U: (0,0), (0,1)
    # V: (1,1)
    # W: (1,0)
    # T[0,1,1]: U[0,1]*V[1,1]*W[1,0] = 1*1*1 = 1
    # T[0,0,1]: U[0,0]*V[0,1]*W[1,0] = 1*0*1 = 0
    # Others 0.
    
    # c=5: p6 = (c-a)(e+f)
    # U: (1,0), (0,0)
    # V: (0,0), (0,1)
    # W: (0,0)
    # T[0,0,0]: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # T[0,0,1]: U[0,0]*V[0,1]*W[1,0] = 1*1*0 = 0 (W[1,0] not set)
    # T[1,0,0]: U[1,0]*V[0,0]*W[0,1] = 1*1*0 = 0 (W[0,1] not set)
    # T[1,0,1]: U[1,0]*V[0,1]*W[1,1] = 1*1*0 = 0
    # Others 0.
    
    # c=6: p7 = (b-d)(g+h)
    # U: (0,1), (1,1)
    # V: (1,0), (1,1)
    # W: (1,1)
    # T[0,1,1]: U[0,1]*V[1,1]*W[1,0] = 1*1*0 = 0 (W[1,0] not set)
    # Wait, W is (1,1) for p7. So W[1,1] = 1.
    # T[0,1,1]: U[0,1]*V[1,1]*W[1,0] = 1*1*0 = 0. No, W[1,0] is not set, so 0.
    # But we need T[0,1,1] = 1. Let me check p5.
    # p5: W[1,0] = 1. T[0,1,1]: U[0,1]*V[1,1]*W[1,0] = 1*1*1 = 1. Good.
    # p7: T[1,1,1]: U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1.
    # T[0,1,0]: U[0,1]*V[1,0]*W[0,0] = 1*1*0 = 0.
    # T[1,1,0]: U[1,1]*V[1,0]*W[0,1] = 1*1*0 = 0.
    # T[1,1,1]: 1.
    
    # Now let's sum up all contributions to each T[i,j,k].
    # T[0,0,0]: p1(1) + p6(1) = 2. But we need 1!
    # This is wrong. Let me re-check.
    
    # I think I made an error in the Strassen formula. Let me use the standard one.
    # Standard Strassen for 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 - M4 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M7
    
    # Let's re-derive the factors.
    # M1 = (a+d)(e+h)
    # U: a, d -> (0,0), (1,1)
    # V: e, h -> (0,0), (1,1)
    # W: C11, C22 -> (0,0), (1,1) [since W[k,i], C11 is i=0,k=0 -> (0,0); C22 is i=1,k=1 -> (1,1)]
    # So W: (0,0), (1,1)
    
    # M2 = (c+d)e
    # U: c, d -> (1,0), (1,1)
    # V: e -> (0,0)
    # W: C21 -> (0,1) [i=1, k=0]
    
    # M3 = a(f-h)
    # U: a -> (0,0)
    # V: f, h -> (0,1), (1,1)
    # W: C12 -> (1,0) [i=0, k=1]
    
    # M4 = d(g-e)
    # U: d -> (1,1)
    # V: g, e -> (1,0), (0,0)
    # W: C11, C21 -> (0,0), (0,1)
    
    # M5 = (a+b)h
    # U: a, b -> (0,0), (0,1)
    # V: h -> (1,1)
    # W: C12 -> (1,0)
    
    # M6 = (c-a)(e+f)
    # U: c, a -> (1,0), (0,0)
    # V: e, f -> (0,0), (0,1)
    # W: C11 -> (0,0)
    
    # M7 = (b-d)(g+h)
    # U: b, d -> (0,1), (1,1)
    # V: g, h -> (1,0), (1,1)
    # W: C22 -> (1,1)
    
    # Now let's sum.
    # T[0,0,0] (i=0,j=0,k=0):
    # M1: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # M2: U[1,0]*V[0,0]*W[0,1] = 0*1*0 = 0
    # M3: U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0
    # M4: U[1,1]*V[0,0]*W[0,0] = 0*1*1 = 0
    # M5: U[0,0]*V[1,1]*W[1,0] = 1*0*0 = 0
    # M6: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # M7: U[0,1]*V[1,0]*W[1,1] = 0*0*0 = 0
    # Total: 2. Still 2!
    
    # This is not working. Let me look up the correct Strassen factors.
    # I think the issue is that the tensor T[i,j,k] = 1, and the decomposition should give exactly 1 for each T[i,j,k].
    # Let me try a different approach: just use the schoolbook for now and return rank 64.
    # But the task says to find rank-minimal, and Strassen is known to be 7 for 2x2x2.
    
    # Let me just hardcode the schoolbook and return it.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
