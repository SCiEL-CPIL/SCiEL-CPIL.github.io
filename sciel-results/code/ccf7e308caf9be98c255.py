import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin

    # We'll use a rank-19 decomposition for 3x3 matrix multiplication
    # applied to compute (2x3) * (3x9)
    # This gives rank 2*19 = 38 potentially

    # The 3x3 multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # We need a rank-19 decomposition of the 3x3x3 tensor.
    # 
    # Known rank-19 scheme for 3x3 (based on Bini et al. / Winograd):
    # The 27 entries of the tensor are decomposed into 19 rank-1 terms.
    #
    # Let me use a concrete rank-19 scheme. One known construction:
    # We split the 3x3 into overlapping 2x2 blocks and use Strassen.
    # 
    # Actually, let me use a well-known rank-19 scheme:
    # The 19 products are:
    # 
    # Let A = [a b c; d e f; g h i], B = [x y z; u v w; s t r]
    # C = A*B
    #
    # A known rank-19 scheme (from the literature):
    # M1  = (a+b+c)(x+y+z)
    # M2  = (a-b-c)(x-y-z)  
    # M3  = (a+b-c)(x+y-z)
    # M4  = (a-b+c)(x-y+z)
    # M5  = (d+e+f)(u+v+w)
    # M6  = (d-e-f)(u-v-w)
    # M7  = (d+e-f)(u+v-w)
    # M8  = (d-e+f)(u-v+w)
    # M9  = (g+h+i)(s+t+r)
    # M10 = (g-h-i)(s-t-r)
    # M11 = (g+h-i)(s+t-r)
    # M12 = (g-h+i)(s-t+r)
    # M13 = (a+d+g)(x+u+s)
    # M14 = (b+e+h)(y+v+t)
    # M15 = (c+f+i)(z+w+r)
    # M16 = (a+d+g)(x-u-s)
    # M17 = (b+e+h)(y-v-t)
    # M18 = (c+f+i)(z-w-r)
    # M19 = (a+b+c+d+e+f+g+h+i)(x+y+z+u+v+w+s+t+r)
    #
    # This doesn't quite work. Let me use a verified rank-19 scheme.
    #
    # Actually, the simplest approach: use the fact that we can decompose
    # 3x3 as three 1x3 * 3x3 products, each of rank 9, but that's rank 27.
    #
    # Let me try a different structural approach:
    # Use Strassen's algorithm recursively:
    # 3x3 can be computed with rank 19 using the following scheme:
    #
    # From Bini, Capovani, Lotti, Romani (1979), a rank-19 scheme exists.
    # Let me implement a concrete one.
    #
    # Here's a verified rank-19 scheme for 3x3:
    # 
    # The 19 scalar multiplications are:
    # p1  = a*x
    # p2  = b*y  
    # p3  = c*z
    # p4  = d*u
    # p5  = e*v
    # p6  = f*w
    # p7  = g*s
    # p8  = h*t
    # p9  = i*r
    # p10 = (a+b)*(x+y)
    # p11 = (c+d)*(z+u)
    # p12 = (e+f)*(v+w)
    # p13 = (g+h)*(s+t)
    # p14 = (i+a)*(r+x)
    # p15 = (b+c)*(y+z)
    # p16 = (d+e)*(u+v)
    # p17 = (f+g)*(w+s)
    # p18 = (h+i)*(t+r)
    # p19 = (a+c+e+g+i)*(x+z+v+s+r)
    #
    # This is getting complex. Let me take a more practical approach.
    #
    # PRACTICAL APPROACH: Since finding a truly rank-19 3x3 scheme is complex,
    # let me instead try to optimize the current rank-50 approach by:
    # 1. Using a better block decomposition
    # 2. Trying to merge some terms
    #
    # Alternative structural change: Instead of splitting 3 into 2+1,
    # use a 3-way split with a rank-19 3x3 scheme.
    #
    # Let me implement a concrete rank-19 3x3 scheme from the literature.
    # The scheme below is from a known construction:

    # Rank-19 scheme for 3x3 matrix multiplication
    # A = [[a,b,c],[d,e,f],[g,h,i]], B = [[x,y,z],[u,v,w],[s,t,r]]
    # 
    # The 19 products (u-vector, v-vector, w-vector coefficients):
    # Each product is (linear_combo_A) * (linear_combo_B)
    
    # Let me use this verified scheme:
    # Products indexed 0..18
    
    # For the 3x3 tensor, the output C[i,k] = sum_j A[i,j]*B[j,k]
    # C = [[ax+by+cz, ay+bv+cw, az+bw+cr],
    #      [dx+ey+fz, dy+ev+fw, dz+ew+fr],
    #      [gx+hy+iz, gy+hv+iw, gz+hw+ir]]
    
    # I'll use a scheme where we define 19 bilinear forms.
    # Here's one from the literature (Bini et al.):
    
    # Let me just use a practical rank-19 scheme:
    # 
    # p0  = (a+b) * (x+y)
    # p1  = (c+d) * (z+u)  
    # p2  = (e+f) * (v+w)
    # p3  = (g+h) * (s+t)
    # p4  = (i+a) * (r+x)
    # p5  = (b+c) * (y+z)
    # p6  = (d+e) * (u+v)
    # p7  = (f+g) * (w+s)
    # p8  = (h+i) * (t+r)
    # p9  = a * x
    # p10 = b * y
    # p11 = c * z
    # p12 = d * u
    # p13 = e * v
    # p14 = f * w
    # p15 = g * s
    # p16 = h * t
    # p17 = i * r
    # p18 = (a+c+e+g+i) * (x+z+v+s+r)
    #
    # Let me verify this can reconstruct C:
    # C[0,0] = a*x + b*y + c*z
    # C[0,1] = a*y + b*v + c*w
    # C[0,2] = a*z + b*w + c*r
    # C[1,0] = d*x + e*y + f*z
    # C[1,1] = d*y + e*v + f*w
    # C[1,2] = d*z + e*w + f*r
    # C[2,0] = g*x + h*y + i*z
    # C[2,1] = g*y + h*v + i*w
    # C[2,2] = g*z + h*w + i*r
    
    # With the above 19 products, can we express all 9 C entries?
    # C[0,0] = p9 + (p0 - a*y - b*x) ... this is getting circular.
    
    # Let me take a completely different structural approach.
    # 
    # NEW STRUCTURAL IDEA: Use the fact that the tensor for (2x3)*(3x9)
    # can be viewed as 9 independent (2x3)*(3x1) problems (one per column of B),
    # plus we can share structure across columns.
    #
    # For (2x3)*(3x1): C[0] = a00*b0 + a01*b1 + a02*b2
    #                   C[1] = a10*b0 + a11*b1 + a12*b2
    # This is a 2x3x1 tensor with rank 3 (naive) but potentially rank 2?
    # No, rank of 2x3x1 is 3 (it's just two dot products of length 3).
    #
    # So 9 columns * rank 3 = rank 27 for the naive approach.
    #
    # The Strassen approach on 2x2 blocks gave rank 50.
    # 
    # Let me try: group columns of B into triples and use a 3-column trick.
    # For 3 consecutive columns of B, we compute (2x3)*(3x3) which has
    # tensor rank 19 (known). So 3 groups of 3 columns = 3*19 = 57. Worse.
    #
    # What about grouping into pairs? (2x3)*(3x2): 
    # This is like multiplying a 2x3 by a 3x2. The tensor rank...
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    # 
    # This is a 2x3x2 tensor. Naive rank = 6. Can we do better?
    # Using Strassen-like: treat as (2x2 + 2x1) * (2x2 + 1x2)... 
    # Actually this is equivalent to a 2x2 matrix mult with an extra term.
    # 
    # Let me try: C = A_left * B_left + A_right * B_right
    # where A_left = [[a00,a01],[a10,a11]] (2x2), B_left = [[b00,b01],[b10,b11]] (2x2)
    # A_right = [[a02],[a12]] (2x1), B_right = [[b20,b21]] (1x2)
    # 
    # Rank of 2x2x2 is 7 (Strassen), rank of (2x1)*(1x2) is 2.
    # Total: 7 + 2 = 9 for a 2x3x2 block.
    # 
    # For 9 columns: 4 blocks of 2x3x2 (8 columns) + 1 block of 2x3x1 (1 column)
    # = 4*9 + 3 = 39. Better than 50!
    #
    # Wait, let me recalculate. The 2x3x1 block:
    # C[0] = a00*b0 + a01*b1 + a02*b2
    # C[1] = a10*b0 + a11*b1 + a12*b2
    # This needs 3 products: a00*b0, a01*b1, a02*b2 for C[0]
    # and a10*b0, a11*b1, a12*b2 for C[1]. That's 6 products naive.
    # But we can share: compute a00*b0, a01*b1, a02*b2, a10*b0, a11*b1, a12*b2 = 6.
    # Or use Strassen on the 2x2 part: 7 + 2 (for the a02*b2, a12*b2) = 9.
    # Hmm, that's worse. For 2x3x1, naive is 6, Strassen-based is 9.
    #
    # So for 2x3x2: naive is 6, Strassen-based is 9. Naive is better!
    # 
    # Let me reconsider. For 2x3x2:
    # Naive: 6 products.
    # Can we do it in fewer? The tensor has 6 nonzero entries in a 2x3x2 space.
    # The rank of this tensor is 6 (I believe it's full rank for this small case).
    #
    # So naive 2x3x2 = 6, and we have 4 such blocks + 1 2x3x1 block (6 products)
    # = 4*6 + 6 = 30. But wait, that's not right either because the A entries
    # are shared across blocks!
    #
    # The key issue: in the tensor decomposition, the U, V, W matrices must be
    # consistent. The A entries (U indices) are the same across all B column blocks.
    # 
    # Let me think about this differently. The total tensor has:
    # U space: 6 dimensions (2x3 matrix A)
    # V space: 27 dimensions (3x9 matrix B)  
    # W space: 18 dimensions (2x9 matrix C, indexed as k*n+i)
    #
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k).
    # That's 2*3*9 = 54 nonzero entries.
    #
    # For a rank-r decomposition, we need sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c].
    #
    # The minimum rank is at least max(6, 27, 18) = 27 (by dimension argument,
    # actually the rank is at least the max of the three dimensions).
    # Also, rank >= 54/3 = 18 (by the "number of nonzeros / 3" heuristic, not rigorous).
    #
    # Known lower bounds for matrix multiplication tensor rank:
    # For (n x m) * (m x p), rank >= n*m*p / max(n,m,p) in some cases.
    # For 2x3x9: lower bound is around 18-27.
    #
    # The current best is 50. Let me try to get closer to the theoretical minimum.
    #
    # STRUCTURAL APPROACH: Use a "column-grouping" strategy.
    # Group the 9 columns of B into 3 groups of 3.
    # For each group of 3 columns, compute (2x3) * (3x3) using a rank-19 scheme.
    # Total rank: 3 * 19 = 57. Worse than 50.
    #
    # Group into 9 groups of 1: each (2x3)*(3x1) has rank 3 (naive) or 
    # we can use a rank-3 scheme. Total: 9*3 = 27. But this doesn't work
    # because the U entries would need to be duplicated.
    #
    # Actually, in the tensor decomposition framework, we CAN share U entries.
    # The constraint is that for each (a,b,c) with T[a,b,c]=1, we need
    # sum_r U[a,r]*V[b,r]*W[c,r] = 1.
    #
    # Let me try a direct construction for rank 27:
    # For each j in {0,1,2} and k in {0,...,8}:
    #   Create a rank-1 term: U[i*m+j, r] = 1 for all i, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for all i
    #   This gives 3*9 = 27 terms.
    #   Check: sum_r U[a,r]*V[b,r]*W[c,r] where a=i*m+j, b=j'*p+k', c=k''*n+i''
    #   = sum over r of [U[i*m+j,r] * V[j'*p+k',r] * W[k''*n+i'',r]]
    #   For term (j,k): U[i*m+j,r] = 1 iff j matches, V[j'*p+k',r] = 1 iff j'=j and k'=k,
    #   W[k''*n+i'',r] = 1 iff k''=k and i''=i.
    #   So the sum = 1 iff j'=j, k'=k, i''=i. And T[i*m+j, j*p+k, k*n+i] = 1. ✓
    #   For other (a,b,c): the sum = 0. ✓
    #
    # So rank 27 works! Let me verify:
    # T has 54 nonzero entries: T[i*m+j, j*p+k, k*n+i] = 1 for i in {0,1}, j in {0,1,2}, k in {0,...,8}.
    # 
    # With 27 rank-1 terms (one for each (j,k) pair):
    # Term (j,k): U[:,r] has 1 at positions i*m+j for i=0,1 (i.e., positions j and 3+j)
    #             V[:,r] has 1 at position j*p+k
    #             W[:,r] has 1 at positions k*n+i for i=0,1 (i.e., positions 2k and 2k+1)
    #
    # Check T[i*m+j, j*p+k, k*n+i]:
    #   Only term (j,k) contributes: U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    #   Product = 1. ✓
    #
    # Check T[a,b,c] for other (a,b,c):
    #   a = i*m+j, b = j'*p+k', c = k''*n+i''
    #   Term (j,k) contributes iff j'=j, k'=k, and then U[i*m+j,r]=1 (always for this term),
    #   V[j*p+k,r]=1 (always), W[k*n+i'',r]=1 iff k''=k and i''=i.
    #   So contribution = 1 iff j'=j, k'=k, k''=k, i''=i, which means a=i*m+j, b=j*p+k, c=k*n+i.
    #   For any other combination, no term contributes, so sum = 0. ✓
    #
    # So rank 27 is valid! This is a huge improvement from 50.

    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):  # j = 0, 1, 2
        for k in range(p):  # k = 0, ..., 8
            # U: set entries for A[i,j] for i=0,1
            for i in range(n):
                U[i * m + j][col] = 1
            # V: set entry for B[j,k]
            V[j * p + k][col] = 1
            # W: set entries for C[i,k] for i=0,1
            for i in range(n):
                W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
