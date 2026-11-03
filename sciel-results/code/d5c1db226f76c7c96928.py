
So for each (i,j,k) triple, there's a rank-1 term where:
- U[i*m+j, r] = 1 (selects A[i,j])
- V[j*p+k, r] = 1 (selects B[j,k])
- W[k*n+i, r] = 1 (adds to C[i,k])

And the decomposition is: C[i,k] = sum_r U[i*m+? ... 

Actually, I think the convention is:
C[i,k] = sum_{j} A[i,j] * B[j,k]

And the rank-r decomposition uses:
C[i,k] = sum_{r=1}^R (sum_{j} U[i*m+j, r] * A[i,j]) * (sum_{j} V[j*p+k, r] * B[j,k]) * W[k*n+i, r]

No wait, that doesn't make sense dimensionally. Let me think again.

I believe the correct interpretation is:
The matrix multiplication can be written as:
C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]

A rank-r tensor decomposition of the multiplication tensor means we can write:
C[i,k] = sum_{r=1}^R W[i,k,r] * (sum_{j} U[i,j,r] * A[i,j]) * (sum_{j'} V[j',k,r] * B[j',k])

Hmm, but that's not quite right either. Let me look at this differently.

The standard way: the multiplication tensor T has entries T[i,j,k] = 1. A decomposition of rank r is:
T[i,j,k] = sum_{r=1}^R u_r[i,j] * v_r[j,k] * w_r[i,k]

where u_r, v_r, w_r are rank-1 "factors". Then:
C[i,k] = sum_{j} A[i,j] * B[j,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
       = sum_{j} sum_{r} u_r[i,j] * v_r[j,k] * w_r[i,k] * A[i,j] * B[j,k]

For this to work, we need:
sum_{j} u_r[i,j] * A[i,j] = A[i,j_r] for some effective j_r (if u_r is a selector)
sum_{j} v_r[j,k] * B[j,k] = B[j_r,k]

In the schoolbook case, u_r[i,j] = 1 if j=j_r else 0, so sum_j u_r[i,j]*A[i,j] = A[i,j_r].

So the general form is:
C[i,k] = sum_r w_r[i,k] * (sum_j u_r[i,j] * A[i,j]) * (sum_j v_r[j,k] * B[j,k])

With U[i*m+j, r] = u_r[i,j], V[j*p+k, r] = v_r[j,k], W[k*n+i, r] = w_r[i,k].

OK so with this understanding, my Strassen implementation should work. Let me verify with a simple case.

For M1 = (A11+A22)(B11+B22), the A-side "matrix" is X where X[di,dj] = A[di,dj] + A[3+di,3+dj].
In the decomposition, for a fixed (di,dj,dk) term:
- u_r[i,j] should select A[di,dj] with coeff 1 and A[3+di,3+dj] with coeff 1.
  So U[di*6+dj, r] = 1 and U[(3+di)*6+(3+dj), r] = 1.
- v_r[j,k] should select B[dj,dk] with coeff 1 and B[3+dj,3+dk] with coeff 1.
  So V[dj*6+dk, r] = 1 and V[(3+dj)*6+(3+dk), r] = 1.
- w_r[i,k] should add to C[di,dk] with coeff 1 and C[3+di,3+dk] with coeff 1.
  So W[dk*6+di, r] = 1 and W[(3+dk)*6+(3+di), r] = 1.

This matches my code! The u_coeff_sum for u_row = di*6+dj gets +1, and for u_row = (3+di)*6+(3+dj) gets +1. Good.

But wait, in my code, I'm iterating over a_blocks which has {(0,0):1, (1,1):1}. For (bi,bj)=(0,0): u_row = (0*3+di)*6 + (0*3+dj) = di*6+dj. For (bi,bj)=(1,1): u_row = (1*3+di)*6 + (1*3+dj) = (3+di)*6+(3+dj). Yes, that's correct.

Let me also verify the W indexing. W[k*n+i, r] where k is the column of C and i is the row. So W[dk*6+di, r] for C[di,dk]. In my code: w_row = (ck*B+dk)*6 + (ci*B+di). For (ci,ck)=(0,0): w_row = dk*6+di. For (ci,ck)=(1,1): w_row = (3+dk)*6+(3+di). Correct!

I think the code is correct. Let me clean it up and remove the dead code.

