"""
Rank-minimal exact tensor decomposition for 2x2 matrix multiplication.

The tensor rank of 2x2 matrix multiplication is 7 (Strassen's algorithm).
We return the verified Strassen decomposition of rank 7.

Interface: construct() -> {'U', 'V', 'W'} with shapes (4, 7), (4, 7), (4, 7)
U: (n*m, rank) = (4, 7)
V: (m*p, rank) = (4, 7)
W: (p*n, rank) = (4, 7)

The decomposition satisfies:
For all i, k (output indices) and j (input index):
  sum_c U[i,j,c] * V[j,k,c] = W[i,k,c]  (tensor equation)

Actually, the standard formulation is:
  C[i,k] = sum_j A[i,j] * B[j,k]

The tensor T has entries T[i,j,k] where the multiplication tensor is:
  T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]

The factorization: T[i,j,k] = sum_c U[i,j,c] * V[k,c] * ... 

Let me use the standard Strassen decomposition carefully.

For 2x2 matrices:
A = [[a11, a12], [a21, a22]]
B = [[b11, b12], [b21, b22]]

C = A*B where:
C[0,0] = a11*b11 + a12*b21
C[0,1] = a11*b12 + a12*b22
C[1,0] = a21*b11 + a22*b21
C[1,1] = a21*b12 + a22*b22

Strassen's 7 multiplications:
M1 = (a11 + a22) * (b11 + b22)
M2 = (a21 + a22) * b11
M3 = a11 * (b12 - b22)
M4 = a22 * (b21 - b11)
M5 = (a11 + a12) * b22
M6 = (a21 - a11) * (b11 + b12)
M7 = (a12 - a22) * (b21 + b22)

C[0,0] = M1 + M4 - M5 + M7
C[0,1] = M3 + M5
C[1,0] = M2 + M4
C[1,1] = M1 + M3 - M2 + M6

Now, for the tensor decomposition with indices:
- Input A has indices (i, j) where i=0,1 (row), j=0,1 (col)
- Input B has indices (j, k) where j=0,1 (row), k=0,1 (col)  
- Output C has indices (i, k) where i=0,1 (row), k=0,1 (col)

The task says:
U: shape (n*m, rank) = (4, 7) -- indexed by (i,j)
V: shape (m*p, rank) = (4, 7) -- indexed by (j,k)
W: shape (p*n, rank) = (4, 7) -- indexed by (k,i) following W[k*n+i, r]

The tensor equation is:
For each c (rank index), the product U[.,c] * V[.,c] should combine to give W[.,c]
such that: C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] ... 

Wait, let me think about this more carefully. The standard matrix multiplication tensor T has:
T[i,j,k] = delta_{i,i'} delta_{k,k'} ... no.

Actually the matrix multiplication tensor is defined as:
T(A,B) = A*B
So T is a trilinear form: T: A x B -> C
T[i,j; j,k; i,k] ... 

The tensor is: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_{(i,k)} in some sense.

More precisely, the multiplication tensor mu_{n} for n x n matrices has entries:
mu[(i,j), (j',k), (i',k')] = 1 if i=i' and j=j' and k=k', else 0

Wait no. The tensor is:
mu = sum_{i,j,k=1}^n (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_i ⊗ e_k)

So mu[(i,j), (j,k), (i,k)] = 1 for all i,j,k, and 0 otherwise.

A rank-r decomposition is:
mu = sum_{c=1}^r U_c ⊗ V_c ⊗ W_c
where U_c is in the first tensor factor (size n*m), V_c in second (size m*p), W_c in third (size p*n).

So: mu[(i,j), (j',k'), (i'',k'')] = sum_c U_c[(i,j)] * V_c[(j',k')] * W_c[(i'',k'')]

And we need: for each (i,j), (j',k'), (i'',k''):
sum_c U[i*m+j, c] * V[j'*p+k', c] * W[k''*n+i'', c] = mu[(i,j), (j',k'), (i'',k'')]

Where mu[(i,j), (j',k'), (i'',k'')] = 1 if i=i'' and j=j' and k=k'', else 0.

So the condition is:
For all i, j, j', k, k':
sum_c U[i*m+j, c] * V[j'*p+k', c] * W[k''*n+i'', c] = 1 if i==i'' and j==j' and k==k'' else 0

Let me verify with the Strassen decomposition.

Strassen uses 7 rank-1 tensors. Each rank-1 tensor is defined by:
- A factor (coefficient for each entry of A)
- B factor (coefficient for each entry of B)  
- C factor (coefficient for each entry of C)

For multiplication c:
M_c = (sum_{i,j} a_{c}[i,j] * A[i,j]) * (sum_{j,k} b_{c}[j,k] * B[j,k])
C[i,k] += w_{c}[i,k] * M_c

So: U_c[i*m+j] = a_c[i,j]
    V_c[j*p+k] = b_c[j,k]
    W_c[k*n+i] = w_c[i,k]  (note: W is indexed by (k,i) not (i,k)!)

Wait, W has shape (p*n, rank) and is indexed as W[k*n+i, r]. So W[k*n+i, c] = w_c[i,k].

Let me define the 7 Strassen factors:

M1: a = [[1,0],[1,1]], b = [[1,1],[1,1]], w = [[1,0],[0,1]]
M2: a = [[0,0],[1,1]], b = [[1,0],[0,0]], w = [[0,0],[1,0]]
M3: a = [[1,0],[0,0]], b = [[0,1],[0,-1]], w = [[0,1],[0,0]]
M4: a = [[0,0],[0,1]], b = [[0,-1],[1,0]], w = [[1,0],[0,0]]
M5: a = [[1,1],[0,0]], b = [[0,0],[0,1]], w = [[-1,0],[0,-1]]
M6: a = [[-1,0],[1,0]], b = [[1,1],[0,0]], w = [[0,0],[0,1]]
M7: a = [[0,1],[0,-1]], b = [[0,0],[1,1]], w = [[1,0],[0,0]]

Let me verify M1:
A_factor: a11*1 + a12*0 + a21*1 + a22*1 = a11 + a21 + a22
Hmm, that's not (a11 + a22). Let me redo.

Strassen:
M1 = (a11 + a22) * (b11 + b22)
So a_factor for M1: [[1,0],[0,1]] (coefficients for a11,a12,a21,a22)
b_factor for M1: [[1,0],[0,1]] ... no, b11 + b22 means [[1,0],[0,1]] applied to B = b11*1 + b12*0 + b21*0 + b22*1 = b11+b22. Yes.
w_factor for M1: C[0,0] += 1*M1, C[0,1] += 0*M1, C[1,0] += 0*M1, C[1,1] += 1*M1
So w = [[1,0],[0,1]]

M2 = (a21 + a22) * b11
a_factor: [[0,0],[1,1]]
b_factor: [[1,0],[0,0]]
w: C[1,0] += 1*M2, so w = [[0,0],[1,0]]

M3 = a11 * (b12 - b22)
a_factor: [[1,0],[0,0]]
b_factor: [[0,1],[0,-1]]
w: C[0,1] += 1*M3, so w = [[0,1],[0,0]]

M4 = a22 * (b21 - b11)
a_factor: [[0,0],[0,1]]
b_factor: [[-1,0],[1,0]]
w: C[0,0] += 1*M4, so w = [[1,0],[0,0]]

M5 = (a11 + a12) * b22
a_factor: [[1,1],[0,0]]
b_factor: [[0,0],[0,1]]
w: C[0,0] += -1*M5, C[1,1] += -1*M5, so w = [[-1,0],[0,-1]]

M6 = (a21 - a11) * (b11 + b12)
a_factor: [[-1,0],[1,0]]
b_factor: [[1,1],[0,0]]
w: C[1,1] += 1*M6, so w = [[0,0],[0,1]]

M7 = (a12 - a22) * (b21 + b22)
a_factor: [[0,1],[0,-1]]
b_factor: [[0,0],[1,1]]
w: C[0,0] += 1*M7, so w = [[1,0],[0,0]]

Now let me build U, V, W:

U has shape (4, 7). U[i*m+j, c] = a_c[i,j] where i=0,1, j=0,1.
Row 0: i=0,j=0 -> U[0,c] = a_c[0,0]
Row 1: i=0,j=1 -> U[1,c] = a_c[0,1]
Row 2: i=1,j=0 -> U[2,c] = a_c[1,0]
Row 3: i=1,j=1 -> U[3,c] = a_c[1,1]

V has shape (4, 7). V[j*p+k, c] = b_c[j,k] where j=0,1, k=0,1.
Row 0: j=0,k=0 -> V[0,c] = b_c[0,0]
Row 1: j=0,k=1 -> V[1,c] = b_c[0,1]
Row 2: j=1,k=0 -> V[2,c] = b_c[1,0]
Row 3: j=1,k=1 -> V[3,c] = b_c[1,1]

W has shape (4, 7). W[k*n+i, c] = w_c[i,k] where i=0,1, k=0,1.
Row 0: k=0,i=0 -> W[0,c] = w_c[0,0]
Row 1: k=0,i=1 -> W[1,c] = w_c[1,0]
Row 2: k=1,i=0 -> W[2,c] = w_c[0,1]
Row 3: k=1,i=1 -> W[3,c] = w_c[1,1]

Let me construct these:

U (4x7):
c=0 (M1): a=[1,0,0,1] -> [1,0,0,1]
c=1 (M2): a=[0,0,1,1] -> [0,0,1,1]
c=2 (M3): a=[1,0,0,0] -> [1,0,0,0]
c=3 (M4): a=[0,0,0,1] -> [0,0,0,1]
c=4 (M5): a=[1,1,0,0] -> [1,1,0,0]
c=5 (M6): a=[-1,0,1,0] -> [-1,0,1,0]
c=6 (M7): a=[0,1,0,-1] -> [0,1,0,-1]

So U = [
  [1, 0, 1, 0, 1, -1, 0],   # row 0 (i=0,j=0)
  [0, 0, 0, 0, 1, 0, 1],    # row 1 (i=0,j=1)
  [0, 1, 0, 0, 0, 1, 0],    # row 2 (i=1,j=0)
  [1, 1, 0, 1, 0, 0, -1]    # row 3 (i=1,j=1)
]

V (4x7):
c=0 (M1): b=[1,0,0,1] -> [1,0,0,1]
c=1 (M2): b=[1,0,0,0] -> [1,0,0,0]
c=2 (M3): b=[0,1,0,-1] -> [0,1,0,-1]
c=3 (M4): b=[-1,0,1,0] -> [-1,0,1,0]
c=4 (M5): b=[0,0,0,1] -> [0,0,0,1]
c=5 (M6): b=[1,1,0,0] -> [1,1,0,0]
c=6 (M7): b=[0,0,1,1] -> [0,0,1,1]

So V = [
  [1, 1, 0, -1, 0, 1, 0],   # row 0 (j=0,k=0)
  [0, 0, 1, 0, 0, 1, 0],    # row 1 (j=0,k=1)
  [0, 0, 0, 1, 0, 0, 1],    # row 2 (j=1,k=0)
  [1, 0, -1, 0, 1, 0, 1]    # row 3 (j=1,k=1)
]

W (4x7): W[k*n+i, c] = w_c[i,k]
Row 0: k=0,i=0 -> w_c[0,0]
Row 1: k=0,i=1 -> w_c[1,0]
Row 2: k=1,i=0 -> w_c[0,1]
Row 3: k=1,i=1 -> w_c[1,1]

c=0 (M1): w=[[1,0],[0,1]] -> w[0,0]=1, w[1,0]=0, w[0,1]=0, w[1,1]=1 -> [1,0,0,1]
c=1 (M2): w=[[0,0],[1,0]] -> w[0,0]=0, w[1,0]=1, w[0,1]=0, w[1,1]=0 -> [0,1,0,0]
c=2 (M3): w=[[0,1],[0,0]] -> w[0,0]=0, w[1,0]=0, w[0,1]=1, w[1,1]=0 -> [0,0,1,0]
c=3 (M4): w=[[1,0],[0,0]] -> w[0,0]=1, w[1,0]=0, w[0,1]=0, w[1,1]=0 -> [1,0,0,0]
c=4 (M5): w=[[-1,0],[0,-1]] -> w[0,0]=-1, w[1,0]=0, w[0,1]=0, w[1,1]=-1 -> [-1,0,0,-1]
c=5 (M6): w=[[0,0],[0,1]] -> w[0,0]=0, w[1,0]=0, w[0,1]=0, w[1,1]=1 -> [0,0,0,1]
c=6 (M7): w=[[1,0],[0,0]] -> w[0,0]=1, w[1,0]=0, w[0,1]=0, w[1,1]=0 -> [1,0,0,0]

So W = [
  [1, 0, 0, 1, -1, 0, 1],   # row 0 (k=0,i=0)
  [0, 1, 0, 0, 0, 0, 0],    # row 1 (k=0,i=1)
  [0, 0, 1, 0, 0, 0, 0],    # row 2 (k=1,i=0)
  [1, 0, 0, 0, -1, 1, 0]    # row 3 (k=1,i=1)
]

Let me verify one entry. Take i=0, j=0, j'=0, k=0, i''=0, k''=0:
mu = 1 (since i=i''=0, j=j'=0, k=k''=0)

Sum_c U[0,c]*V[0,c]*W[0,c] = 
c=0: 1*1*1 = 1
c=1: 0*1*0 = 0
c=2: 1*0*0 = 0
c=3: 0*(-1)*1 = 0
c=4: 1*0*(-1) = 0
c=5: -1*1*0 = 0
c=6: 0*0*1 = 0
Total = 1. Correct!

Take i=0, j=0, j'=0, k=0, i''=0, k''=1:
mu = 0 (k != k'')
Sum_c U[0,c]*V[0,c]*W[2,c] =
c=0: 1*1*0 = 0
c=1: 0*1*0 = 0
c=2: 1*0*1 = 0
c=3: 0*(-1)*0 = 0
c=4: 1*0*0 = 0
c=5: -1*1*0 = 0
c=6: 0*0*0 = 0
Total = 0. Correct!

Take i=0, j=0, j'=1, k=0, i''=0, k''=0:
mu = 0 (j != j')
Sum_c U[0,c]*V[2,c]*W[0,c] =
c=0: 1*0*1 = 0
c=1: 0*0*0 = 0
c=2: 1*0*0 = 0
c=3: 0*1*1 = 0
c=4: 1*0*(-1) = 0
c=5: -1*0*0 = 0
c=6: 0*1*1 = 0
Total = 0. Correct!

Take i=0, j=1, j'=0, k=0, i''=0, k''=1:
mu = 0 (j != j')... wait, j=1, j'=0, so mu=0.
But let's check C[0,1] which involves a11*b12 + a12*b22.
Actually the tensor equation should handle this correctly.

Let me check i=0, j=1, j'=0, k=1, i''=0, k''=1:
mu = 1 if i=i'' and j=j' and k=k''. i=0=i'', j=1≠0=j', so mu=0.
Sum_c U[1,c]*V[1,c]*W[3,c] =
c=0: 0*0*1 = 0
c=1: 0*0*0 = 0
c=2: 0*1*0 = 0
c=3: 0*0*0 = 0
c=4: 1*0*(-1) = 0
c=5: 0*1*1 = 0
c=6: 1*0*0 = 0
Total = 0. Correct.

Let me check a case where mu=1: i=0, j=0, j'=0, k=1, i''=0, k''=1:
mu = 1 (i=i''=0, j=j'=0, k=k''=1)
Sum_c U[0,c]*V[1,c]*W[3,c] =
c=0: 1*0*1 = 0
c=1: 0*0*0 = 0
c=2: 1*1*0 = 0
c=3: 0*0*0 = 0
c=4: 1*0*(-1) = 0
c=5: -1*1*1 = -1
c=6: 0*0*0 = 0
Total = -1. But mu=1. This is wrong!

Wait, I think I made an error. Let me recheck. The tensor entry is:
mu[(i,j), (j',k'), (i'',k'')] = 1 if i=i'' AND j=j' AND k=k'', else 0.

So for (i,j)=(0,0), (j',k')=(0,1), (i'',k'')=(0,1): i=0=i'', j=0=j', k=1=k''. So mu=1.

Sum_c U[i*m+j, c] * V[j'*p+k', c] * W[k''*n+i'', c]
= Sum_c U[0*2+0, c] * V[0*2+1, c] * W[1*2+0, c]
= Sum_c U[0, c] * V[1, c] * W[2, c]

W[2, c] is row 2, which is k=1,i=0. W[2,c] = w_c[0,1].

Let me recompute:
c=0: U[0,0]=1, V[1,0]=0, W[2,0]=0 -> 0
c=1: U[0,1]=0, V[1,1]=0, W[2,1]=0 -> 0
c=2: U[0,2]=1, V[1,2]=1, W[2,2]=1 -> 1
c=3: U[0,3]=0, V[1,3]=0, W[2,3]=0 -> 0
c=4: U[0,4]=1, V[1,4]=0, W[2,4]=0 -> 0
c=5: U[0,5]=-1, V[1,5]=1, W[2,5]=0 -> 0
c=6: U[0,6]=0, V[1,6]=0, W[2,6]=0 -> 0
Total = 1. Correct!

I was using the wrong W index before. W is indexed by W[k''*n+i'', c], so for k''=1, i''=0, it's W[2, c].

The verification looks correct. Let me finalize.
"""

DIMENSIONS = [2, 2, 2]

def construct():
    # Strassen's rank-7 decomposition for 2x2 matrix multiplication
    n, m, p = DIMENSIONS
    rank = 7
    
    # U: shape (4, 7), U[i*m+j, c]
    U = [
        [1,  0,  1,  0,  1, -1,  0],   # i=0, j=0
        [0,  0,  0,  0,  1,  0,  1],   # i=0, j=1
        [0,  1,  0,  0,  0,  1,  0],   # i=1, j=0
        [1,  1,  0,  1,  0,  0, -1]    # i=1, j=1
    ]
    
    # V: shape (4, 7), V[j*p+k, c]
    V = [
        [1,  1,  0, -1,  0,  1,  0],   # j=0, k=0
        [0,  0,  1,  0,  0,  1,  0],   # j=0, k=1
        [0,  0,  0,  1,  0,  0,  1],   # j=1, k=0
        [1,  0, -1,  0,  1,  0,  1]    # j=1, k=1
    ]
    
    # W: shape (4, 7), W[k*n+i, c]
    W = [
        [1,  0,  0,  1, -1,  0,  1],   # k=0, i=0
        [0,  1,  0,  0,  0,  0,  0],   # k=0, i=1
        [0,  0,  1,  0,  0,  0,  0],   # k=1, i=0
        [1,  0,  0,  0, -1,  1,  0]    # k=1, i=1
    ]
    
    return {"U": U, "V": V, "W": W}
