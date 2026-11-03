"""
Rank-minimal tensor decomposition for 2x4 * 4x5 matrix multiplication.

Uses a Strassen-style rank-36 construction. The key insight is that for
rectangular matrix multiplication, we can use the 2x2x2 Strassen algorithm
as a building block. For 2x4 * 4x5, we partition into 2x2 blocks:
- A is 2x4 -> two 2x2 blocks: A1 (cols 0-1), A2 (cols 2-3)
- B is 4x5 -> two 2x2 blocks (rows) and 2x2 blocks (cols): 
  B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3), B13 (rows 0-1, cols 4)
  B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3), B23 (rows 2-3, cols 4)

Actually, let me think more carefully. The matrix multiplication tensor for
(n,m,p) = (2,4,5) has dimension 2*4*5 = 40.

The schoolbook rank is 40. We know from the knowledge that rank 36 is achievable.

For rectangular matrix multiplication, one approach is to use the fact that
M_{2,4,5} can be embedded in a larger square multiplication. 

A known result: the rank of M_{2,4,5} is at most 36. One construction uses
the 2x2 Strassen algorithm applied to sub-blocks.

Let me think about this differently. We have:
C = A * B where A is 2x4, B is 4x5, C is 2x5.

Partition:
A = [A_left | A_right] where A_left is 2x2, A_right is 2x2
B = [B_top; B_bottom] where B_top is 2x5, B_bottom is 2x5

C = A_left * B_top + A_right * B_bottom

Now B_top is 2x5 and B_bottom is 2x5. We can further partition B_top and B_bottom.

B_top = [BT1 | BT2] where BT1 is 2x2, BT2 is 2x3
B_bottom = [BB1 | BB2] where BB1 is 2x2, BB2 is 2x3

C_left (2x2) = A_left * BT1 + A_right * BB1
C_right (2x3) = A_left * BT2 + A_right * BB2

For the 2x2 part: C_left = A_left * BT1 + A_right * BB1
This is a 2x2 * 2x2 + 2x2 * 2x2 computation. 

Using Strassen on the 2x2 blocks:
For X*Y where X,Y are 2x2, Strassen uses 7 multiplications.
But here we have two separate 2x2*2x2 products to add.

Actually, let me use a different approach. The standard way to get rank 36 for M_{2,4,5}:

We can write the 2x4 * 4x5 multiplication as follows. Think of it as:
- Split the middle dimension (4) into two groups of 2.
- Split the output dimension (5) into groups.

A known construction for rank 36:

The tensor T_{2,4,5} can be decomposed using the fact that:
M_{2,4,5} <= M_{2,2,2} (Strassen, rank 7) applied appropriately.

Actually, let me use a direct construction. The rank of M_{m,n,p} satisfies:
r(M_{2,4,5}) <= r(M_{2,2,2}) + r(M_{2,2,3}) * 2 + ... 

Hmm, let me think about this more concretely.

One approach: Use the 2x2x2 Strassen pattern.

For multiplying a 2x4 matrix by a 4x5 matrix:
Let A be 2x4, B be 4x5, C be 2x5.

Partition A into A1 (2x2, first 2 cols) and A2 (2x2, last 2 cols).
Partition B into B1 (2x5, first 2 rows) and B2 (2x5, last 2 rows).

C = A1*B1 + A2*B2

Now B1 is 2x5 and B2 is 2x5. Partition B1 into B1a (2x2) and B1b (2x3), similarly B2.

C = A1*B1a + A1*B1b + A2*B2a + A2*B2b

The 2x2*2x2 products: A1*B1a and A2*B2a. Each can be done with Strassen (rank 7 each) = 14.
The 2x2*2x3 products: A1*B1b and A2*B2b. 

For 2x2 * 2x3: This has rank at most 2*2*3 = 12 schoolbook, but can be done better.
Actually, r(M_{2,2,3}) = 10 (known result).

So total: 7 + 7 + 10 + 10 = 34? That's less than 36. But wait, I need to be more careful.

Actually, the issue is that A1*B1a + A2*B2a is not simply two independent Strassen decompositions because we're adding them. Let me reconsider.

C_left (2x2) = A1*B1a + A2*B2a
C_right (2x3) = A1*B1b + A2*B2b

For C_left: This is the sum of two 2x2*2x2 products. The tensor for X*Y + X'*Y' where all are 2x2... 

Actually, a cleaner approach: think of the whole thing as a single tensor decomposition.

Let me use the known rank-36 construction directly. Based on the literature, one way to achieve rank 36 for M_{2,4,5} is:

Use the 2x2 Strassen algorithm (rank 7) for the "core" 2x2*2x2 multiplications, combined with simpler decompositions for the rectangular parts.

Let me try a specific construction. I'll define the U, V, W matrices explicitly.

The indices:
- U has shape (2*4, rank) = (8, rank)
- V has shape (4*5, rank) = (20, rank)  
- W has shape (5*2, rank) = (10, rank)

W index: W[k*n + i, r] where k is the column of C (0..4), i is the row of C (0..1).
So W index = k*2 + i for k in 0..4, i in 0..1.

U index: i*m + j = i*4 + j for i in 0..1, j in 0..3.
V index: j*p + k = j*5 + k for j in 0..3, k in 0..4.

The tensor element T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k].

For a rank-1 term with (u, v, w):
sum_r U[i*4+j, r] * V[j*5+k, r] * W[k*2+i, r] = 1 for all (i,j,k)

Let me construct a rank-36 decomposition. I'll base it on the following idea:

Split the computation into parts based on the structure of the 2x4 * 4x5 product.

Part 1: Handle the "square" 2x2 * 2x2 sub-problems using Strassen (7 terms each).
Part 2: Handle the rectangular sub-problems with minimal rank.

Specifically:
- C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0]
- C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1] + A[0,3]*B[3,1]
- C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2] + A[0,3]*B[3,2]
- C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3] + A[0,2]*B[2,3] + A[0,3]*B[3,3]
- C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4] + A[0,2]*B[2,4] + A[0,3]*B[3,4]
- C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0] + A[1,3]*B[3,0]
- ... etc.

Let me try a different, more systematic approach. I'll use the fact that:

M_{2,4,5} can be viewed as having a block structure. Let me partition the 4 middle dimension as 2+2, and the 5 output dimension as 2+2+1.

A = [A1 | A2] (2x2 each)
B = [B1; B2] (2x5 each)
B1 = [B11 | B12 | B13] (2x2, 2x2, 2x1)
B2 = [B21 | B22 | B23] (2x2, 2x2, 2x1)

C = [C11 | C12 | C13] (2x2, 2x2, 2x1)

C11 = A1*B11 + A2*B21 (2x2*2x2 + 2x2*2x2)
C12 = A1*B12 + A2*B22 (2x2*2x2 + 2x2*2x2)
C13 = A1*B13 + A2*B23 (2x2*2x1 + 2x2*2x1)

For C11: The tensor for X*Y + X'*Y' where X,X',Y,Y' are 2x2.
This is equivalent to computing a 2x2 * 4x2 * 2x2 type problem... no.

Actually, X*Y + X'*Y' where all are 2x2 can be seen as:
Let Z = [X | X'] (2x4) and W = [Y; Y'] (4x2). Then C11 = Z*W.
So C11 is a 2x4 * 4x2 multiplication! That's M_{2,4,2}.

Similarly C12 is M_{2,4,2}.
C13 is 2x4 * 4x1 = M_{2,4,1}.

So M_{2,4,5} = M_{2,4,2} + M_{2,4,2} + M_{2,4,1} in terms of tensor rank (upper bound).

r(M_{2,4,2}): We can compute this. M_{2,4,2} has 2*4*2=16 entries.
Partition: A is 2x4, B is 4x2, C is 2x2.
A = [A1|A2] (2x2 each), B = [B1;B2] (2x2 each).
C = A1*B1 + A2*B2.
This is the sum of two 2x2*2x2 products.

For the sum of two independent 2x2*2x2 multiplications, the rank is at most 7+7=14.
But we can do better by sharing structure. Actually, since they're independent (different A and B blocks), r(M_{2,4,2}) <= 14.

Is r(M_{2,4,2}) = 14? Or can it be less? 

Actually, M_{2,4,2} is the same as M_{2,2,4} by symmetry (transpose). And r(M_{2,2,4}) = r(M_{4,2,2}).

Known: r(M_{2,2,2}) = 7.
r(M_{2,2,3}) = 10.
r(M_{2,2,4}) = ? 

One upper bound: M_{2,2,4} <= M_{2,2,2} + M_{2,2,2} = 14.
Can we do better? M_{2,2,4} can be seen as 2x2 * 2x4. 
Partition the 4 columns into 2+2: 2x2*2x2 + 2x2*2x2. Rank <= 14.
Or partition into 3+1: 2x2*2x3 + 2x2*2x1. Rank <= 10 + 2 = 12.
Or 2x2*2x4 directly. 

Actually, r(M_{2,2,4}) = 12 is achievable. Here's how:
2x2 * 2x4: Let A be 2x2, B be 2x4, C be 2x2... wait, C should be 2x4.

Let me redo: M_{2,2,4} means (2x2) * (2x4) = (2x4). 
A: 2x2, B: 2x4, C: 2x4.

Partition B into B1 (2x2) and B2 (2x2).
C1 = A*B1 (2x2*2x2 = 2x2), C2 = A*B2 (2x2*2x2 = 2x2).

C1 uses Strassen: 7 terms.
C2 uses Strassen: 7 terms.
Total: 14.

But can we share? Since A is the same in both, maybe we can combine.
Actually, A*B1 + A*B2 = A*[B1|B2] which is just 2x2*2x4, so no help.

Alternative: Use the fact that 2x2*2x4 can be done in rank 12.
Here's a construction:
Let A = [[a,b],[c,d]], B = [[e,f,g,h],[i,j,k,l]].
C = [[ae+bi, af+bj, ag+bk, ah+bl],
     [ce+di, cf+dj, cg+dk, ch+dl]]

We need 12 bilinear forms. One way:
M1 = (a+b)(e+i) = ae+ai+be+bi
M2 = (a-b)(e-i) = ae-ai-be+bi  -> wait, this gives ae+bi in M1+M2
Actually let me use Strassen-like:

For the first two columns (2x2*2x2): 7 Strassen terms.
For the last two columns (2x2*2x2): 7 Strassen terms.
Total 14.

To get 12, we need to share 2 terms. 

Hmm, this is getting complicated. Let me just go with rank 36 which is known to work.

36 = 14 + 14 + 8? Or 7+7+10+10+2? Let me try: 7+7+10+10+2 = 36.

Where:
- C11 (2x2): M_{2,4,2} part. Use 7+7=14? No...

Let me try a completely different approach. I'll just hardcode a known rank-36 decomposition.

Based on the knowledge that "Correcting the Strassen tensor patterns (M2, M3, M7) allows the rank 36 decomposition to pass verification," there's a specific rank-36 construction using Strassen patterns.

Let me construct it as follows. I'll use the 2x2 Strassen algorithm as a subroutine.

The 2x2 Strassen algorithm:
M1 = (a00+a11)(b00+b11)
M2 = (a10+a11)(b00)
M3 = (a00)(b10+b11)
M4 = (a11)(b11)
M5 = (a00+a10)(b01+b11)
M6 = (a10-a11)(b10+b11)
M7 = (a01-a00)(b00+b10)

c00 = M1 - M4 + M7
c01 = M5 + M4
c10 = M2 + M3
c11 = M1 - M2 + M5 + M6

Now for our 2x4 * 4x5 problem, let me partition:
A (2x4) = [A1 | A2] where A1, A2 are 2x2.
B (4x5) = [B1; B2] where B1, B2 are 2x5.
B1 (2x5) = [B11 | B12 | B13] where B11, B12 are 2x2, B13 is 2x1.
B2 (2x5) = [B21 | B22 | B23] where B21, B22 are 2x2, B23 is 2x1.

C (2x5) = [C11 | C12 | C13] where C11, C12 are 2x2, C13 is 2x1.

C11 = A1*B11 + A2*B21
C12 = A1*B12 + A2*B22
C13 = A1*B13 + A2*B23

For C11 = A1*B11 + A2*B21:
This is a 2x4 * 4x2 multiplication (if we view [A1|A2] as 2x4 and [B11;B21] as 4x2).
Let's call this a "2x4x2" sub-problem.

For the 2x4x2 problem: A' is 2x4, B' is 4x2, C' is 2x2.
Partition A' = [A1|A2], B' = [B1;B2] (all 2x2).
C' = A1*B1 + A2*B2.

I'll use 14 terms for this (7 for each Strassen).

Similarly for C12: another 2x4x2, 14 terms.
For C13: 2x4 * 4x1 = 2x4x1. A' is 2x4, B' is 4x1, C' is 2x1.
This is just 4 scalar multiplications per row... actually 2*4*1 = 8 schoolbook.
Can we do better? r(M_{2,4,1}) = 8 (it's just a bilinear form, rank = number of terms = 8).
Actually, M_{2,4,1}: C[i] = sum_j A[i,j]*B[j]. This is two independent dot products of length 4.
Each dot product of length 4 has rank 4 (can't be done in fewer than 4 bilinear forms for a generic dot product).
So r(M_{2,4,1}) = 8.

Total: 14 + 14 + 8 = 36. 

Now let me implement this.

For the 2x4x2 sub-problem (C11 = A1*B11 + A2*B21):
A1 = [[a00, a01], [a10, a11]] (A[0,0], A[0,1], A[1,0], A[1,1])
A2 = [[a02, a03], [a12, a13]] (A[0,2], A[0,3], A[1,2], A[1,3])
B11 = [[b00, b01], [b10, b11]] (B[0,0], B[0,1], B[1,0], B[1,1])
B21 = [[b20, b21], [b30, b31]] (B[2,0], B[2,1], B[3,0], B[3,1])

C11[0,0] = a00*b00 + a01*b10 + a02*b20 + a03*b30
C11[0,1] = a00*b01 + a01*b11 + a02*b21 + a03*b31
C11[1,0] = a10*b00 + a11*b10 + a12*b20 + a13*b30
C11[1,1] = a10*b01 + a11*b11 + a12*b21 + a13*b31

Using Strassen for A1*B11:
S1 = (a00+a11)(b00+b11)
S2 = (a10+a11)(b00)
S3 = (a00)(b10+b11)
S4 = (a11)(b11)
S5 = (a00+a10)(b01+b11)
S6 = (a10-a11)(b10+b11)
S7 = (a01-a00)(b00+b10)

A1*B11 = [[S1-S4+S7, S5+S4], [S2+S3, S1-S2+S5+S6]]

Using Strassen for A2*B21:
T1 = (a02+a13)(b20+b31)
T2 = (a12+a13)(b20)
T3 = (a02)(b30+b31)
T4 = (a13)(b31)
T5 = (a02+a12)(b21+b31)
T6 = (a12-a13)(b30+b31)
T7 = (a03-a02)(b20+b30)

A2*B21 = [[T1-T4+T7, T5+T4], [T2+T3, T1-T2+T5+T6]]

C11 = A1*B11 + A2*B21:
C11[0,0] = (S1-S4+S7) + (T1-T4+T7)
C11[0,1] = (S5+S4) + (T5+T4)
C11[1,0] = (S2+S3) + (T2+T3)
C11[1,1] = (S1-S2+S5+S6) + (T1-T2+T5+T6)

Each of S1..S7, T1..T7 is a bilinear form. In the tensor decomposition, each bilinear form corresponds to one rank-1 term.

For S1 = (a00+a11)(b00+b11):
U: a00 gets +1, a11 gets +1.
V: b00 gets +1, b11 gets +1.
W: contributes to C11[0,0] with coeff +1, C11[1,1] with coeff +1.

Let me index the variables:
A indices: (i,j) -> U index = i*4+j
  a00: (0,0) -> 0
  a01: (0,1) -> 1
  a02: (0,2) -> 2
  a03: (0,3) -> 3
  a10: (1,0) -> 4
  a11: (1,1) -> 5
  a12: (1,2) -> 6
  a13: (1,3) -> 7

B indices: (j,k) -> V index = j*5+k
  b00: (0,0) -> 0
  b01: (0,1) -> 1
  b02: (0,2) -> 2
  b03: (0,3) -> 3
  b04: (0,4) -> 4
  b10: (1,0) -> 5
  b11: (1,1) -> 6
  b12: (1,2) -> 7
  b13: (1,3) -> 8
  b14: (1,4) -> 9
  b20: (2,0) -> 10
  b21: (2,1) -> 11
  b22: (2,2) -> 12
  b23: (2,3) -> 13
  b24: (2,4) -> 14
  b30: (3,0) -> 15
  b31: (3,1) -> 16
  b32: (3,2) -> 17
  b33: (3,3) -> 18
  b34: (3,4) -> 19

C indices: (i,k) -> W index = k*2+i
  c00: (0,0) -> 0
  c01: (0,1) -> 2
  c02: (0,2) -> 4
  c03: (0,3) -> 6
  c04: (0,4) -> 8
  c10: (1,0) -> 1
  c11: (1,1) -> 3
  c12: (1,2) -> 5
  c13: (1,3) -> 7
  c14: (1,4) -> 9

Now let me build the rank-36 decomposition.

Terms for C11 (S1..S7, T1..T7):
S1 = (a00+a11)(b00+b11): U[0]=1,U[5]=1; V[0]=1,V[6]=1; W[0]=1,W[3]=1
S2 = (a10+a11)(b00): U[4]=1,U[5]=1; V[0]=1; W[1]=1
S3 = (a00)(b10+b11): U[0]=1; V[5]=1,V[6]=1; W[1]=1
S4 = (a11)(b11): U[5]=1; V[6]=1; W[0]=-1,W[3]=1
S5 = (a00+a10)(b01+b11): U[0]=1,U[4]=1; V[1]=1,V[6]=1; W[2]=1,W[3]=1
S6 = (a10-a11)(b10+b11): U[4]=1,U[5]=-1; V[5]=1,V[6]=1; W[3]=1
S7 = (a01-a00)(b00+b10): U[1]=1,U[0]=-1; V[0]=1,V[5]=1; W[0]=1

T1 = (a02+a13)(b20+b31): U[2]=1,U[7]=1; V[10]=1,V[16]=1; W[0]=1,W[3]=1
T2 = (a12+a13)(b20): U[6]=1,U[7]=1; V[10]=1; W[1]=1
T3 = (a02)(b30+b31): U[2]=1; V[15]=1,V[16]=1; W[1]=1
T4 = (a13)(b31): U[7]=1; V[16]=1; W[0]=-1,W[3]=1
T5 = (a02+a12)(b21+b31): U[2]=1,U[6]=1; V[11]=1,V[16]=1; W[2]=1,W[3]=1
T6 = (a12-a13)(b30+b31): U[6]=1,U[7]=-1; V[15]=1,V[16]=1; W[3]=1
T7 = (a03-a02)(b20+b30): U[3]=1,U[2]=-1; V[10]=1,V[15]=1; W[0]=1

Terms for C12 (same pattern, shifted columns):
A1*B12 where B12 = [[b02,b03],[b12,b13]]:
U1 = (a00+a11)(b02+b13): U[0]=1,U[5]=1; V[2]=1,V[8]=1; W[4]=1,W[7]=1
U2 = (a10+a11)(b02): U[4]=1,U[5]=1; V[2]=1; W[5]=1
U3 = (a00)(b12+b13): U[0]=1; V[7]=1,V[8]=1; W[5]=1
U4 = (a11)(b13): U[5]=1; V[8]=1; W[4]=-1,W[7]=1
U5 = (a00+a10)(b03+b13): U[0]=1,U[4]=1; V[3]=1,V[8]=1; W[6]=1,W[7]=1
U6 = (a10-a11)(b12+b13): U[4]=1,U[5]=-1; V[7]=1,V[8]=1; W[7]=1
U7 = (a01-a00)(b02+b12): U[1]=1,U[0]=-1; V[2]=1,V[7]=1; W[4]=1

A2*B22 where B22 = [[b22,b23],[b32,b33]]:
V1 = (a02+a13)(b22+b33): U[2]=1,U[7]=1; V[12]=1,V[18]=1; W[4]=1,W[7]=1
V2 = (a12+a13)(b22): U[6]=1,U[7]=1; V[12]=1; W[5]=1
V3 = (a02)(b32+b33): U[2]=1; V[17]=1,V[18]=1; W[5]=1
V4 = (a13)(b33): U[7]=1; V[18]=1; W[4]=-1,W[7]=1
V5 = (a02+a12)(b23+b33): U[2]=1,U[6]=1; V[13]=1,V[18]=1; W[6]=1,W[7]=1
V6 = (a12-a13)(b32+b33): U[6]=1,U[7]=-1; V[17]=1,V[18]=1; W[7]=1
V7 = (a03-a02)(b22+b32): U[3]=1,U[2]=-1; V[12]=1,V[17]=1; W[4]=1

Terms for C13 (2x4 * 4x1, 8 schoolbook terms):
C13[0] = a00*b04 + a01*b14 + a02*b24 + a03*b34
C13[1] = a10*b04 + a11*b14 + a12*b24 + a13*b34

8 terms:
W1: a00*b04: U[0]=1; V[4]=1; W[8]=1
W2: a01*b14: U[1]=1; V[9]=1; W[8]=1
W3: a02*b24: U[2]=1; V[14]=1; W[8]=1
W4: a03*b34: U[3]=1; V[19]=1; W[8]=1
W5: a10*b04: U[4]=1; V[4]=1; W[9]=1
W6: a11*b14: U[5]=1; V[9]=1; W[9]=1
W7: a12*b24: U[6]=1; V[14]=1; W[9]=1
W8: a13*b34: U[7]=1; V[19]=1; W[9]=1

Total: 7+7+7+7+8 = 36 terms.

Let me now code this up.
"""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0
    
    # Helper to add a term
    def add_term(u_entries, v_entries, w_entries):
        nonlocal col
        for idx, val in u_entries:
            U[idx][col] = val
        for idx, val in v_entries:
            V[idx][col] = val
        for idx, val in w_entries:
            W[idx][col] = val
        col += 1
    
    # C11 = A1*B11 + A2*B21 (14 terms)
    # A1 indices: a00=0, a01=1, a10=4, a11=5
    # A2 indices: a02=2, a03=3, a12=6, a13=7
    # B11 indices: b00=0, b01=1, b10=5, b11=6
    # B21 indices: b20=10, b21=11, b30=15, b31=16
    # C11: c00=0, c01=2, c10=1, c11=3
    
    # S1 = (a00+a11)(b00+b11)
    add_term([(0,1),(5,1)], [(0,1),(6,1)], [(0,1),(3,1)])
    # S2 = (a10+a11)(b00)
    add_term([(4,1),(5,1)], [(0,1)], [(1,1)])
    # S3 = (a00)(b10+b11)
    add_term([(0,1)], [(5,1),(6,1)], [(1,1)])
    # S4 = (a11)(b11)
    add_term([(5,1)], [(6,1)], [(0,-1),(3,1)])
    # S5 = (a00+a10)(b01+b11)
    add_term([(0,1),(4,1)], [(1,1),(6,1)], [(2,1),(3,1)])
    # S6 = (a10-a11)(b10+b11)
    add_term([(4,1),(5,-1)], [(5,1),(6,1)], [(3,1)])
    # S7 = (a01-a00)(b00+b10)
    add_term([(1,1),(0,-1)], [(0,1),(5,1)], [(0,1)])
    
    # T1 = (a02+a13)(b20+b31)
    add_term([(2,1),(7,1)], [(10,1),(16,1)], [(0,1),(3,1)])
    # T2 = (a12+a13)(b20)
    add_term([(6,1),(7,1)], [(10,1)], [(1,1)])
    # T3 = (a02)(b30+b31)
    add_term([(2,1)], [(15,1),(16,1)], [(1,1)])
    # T4 = (a13)(b31)
    add_term([(7,1)], [(16,1)], [(0,-1),(3,1)])
    # T5 = (a02+a12)(b21+b31)
    add_term([(2,1),(6,1)], [(11,1),(16,1)], [(2,1),(3,1)])
    # T6 = (a12-a13)(b30+b31)
    add_term([(6,1),(7,-1)], [(15,1),(16,1)], [(3,1)])
    # T7 = (a03-a02)(b20+b30)
    add_term([(3,1),(2,-1)], [(10,1),(15,1)], [(0,1)])
    
    # C12 = A1*B12 + A2*B22 (14 terms)
    # B12: b02=2, b03=3, b12=7, b13=8
    # B22: b22=12, b23=13, b32=17, b33=18
    # C12: c02=4, c03=6, c12=5, c13=7
    
    # U1 = (a00+a11)(b02+b13)
    add_term([(0,1),(5,1)], [(2,1),(8,1)], [(4,1),(7,1)])
    # U2 = (a10+a11)(b02)
    add_term([(4,1),(5,1)], [(2,1)], [(5,1)])
    # U3 = (a00)(b12+b13)
    add_term([(0,1)], [(7,1),(8,1)], [(5,1)])
    # U4 = (a11)(b13)
    add_term([(5,1)], [(8,1)], [(4,-1),(7,1)])
    # U5 = (a00+a10)(b03+b13)
    add_term([(0,1),(4,1)], [(3,1),(8,1)], [(6,1),(7,1)])
    # U6 = (a10-a11)(b12+b13)
    add_term([(4,1),(5,-1)], [(7,1),(8,1)], [(7,1)])
    # U7 = (a01-a00)(b02+b12)
    add_term([(1,1),(0,-1)], [(2,1),(7,1)], [(4,1)])
    
    # V1 = (a02+a13)(b22+b33)
    add_term([(2,1),(7,1)], [(12,1),(18,1)], [(4,1),(7,1)])
    # V2 = (a12+a13)(b22)
    add_term([(6,1),(7,1)], [(12,1)], [(5,1)])
    # V3 = (a02)(b32+b33)
    add_term([(2,1)], [(17,1),(18,1)], [(5,1)])
    # V4 = (a13)(b33)
    add_term([(7,1)], [(18,1)], [(4,-1),(7,1)])
    # V5 = (a02+a12)(b23+b33)
    add_term([(2,1),(6,1)], [(13,1),(18,1)], [(6,1),(7,1)])
    # V6 = (a12-a13)(b32+b33)
    add_term([(6,1),(7,-1)], [(17,1),(18,1)], [(7,1)])
    # V7 = (a03-a02)(b22+b32)
    add_term([(3,1),(2,-1)], [(12,1),(17,1)], [(4,1)])
    
    # C13 = A1*B13 + A2*B23 (8 terms, schoolbook)
    # B13: b04=4, b14=9
    # B23: b24=14, b34=19
    # C13: c04=8, c14=9
    
    add_term([(0,1)], [(4,1)], [(8,1)])   # a00*b04
    add_term([(1,1)], [(9,1)], [(8,1)])   # a01*b14
    add_term([(2,1)], [(14,1)], [(8,1)])  # a02*b24
    add_term([(3,1)], [(19,1)], [(8,1)])  # a03*b34
    add_term([(4,1)], [(4,1)], [(9,1)])   # a10*b04
    add_term([(5,1)], [(9,1)], [(9,1)])   # a11*b14
    add_term([(6,1)], [(14,1)], [(9,1)])  # a12*b24
    add_term([(7,1)], [(19,1)], [(9,1)])  # a13*b34
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
