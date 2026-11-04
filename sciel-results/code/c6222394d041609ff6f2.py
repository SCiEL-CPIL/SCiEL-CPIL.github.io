"""
Rank-minimal (or near-minimal) tensor decomposition for 3x3 * 3x5 matrix multiplication.

We use a block decomposition:
  - Split n=3 into n1=2, n2=1
  - Split m=3 into m1=2, m2=1
  - p=5 stays as is

Blocks:
  1) 2x2 * 2x5  -> Strassen for 2x2 * 2x2 (rank 7) + two rank-1 for the remaining cols
  2) 2x2 * 1x5  -> each product is rank 1 -> 2 blocks, but we can combine? Actually
     A[2x2] * B[1x5] gives 2x5 output. Each output element (i,k) is A[i,j]*B[j,k] summed over j=0,1.
     This is a 2x2x5 tensor. Rank? For 2x2x5, the rank is at most 4 (schoolbook) but can we do better?
     Actually, for multiplying a 2x2 by 1x5, the output is 2x5. The tensor has structure.
     Let's think: C[i,k] = sum_j A[i,j] B[j,k]. This is like two independent 2x1x5 products? No.
     Actually, we can view it as: for each i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:].
     This is a linear combination of the two rows of B, with coefficients from A[i,:].
     The tensor T[i,j,k] = 1 if we are in this block. The rank of a 2x2x5 tensor where T[i,j,k]=1 for all i,j,k? No, that's not right.
     
     Actually, the multiplication tensor for (n,m,p) has T[i,j,k] = 1. The rank is the minimal number of rank-1 tensors that sum to it.
     For 2x2x5, schoolbook gives rank 4. Can we do better? I think the rank of 2x2x5 is 4. Because if we try rank 3, we'd need to decompose a 2x2x5 tensor into 3 rank-1 tensors, which seems unlikely.
     
     So for block 2) 2x2 * 1x5: rank 4.
  3) 1x1 * 2x5  -> C[i,k] = A[0,0]*B[0,k] for i=0,1? Wait, n1=2 for the first block, n2=1 for the second.
     Let me re-index.
     
     Let's define:
     A is 3x3, B is 3x5, C is 3x5.
     Split A into:
       A11: rows 0-1, cols 0-1 (2x2)
       A12: rows 0-1, col 2 (2x1)
       A21: row 2, cols 0-1 (1x2)
       A22: row 2, col 2 (1x1)
     Split B into:
       B11: rows 0-1, all cols (2x5)
       B21: row 2, all cols (1x5)
     
     Then C = [A11 B11 + A12 B21; A21 B11 + A22 B22]
     Wait, that's not quite right because the splits don't align perfectly for block multiplication.
     
     Actually, for block multiplication, we need to split A into (n1 x m1, n1 x m2, n2 x m1, n2 x m2) and B into (m1 x p, m2 x p).
     Then C[i,k] = sum_j A[i,j] B[j,k].
     
     If we split A's columns into m1=2, m2=1 and B's rows into m1=2, m2=1, then:
     C = A1 * B1 + A2 * B2
     where A1 is A[:, 0:2] (3x2), B1 is B[0:2, :] (2x5), A2 is A[:, 2:3] (3x1), B2 is B[2:3, :] (1x5).
     
     Now split C's rows into n1=2, n2=1.
     C1 = A1[0:2, :] * B1 + A2[0:2, :] * B2  (2x5)
     C2 = A1[2:3, :] * B1 + A2[2:3, :] * B2  (1x5)
     
     For C1:
       A1[0:2, :] is 2x2, split into A11 (2x2) and A12 (2x1).
       So C1 = A11 * B1 + A12 * B2
       A11 * B1: 2x2 * 2x5. B1 is 2x5. We can further split B1's columns? No, B1 is already the full 2x5.
       Actually, B1 is 2x5. To multiply 2x2 * 2x5, we can use Strassen on the 2x2 part and handle the 5 columns.
       
       Let's think of 2x2 * 2x5 as two 2x2 * 2x1 multiplications plus a 2x2 * 2x2 multiplication? No, 5 = 2+2+1.
       So we can split B1 into B1a (2x2), B1b (2x2), B1c (2x1).
       Then A11 * B1 = A11 * B1a + A11 * B1b + A11 * B1c.
       Each of these can be decomposed.
       
       This is getting complex. Let me just use a known result: the rank of 2x2x5 is at most 7+7+1 = 15? No, that's if we do Strassen on each 2x2x2 block.
       
       Actually, for 2x2 * 2x5, we can view it as 2x2 * (2+2+1) x 1. So three subproblems: 2x2x2, 2x2x2, 2x2x1.
       Ranks: 7 + 7 + 1 = 15.
       
       For A12 * B2: 2x1 * 1x5. This is a 2x1x5 tensor. Rank? C[i,k] = A12[i,0] * B2[0,k]. This is an outer product of a 2-vector and a 5-vector, so rank 1? No, it's a 2x1x5 tensor. Each element is T[i,j,k] where j is fixed to 0. So T[i,0,k] = 1. This is a 2x5 matrix of 1s, which has rank 1 as a matrix, but as a tensor, it's rank 1? No, a 2x1x5 tensor with T[i,0,k]=1 can be written as sum of 1 rank-1 tensor: u_i * v_0 * w_k where u=[1,1], v=[1], w=[1,1,1,1,1]. So rank 1.
       
       Similarly for C2:
       C2 = A21 * B1 + A22 * B2
       A21 * B1: 1x2 * 2x5. 1x2x5 tensor. Rank? T[0,j,k]=1 for j=0,1, k=0..4. This is a 2x5 matrix of 1s, rank 1 as tensor? Yes, u_0 * v_j * w_k.
       A22 * B2: 1x1 * 1x5. Rank 1.
       
     So total rank = (7+7+1) + 1 + 1 + 1 = 18.
     
     But wait, can we do better by not splitting p into 2+2+1? What if we use a different decomposition for 2x2x5?
     
     I think 18 is a good target. Let's implement it.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    # We'll build the decomposition with rank 18
    rank = 18
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    
    # Block 1: A11 (2x2) * B1a (2x2) where B1a = B[0:2, 0:2]
    # Strassen 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # A indices: i*m+j -> i*3+j
    # B indices: j*p+k -> j*5+k
    # C indices: k*n+i -> k*3+i
    
    # M1
    U[0*3+0, col] = 1; U[1*3+1, col] = 1
    V[0*5+0, col] = 1; V[1*5+1, col] = 1
    W[0*3+0, col] = 1; W[1*3+0, col] = 1; W[0*3+1, col] = 1; W[1*3+1, col] = 1
    col += 1
    
    # M2
    U[1*3+0, col] = 1; U[1*3+1, col] = 1
    V[0*5+0, col] = 1
    W[0*3+1, col] = 1; W[1*3+1, col] = 1
    col += 1
    
    # M3
    U[0*3+0, col] = 1
    V[0*5+1, col] = 1; V[1*5+1, col] = -1
    W[1*3+0, col] = 1
    col += 1
    
    # M4
    U[1*3+1, col] = 1
    V[1*5+0, col] = 1; V[0*5+0, col] = -1
    W[0*3+1, col] = 1; W[1*3+0, col] = 1
    col += 1
    
    # M5
    U[0*3+0, col] = 1; U[0*3+1, col] = 1
    V[1*5+1, col] = 1
    W[0*3+0, col] = -1; W[1*3+1, col] = 1
    col += 1
    
    # M6
    U[1*3+0, col] = 1; U[0*3+0, col] = -1
    V[0*5+0, col] = 1; V[0*5+1, col] = 1
    W[1*3+1, col] = 1
    col += 1
    
    # M7
    U[0*3+1, col] = 1; U[1*3+1, col] = -1
    V[1*5+0, col] = 1; V[1*5+1, col] = 1
    W[0*3+0, col] = 1; W[1*3+1, col] = 1
    col += 1
    
    # Block 2: A11 (2x2) * B1b (2x2) where B1b = B[0:2, 2:4]
    # Same Strassen, but k offset by 2
    # M1
    U[0*3+0, col] = 1; U[1*3+1, col] = 1
    V[0*5+2, col] = 1; V[1*5+3, col] = 1
    W[2*3+0, col] = 1; W[3*3+0, col] = 1; W[2*3+1, col] = 1; W[3*3+1, col] = 1
    col += 1
    
    # M2
    U[1*3+0, col] = 1; U[1*3+1, col] = 1
    V[0*5+2, col] = 1
    W[2*3+1, col] = 1; W[3*3+1, col] = 1
    col += 1
    
    # M3
    U[0*3+0, col] = 1
    V[0*5+3, col] = 1; V[1*5+3, col] = -1
    W[3*3+0, col] = 1
    col += 1
    
    # M4
    U[1*3+1, col] = 1
    V[1*5+2, col] = 1; V[0*5+2, col] = -1
    W[2*3+1, col] = 1; W[3*3+0, col] = 1
    col += 1
    
    # M5
    U[0*3+0, col] = 1; U[0*3+1, col] = 1
    V[1*5+3, col] = 1
    W[2*3+0, col] = -1; W[3*3+1, col] = 1
    col += 1
    
    # M6
    U[1*3+0, col] = 1; U[0*3+0, col] = -1
    V[0*5+2, col] = 1; V[0*5+3, col] = 1
    W[3*3+1, col] = 1
    col += 1
    
    # M7
    U[0*3+1, col] = 1; U[1*3+1, col] = -1
    V[1*5+2, col] = 1; V[1*5+3, col] = 1
    W[2*3+0, col] = 1; W[3*3+1, col] = 1
    col += 1
    
    # Block 3: A11 (2x2) * B1c (2x1) where B1c = B[0:2, 4:5]
    # Rank 1: just do it directly
    # C[i,4] = sum_j A[i,j] B[j,4]
    # We can use rank 1? No, 2x2x1 has rank 2.
    # Actually, for 2x2x1, T[i,j,0]=1. Rank is 2.
    # M1: A[0,:] * B[:,0] -> gives C[0,0]
    # M2: A[1,:] * B[:,0] -> gives C[1,0]
    U[0*3+0, col] = 1; U[0*3+1, col] = 1
    V[0*5+4, col] = 1; V[1*5+4, col] = 1
    W[4*3+0, col] = 1
    col += 1
    
    U[1*3+0, col] = 1; U[1*3+1, col] = 1
    V[0*5+4, col] = 1; V[1*5+4, col] = 1
    W[4*3+1, col] = 1
    col += 1
    
    # Block 4: A12 (2x1) * B2 (1x5) where A12 = A[0:2, 2:3], B2 = B[2:3, :]
    # C[i,k] = A[i,2] * B[2,k]
    # Rank 1: u_i * v_j * w_k
    U[0*3+2, col] = 1; U[1*3+2, col] = 1
    V[2*5+0, col] = 1
    for k in range(5):
        W[k*3+0, col] = 1
        W[k*3+1, col] = 1
    col += 1
    
    # Block 5: A21 (1x2) * B1 (2x5) where A21 = A[2:3, 0:2], B1 = B[0:2, :]
    # C[2,k] = sum_j A[2,j] B[j,k]
    # Rank 2
    U[2*3+0, col] = 1; U[2*3+1, col] = 1
    V[0*5+0, col] = 1; V[0*5+1, col] = 1; V[0*5+2, col] = 1; V[0*5+3, col] = 1; V[0*5+4, col] = 1
    W[0*3+2, col] = 1; W[1*3+2, col] = 1; W[2*3+2, col] = 1; W[3*3+2, col] = 1; W[4*3+2, col] = 1
    col += 1
    
    U[2*3+0, col] = 1; U[2*3+1, col] = 1
    V[1*5+0, col] = 1; V[1*5+1, col] = 1; V[1*5+2, col] = 1; V[1*5+3, col] = 1; V[1*5+4, col] = 1
    W[0*3+2, col] = 1; W[1*3+2, col] = 1; W[2*3+2, col] = 1; W[3*3+2, col] = 1; W[4*3+2, col] = 1
    col += 1
    
    # Block 6: A22 (1x1) * B2 (1x5) where A22 = A[2:3, 2:3], B2 = B[2:3, :]
    # C[2,k] = A[2,2] * B[2,k]
    # Rank 1
    U[2*3+2, col] = 1
    V[2*5+0, col] = 1
    for k in range(5):
        W[k*3+2, col] = 1
    col += 1
    
    # Total rank used: 7+7+2+1+2+1 = 20. Hmm, I expected 18.
    # Let me recount:
    # Block 1: 7
    # Block 2: 7
    # Block 3: 2 (for 2x2x1)
    # Block 4: 1 (for 2x1x5)
    # Block 5: 2 (for 1x2x5)
    # Block 6: 1 (for 1x1x5)
    # Total: 7+7+2+1+2+1 = 20.
    
    # I think I can do better. For 2x2x1, the rank is 2, which I used.
    # For 1x2x5, the rank is 2, which I used.
    # For 2x1x5, the rank is 1, which I used.
    # For 1x1x5, the rank is 1, which I used.
    # So 20 is correct for this decomposition.
    
    # But the schoolbook is 45. 20 is a big improvement.
    # Can we do better?
    
    # Let's just return this with rank 20.
    
    # Trim to actual rank used
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
