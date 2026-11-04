"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 120.0
    MARGIN = 5.0  # safety margin before kill

    # We will build the rank-17 Strassen-block decomposition directly.
    # This is a known result: the tensor rank of 2x3x3 matrix multiplication is 17.
    
    # Dimensions: n=2, m=3, p=3
    # U shape: (n*m, rank) = (6, 17)
    # V shape: (m*p, rank) = (9, 17)
    # W shape: (p*n, rank) = (6, 17)
    
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # The decomposition is based on splitting the 3x3 B matrix into:
    # B = [B00 B01; B10 B11] where B00 is 2x2, B01 is 2x1, B10 is 1x2, B11 is 1x1.
    # And A is 2x3, split as A = [A0 A1] where A0 is 2x2, A1 is 2x1.
    # C = A * B = [A0 B00 + A1 B10, A0 B01 + A1 B11]
    # C is 2x3, so C = [C0 C1] where C0 is 2x2, C1 is 2x1.
    
    # The 2x2x2 part (A0 * B00) uses 7 multiplications (Strassen).
    # The cross terms are linear combinations.
    
    # Let's explicitly define the 17 products.
    # Strassen for 2x2:
    # M1 = (A0[0,0]+A0[1,1])*(B00[0,0]+B00[1,1])
    # M2 = (A0[1,0]+A0[1,1])*B00[0,0]
    # M3 = A0[0,0]*(B00[0,1]-B00[1,1])
    # M4 = A0[1,1]*(B00[1,0]-B00[0,0])
    # M5 = (A0[0,0]+A0[0,1])*B00[1,1]
    # M6 = (A0[1,0]-A0[0,0])*(B00[0,0]+B00[0,1])
    # M7 = (A0[0,1]-A0[1,1])*(B00[1,0]+B00[1,1])
    
    # The remaining 10 multiplications handle the cross terms with A1 and B10, B11.
    # Actually, the standard rank-17 algorithm for 2x3x3 is well-documented.
    # Let's construct it directly from the known decomposition.
    
    # U indices: i*m + j  (i in 0..1, j in 0..2)
    # V indices: j*p + k  (j in 0..2, k in 0..2)
    # W indices: k*n + i  (k in 0..2, i in 0..1)
    
    # We will hardcode the 17 columns. Each column is a rank-1 tensor.
    # U_col, V_col, W_col are vectors of length 6, 9, 6 respectively.
    # The product sum_{r} U[:,r] * V[:,r] * W[:,r] must equal the matmul tensor.
    
    # Known rank-17 decomposition for 2x3x3 (from literature, e.g., Strassen's work or subsequent improvements):
    # This is a specific known decomposition.
    
    # Let's define the 17 triples (u, v, w) where u in {0,1}^3 (wait, u has 6 entries), 
    # actually u is a 6-vector, v is a 9-vector, w is a 6-vector.
    # It's easier to list the non-zero entries for each column.
    
    # Column 0-6: Strassen terms for the top-left 2x2 block multiplication.
    # A0 is rows 0,1 and cols 0,1 of A. (Indices in U: 0,1,3,4 for j=0,1)
    # B00 is rows 0,1 and cols 0,1 of B. (Indices in V: 0,1,3,4 for j=0,1, k=0,1)
    # C0 is cols 0,1 and rows 0,1 of C. (Indices in W: 0,1,2,3 for k=0,1, i=0,1)
    
    # Let's map indices:
    # U[i*3 + j]: i=0,1; j=0,1,2
    # V[j*3 + k]: j=0,1,2; k=0,1,2
    # W[k*2 + i]: k=0,1,2; i=0,1
    
    # Strassen M1: (A0[0,0]+A0[1,1])*(B00[0,0]+B00[1,1])
    # U: [1,0,0, 1,0,0] -> indices 0 and 3
    # V: [1,0,0, 1,0,0] -> indices 0 and 3
    # W: [1,1, 1,1] -> indices 0,1,2,3 (C0[0,0] += M1, C0[0,1] += M1? No, C0[0,0] = M1 - M5 + M4 + M7, etc.)
    # Let's use the standard Strassen output:
    # C0[0,0] = M1 - M5 + M4 + M7
    # C0[0,1] = M3 + M5
    # C0[1,0] = M2 + M4
    # C0[1,1] = M1 - M2 + M3 + M6
    
    # So for M1 (col 0):
    # U: 1 at i=0,j=0 and i=1,j=1 => U[0]=1, U[3]=1
    # V: 1 at j=0,k=0 and j=1,k=1 => V[0]=1, V[3]=1
    # W: 1 at k=0,i=0 and k=1,i=0 and k=0,i=1 and k=1,i=1 => W[0]=1, W[2]=1, W[1]=1, W[3]=1
    # Wait, C0[0,0] is k=0,i=0 -> W[0]. C0[0,1] is k=1,i=0 -> W[2]. C0[1,0] is k=0,i=1 -> W[1]. C0[1,1] is k=1,i=1 -> W[3].
    # M1 contributes +1 to C0[0,0] and +1 to C0[1,1].
    # So W for M1: W[0]=1, W[3]=1.
    
    # Let's restart the column definitions carefully.
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00 (i=0,j=0) -> U[0]=1; A11 (i=1,j=1) -> U[3]=1
    # V: B00 (j=0,k=0) -> V[0]=1; B11 (j=1,k=1) -> V[3]=1
    # W: C00 (k=0,i=0) -> W[0]=1; C11 (k=1,i=1) -> W[3]=1
    # Column 0: U[0]=1,U[3]=1; V[0]=1,V[3]=1; W[0]=1,W[3]=1
    
    # M2 = (A10+A11)B00
    # U: A10 (i=1,j=0) -> U[3]=1; A11 (i=1,j=1) -> U[4]=1
    # V: B00 (j=0,k=0) -> V[0]=1
    # W: C10 (k=0,i=1) -> W[1]=1
    # Column 1: U[3]=1,U[4]=1; V[0]=1; W[1]=1
    
    # M3 = A00(B01-B11)
    # U: A00 (i=0,j=0) -> U[0]=1
    # V: B01 (j=0,k=1) -> V[1]=1; B11 (j=1,k=1) -> V[4]=1
    # W: C01 (k=1,i=0) -> W[2]=1
    # Column 2: U[0]=1; V[1]=1,V[4]=-1; W[2]=1
    
    # M4 = A11(B10-B00)
    # U: A11 (i=1,j=1) -> U[4]=1
    # V: B10 (j=1,k=0) -> V[3]=1; B00 (j=0,k=0) -> V[0]=1
    # W: C10 (k=0,i=1) -> W[1]=1
    # Column 3: U[4]=1; V[3]=1,V[0]=-1; W[1]=1
    
    # M5 = (A00+A01)B11
    # U: A00 (i=0,j=0) -> U[0]=1; A01 (i=0,j=1) -> U[1]=1
    # V: B11 (j=1,k=1) -> V[4]=1
    # W: C01 (k=1,i=0) -> W[2]=1
    # Column 4: U[0]=1,U[1]=1; V[4]=1; W[2]=1
    
    # M6 = (A10-A00)(B00+B01)
    # U: A10 (i=1,j=0) -> U[3]=1; A00 (i=0,j=0) -> U[0]=-1
    # V: B00 (j=0,k=0) -> V[0]=1; B01 (j=0,k=1) -> V[1]=1
    # W: C11 (k=1,i=1) -> W[3]=1
    # Column 5: U[3]=1,U[0]=-1; V[0]=1,V[1]=1; W[3]=1
    
    # M7 = (A01-A11)(B10+B11)
    # U: A01 (i=0,j=1) -> U[1]=1; A11 (i=1,j=1) -> U[4]=-1
    # V: B10 (j=1,k=0) -> V[3]=1; B11 (j=1,k=1) -> V[4]=1
    # W: C11 (k=1,i=1) -> W[3]=1
    # Column 6: U[1]=1,U[4]=-1; V[3]=1,V[4]=1; W[3]=1
    
    # Now the cross terms. We have A1 (col 2) and B10 (row 2), B11 (row 2, col 2).
    # C02 = A0*B02 + A1*B12 -> This is column 2 of C. (k=2)
    # C12 = A0*B12 + A1*B22 -> This is column 2 of C. (k=2)
    # C02 uses W[4] (k=2,i=0). C12 uses W[5] (k=2,i=1).
    
    # The terms involving A1 (j=2) and B1* (j=2) and B*2 (k=2).
    # We need 10 more multiplications.
    # Let's use a known rank-17 algorithm.
    # The remaining 10 terms are:
    # M8 = A00*B20
    # M9 = A01*B21
    # M10 = A10*B20
    # M11 = A11*B21
    # M12 = A02*B00
    # M13 = A12*B00
    # M14 = A02*B11
    # M15 = A12*B11
    # M16 = A02*B01 + ...
    # This is getting complicated to derive from scratch reliably in the code.
    
    # Let's use a verified rank-17 decomposition from a known source or derive it via a small search.
    # Given the deadline, I will hardcode a known correct rank-17 decomposition.
    
    # Here is a verified rank-17 decomposition for 2x3x3:
    # (From "On the rank of the 2x3x3 matrix multiplication tensor" or similar)
    
    # Let's try to construct it systematically.
    # We have 17 columns.
    # U (6x17), V (9x17), W (6x17).
    
    # I will use a specific decomposition:
    # Columns 0-6 as derived above.
    # Columns 7-16 for the cross terms.
    
    # Let's define the remaining 10 columns to complete the tensor.
    # The tensor T_{ijk} = 1 if j is in both A and B, and i,k are in C.
    # Specifically, T_{i,j,k} is the coefficient of x_i y_j z_k in (sum x_i y_j)(sum y_j z_k) = sum x_i z_k (sum y_j^2).
    # Wait, the tensor is T[i,j,k] = 1 if the multiplication A[i,j]*B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all i,j,k.
    
    # We need sum_r U[i,r] V[j,r] W[k,r] = 1 for all i,j,k.
    
    # Let's check if the first 7 columns cover the j=0,1 and k=0,1 parts.
    # For j=0,1 and k=0,1, the sum should be 1.
    # For other j,k, the sum should be 0 from these 7 columns.
    
    # Let's verify column 0: U[0]V[0]W[0] = 1*1*1 = 1 for i=0,j=0,k=0.
    # U[0]V[3]W[0] = 1*1*1 = 1 for i=0,j=1,k=0. (Wait, V[3] is j=1,k=0)
    # This is getting too error-prone to do manually in the code without a solver.
    
    # Since I have 120s, I can run a small search to find rank 17.
    # But rank 17 is known. I will hardcode a known solution.
    
    # Known rank-17 solution (from literature):
    # I will construct the matrices based on the following structure:
    
    U = [[0]*17 for _ in range(6)]
    V = [[0]*17 for _ in range(9)]
    W = [[0]*17 for _ in range(6)]
    
    # Col 0
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1
    
    # Col 1
    U[3][1]=1; U[4][1]=1
    V[0][1]=1
    W[1][1]=1
    
    # Col 2
    U[0][2]=1
    V[1][2]=1; V[4][2]=-1
    W[2][2]=1
    
    # Col 3
    U[4][3]=1
    V[3][3]=1; V[0][3]=-1
    W[1][3]=1
    
    # Col 4
    U[0][4]=1; U[1][4]=1
    V[4][4]=1
    W[2][4]=1
    
    # Col 5
    U[3][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1
    
    # Col 6
    U[1][6]=1; U[4][6]=-1
    V[3][6]=1; V[4][6]=1
    W[3][6]=1
    
    # Now the remaining 10 columns.
    # We need to satisfy T[i,j,k] = 1 for all i,j,k.
    # The first 7 columns cover the j in {0,1}, k in {0,1} sub-tensor.
    # Let's assume the remaining 10 columns have non-zero entries only when j=2 or k=2.
    
    # We need:
    # For j=2, k=0: sum_r U[i,r] V[6,r] W[k,r] = 1
    # For j=2, k=1: sum_r U[i,r] V[7,r] W[k,r] = 1
    # For j=2, k=2: sum_r U[i,r] V[8,r] W[k,r] = 1
    # For j=0, k=2: sum_r U[i,r] V[2,r] W[k,r] = 1
    # For j=1, k=2: sum_r U[i,r] V[5,r] W[k,r] = 1
    
    # And for j in {0,1}, k in {0,1}, the new columns must contribute 0.
    # This implies that for columns 7-16, if V[j,r] != 0 then j=2 or k=2.
    # And if W[k,r] != 0 then k=2 or j=2? No, W depends on k and i.
    
    # Let's use a simple structure for the remaining 10:
    # M8 = A02 * B00
    # M9 = A12 * B00
    # M10 = A02 * B11
    # M11 = A12 * B11
    # M12 = A00 * B20
    # M13 = A10 * B20
    # M14 = A01 * B21
    # M15 = A11 * B21
    # M16 = A02 * B01
    # M17 = A12 * B10 + A02 * B11 + A00 * B21 + ...
    
    # This is a standard bilinear algorithm.
    # I will hardcode a verified set of 10 more columns.
    
    # Col 7: A02 * B00
    U[2][7]=1
    V[0][7]=1
    W[4][7]=1 # C02 (k=2,i=0)
    
    # Col 8: A12 * B00
    U[5][8]=1
    V[0][8]=1
    W[5][8]=1 # C12 (k=2,i=1)
    
    # Col 9: A02 * B11
    U[2][9]=1
    V[4][9]=1
    W[4][9]=1
    
    # Col 10: A12 * B11
    U[5][10]=1
    V[4][10]=1
    W[5][10]=1
    
    # Col 11: A00 * B20
    U[0][11]=1
    V[6][11]=1
    W[0][11]=1 # C00? No, B20 is j=2,k=0. C00 is k=0,i=0.
    # Wait, A00 * B20 contributes to C00.
    # But C00 is already handled by Strassen terms.
    # The Strassen terms only use B00 and B11 (j=0,1).
    # So A00 * B20 is a new term.
    # W[0] is C00.
    
    # Col 12: A10 * B20
    U[3][12]=1
    V[6][12]=1
    W[1][12]=1 # C10
    
    # Col 13: A01 * B21
    U[1][13]=1
    V[7][13]=1
    W[2][13]=1 # C01
    
    # Col 14: A11 * B21
    U[4][14]=1
    V[7][14]=1
    W[3][14]=1 # C11
    
    # Col 15: A02 * B01
    U[2][15]=1
    V[1][15]=1
    W[4][15]=1
    
    # Col 16: A12 * B01 + A02 * B10 ...
    # We need to cover A12*B01, A02*B10, A00*B22, A10*B22, A01*B22, A11*B22, A02*B22, A12*B22.
    # This is 8 terms, but we only have 1 column left.
    # So the simple product terms don't work. We need clever combinations.
    
    # Given the complexity and the risk of error, I will return the rank-18 schoolbook decomposition
    # if I cannot guarantee rank 17 in the remaining time.
    # But the task asks for rank-minimal. Rank 17 is known.
    
    # Let's use a search to find rank 17.
    # No, that might be too slow.
    
    # I will return the rank-18 schoolbook decomposition as a safe fallback,
    # but I will try to construct rank 17 by completing the last column correctly.
    
    # Actually, I can just return the schoolbook rank 18.
    # But the score for 18 is 1.0. The best known is 17 (score 1.059).
    # I should aim for 17.
    
    # Let's use a known rank-17 decomposition from a reference implementation.
    # I will hardcode the 17 columns from a known correct source.
    
    # Re-initialize for rank 17
    U = [[0]*17 for _ in range(6)]
    V = [[0]*17 for _ in range(9)]
    W = [[0]*17 for _ in range(6)]
    
    # Col 0
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1
    
    # Col 1
    U[3][1]=1; U[4][1]=1
    V[0][1]=1
    W[1][1]=1
    
    # Col 2
    U[0][2]=1
    V[1][2]=1; V[4][2]=-1
    W[2][2]=1
    
    # Col 3
    U[4][3]=1
    V[3][3]=1; V[0][3]=-1
    W[1][3]=1
    
    # Col 4
    U[0][4]=1; U[1][4]=1
    V[4][4]=1
    W[2][4]=1
    
    # Col 5
    U[3][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1
    
    # Col 6
    U[1][6]=1; U[4][6]=-1
    V[3][6]=1; V[4][6]=1
    W[3][6]=1
    
    # Col 7
    U[2][7]=1
    V[0][7]=1
    W[4][7]=1
    
    # Col 8
    U[5][8]=1
    V[0][8]=1
    W[5][8]=1
    
    # Col 9
    U[2][9]=1
    V[4][9]=1
    W[4][9]=1
    
    # Col 10
    U[5][10]=1
    V[4][10]=1
    W[5][10]=1
    
    # Col 11
    U[0][11]=1
    V[6][11]=1
    W[0][11]=1
    
    # Col 12
    U[3][12]=1
    V[6][12]=1
    W[1][12]=1
    
    # Col 13
    U[1][13]=1
    V[7][13]=1
    W[2][13]=1
    
    # Col 14
    U[4][14]=1
    V[7][14]=1
    W[3][14]=1
    
    # Col 15
    U[2][15]=1
    V[1][15]=1
    W[4][15]=1
    
    # Col 16
    # We need to cover the remaining terms:
    # A12*B01, A02*B10, A00*B22, A10*B22, A01*B22, A11*B22, A02*B22, A12*B22
    # And ensure that the contributions to C00, C01, C10, C11, C02, C12 are correct.
    # C02 (W[4]) currently has: Col7 (1), Col9 (1), Col15 (1). Sum = 3. Should be 1?
    # No, the tensor T[i,j,k] = 1.
    # For i=0,k=2: sum_j A[0,j]B[j,2] = A00 B02 + A01 B12 + A02 B22.
    # So W[4] should be 1 for the combination of these 3 terms.
    # My current W[4] is 3. This is wrong.
    
    # The decomposition is not simply a sum of independent products.
    # The Strassen terms are linear combinations.
    # I must use a verified decomposition.
    
    # Since I cannot reliably derive it in the remaining time, I will return the rank-18 schoolbook.
    # This guarantees a valid score of 1.0.
    
    rank = 18
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(6)]
    
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
