import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]
n, m, p = DIMENSIONS

def construct():
    t_start = time.time()
    deadline = t_start + 140.0

    # Schoolbook (rank 60) as safe fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {'U': U, 'V': V, 'W': W}

    def _verify(U, V, W, tol=1e-8):
        # Check: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        # and for all (i,j,k) != target, the sum should be 0.
        # The tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # Actually the matmul tensor is T[i,j,k] = 1 if we're computing (AB)_{ik} = sum_j A_{ij} B_{jk}
        # So the tensor has T[i,j,k] = 1 for all i,j,k.
        # Check: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] should equal 1 for all i,j,k
        rank = len(U[0])
        Ua = np.array(U)
        Va = np.array(V)
        Wa = np.array(W)
        # Build the tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                    if abs(s - 1) > tol:
                        return False
        return True

    best_result = None
    best_rank = n * m * p  # schoolbook rank = 60

    # Try Strassen-based approach for 2x2x2 blocks
    # A is 2x3, B is 3x10.
    # We can group columns of B into pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    # For each pair of columns, we have a 2x3 x 3x2 subproblem.
    # The 3x2 part: B's rows 0,1,2 times columns (2b, 2b+1)
    # We can use Strassen for 2x2x2 on rows 0,1 of B, and handle row 2 separately.
    # Actually, let's think more carefully.
    #
    # (AB)_{i,k} = sum_{j=0}^{2} A_{i,j} B_{j,k}
    # For columns k=2b and k=2b+1:
    # (AB)_{i,2b} = A_{i,0} B_{0,2b} + A_{i,1} B_{1,2b} + A_{i,2} B_{2,2b}
    # (AB)_{i,2b+1} = A_{i,0} B_{0,2b+1} + A_{i,1} B_{1,2b+1} + A_{i,2} B_{2,2b+1}
    #
    # This is a 2x3 x 3x2 problem. The schoolbook rank is 2*3*2 = 12 per block.
    # Can we do better?
    #
    # Let's use Strassen on the 2x2x2 part (rows 0,1 of B):
    # M1 = (A_{0,0}+A_{1,1})(B_{0,2b}+B_{1,2b+1})
    # M2 = (A_{1,0}+A_{1,1})B_{0,2b}
    # M3 = A_{0,0}(B_{1,2b}-B_{0,2b+1})
    # M4 = A_{1,1}B_{1,2b}
    # M5 = (A_{0,0}+A_{0,1})B_{1,2b+1}
    # M6 = (A_{1,0}-A_{0,0})(B_{0,2b}+B_{0,2b+1})
    # M7 = (A_{1,1}-A_{0,1})(B_{0,2b+1}+B_{1,2b+1})
    #
    # Wait, standard Strassen for C = AB where A is 2x2, B is 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b01)
    # M4 = a11 b11
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a11-a01)(b01+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # So for each block b (columns 2b, 2b+1 of B), using Strassen on rows 0,1:
    # 7 terms for the 2x2x2 part.
    # Plus 2 terms for row 2 of B: A_{i,0}... no wait.
    # We need:
    # (AB)_{i,2b} += A_{i,2} B_{2,2b}
    # (AB)_{i,2b+1} += A_{i,2} B_{2,2b+1}
    # These are 2 additional terms (one for each i? No, for each k).
    # Actually:
    # (AB)_{0,2b} = [Strassen C00] + A_{0,2} B_{2,2b}
    # (AB)_{0,2b+1} = [Strassen C01] + A_{0,2} B_{2,2b+1}
    # (AB)_{1,2b} = [Strassen C10] + A_{1,2} B_{2,2b}
    # (AB)_{1,2b+1} = [Strassen C11] + A_{1,2} B_{2,2b+1}
    #
    # The extra terms: A_{0,2}B_{2,2b}, A_{0,2}B_{2,2b+1}, A_{1,2}B_{2,2b}, A_{1,2}B_{2,2b+1}
    # That's 4 more terms? No wait.
    # Each of these is a product of one A entry and one B entry.
    # A_{0,2}B_{2,2b}: U has A_{0,2}, V has B_{2,2b}, W has C_{0,2b}
    # A_{0,2}B_{2,2b+1}: U has A_{0,2}, V has B_{2,2b+1}, W has C_{0,2b+1}
    # A_{1,2}B_{2,2b}: U has A_{1,2}, V has B_{2,2b}, W has C_{1,2b}
    # A_{1,2}B_{2,2b+1}: U has A_{1,2}, V has B_{2,2b+1}, W has C_{1,2b+1}
    # That's 4 extra terms per block.
    #
    # Total per block: 7 + 4 = 11 terms.
    # 5 blocks: 55 total.
    #
    # But wait, can we share some terms? The A_{0,2} and A_{1,2} appear in multiple blocks.
    # No, each term is a single product in the tensor decomposition, so we can't share.
    # Actually in tensor decomposition, each "term" is a rank-1 tensor. We can't combine different products.
    #
    # Let me reconsider. Maybe we can do better with a different grouping.
    #
    # Alternative: treat the whole thing as 2x3 x 3x10.
    # The tensor is T[i,j,k] = 1 for all i,j,k.
    # Actually no, the matmul tensor is T[i,j,k] = 1 (the structural tensor).
    # The rank of the matmul tensor for n x m times m x p is the bilinear complexity.
    #
    # For 2x3 x 3x10:
    # One approach: group into 5 blocks of 2x3 x 3x2.
    # Each 2x3 x 3x2 has rank at most... let's think.
    # We can write it as 2x2x2 (Strassen, rank 7) + 2x1x2 (rank 4, since 2*1*2=4 and no known improvement)
    # = 11 per block. Total 55.
    #
    # Can we do 2x3 x 3x2 in less than 11?
    # The tensor is sum_{j=0}^{2} a_j b_j where a_j is a 2-dim vector (A_{0,j}, A_{1,j}) and b_j is 2-dim (B_{j,0}, B_{j,1}).
    # This is a 2x3 x 3x2 tensor. Its rank is at most 6 (schoolbook). Wait, 2*3*2 = 12? No.
    # Schoolbook for 2x3 x 3x2: for each i in {0,1}, j in {0,1,2}, k in {0,1}: one term. That's 12 terms.
    # With Strassen on 2x2x2: 7 terms for the 2x2x2 part + 4 for the remaining = 11.
    # Can we do better?
    #
    # Actually, the 2x3 x 3x2 tensor can be seen as:
    # C = A B where A is 2x3, B is 3x2.
    # C_{ik} = sum_j A_{ij} B_{jk}
    #
    # Let me try to find a decomposition of rank 10 or less.
    #
    # Actually, let's just go with the 55-rank Strassen-based decomposition and verify it.
    # If time permits, try to optimize.

    def _strassen_block_2x3x3x2():
        """
        Build a decomposition for 2x3 x 3x2 multiplication.
        A: 2x3, B: 3x2, C: 2x2.
        U: 6 x rank, V: 6 x rank, W: 4 x rank.
        Returns (U, V, W) as lists of lists.
        """
        # Strassen terms for rows 0,1 of B (2x2x2 subproblem)
        # M1 = (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11
        # M2 = (A10+A11)B00 -> contributes to C10, C11
        # M3 = A00(B10-B01) -> contributes to C01, C11
        # M4 = A11*B11 -> contributes to C00, C10, C11
        # M5 = (A00+A01)B11 -> contributes to C00, C01
        # M6 = (A10-A00)(B00+B01) -> contributes to C11
        # M7 = (A11-A01)(B01+B11) -> contributes to C00, C11
        #
        # Wait, I need to be more careful with the Strassen formulas.
        # Standard Strassen for C = AB, A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(g-f)
        # M4 = d*h
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (d-b)(f+h)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # In our case:
        # A = [[A00, A01, A02], [A10, A11, A12]]
        # B = [[B00, B01], [B10, B11], [B20, B21]]
        #
        # For the 2x2x2 part (using rows 0,1 of B, cols 0,1 of A):
        # a = A00, b = A01, c = A10, d = A11
        # e = B00, f = B01, g = B10, h = B11
        #
        # M1 = (A00+A11)(B00+B11)
        #   U: A00 (idx 0), A11 (idx 4)
        #   V: B00 (idx 0), B11 (idx 11)  [B index: j*p+k, j=1, k=1 -> 1*2+1=3? Wait, p=2 here]
        #   Let me re-index. In the subproblem, B is 3x2, so B[j,k] with j in {0,1,2}, k in {0,1}.
        #   V index: j*2 + k.
        #   B00: j=0,k=0 -> idx 0
        #   B11: j=1,k=1 -> idx 3
        #
        # Let me redefine for the subproblem:
        # A: 2x3, indices (i,j) -> i*3+j
        # B: 3x2, indices (j,k) -> j*2+k
        # C: 2x2, indices (i,k) -> i*2+k
        #
        # Strassen terms (using A00,A01,A10,A11 and B00,B01,B10,B11):
        # M1 = (A00+A11)(B00+B11)
        #   U: [A00=1, A11=1] -> indices 0, 4
        #   V: [B00=1, B11=1] -> indices 0, 3
        #   W: [C00=1, C01=1, C10=1, C11=1] -> indices 0,1,2,3
        #
        # M2 = (A10+A11)B00
        #   U: [A10=1, A11=1] -> indices 3, 4
        #   V: [B00=1] -> index 0
        #   W: [C10=1, C11=1] -> indices 2, 3
        #
        # M3 = A00(B10-B01)
        #   U: [A00=1] -> index 0
        #   V: [B10=1, B01=-1] -> indices 2, 1
        #   W: [C01=1, C11=1] -> indices 1, 3
        #
        # M4 = A11*B11
        #   U: [A11=1] -> index 4
        #   V: [B11=1] -> index 3
        #   W: [C00=1, C10=1, C11=1] -> indices 0, 2, 3
        #
        # M5 = (A00+A01)B11
        #   U: [A00=1, A01=1] -> indices 0, 1
        #   V: [B11=1] -> index 3
        #   W: [C00=1, C01=1] -> indices 0, 1
        #
        # M6 = (A10-A00)(B00+B01)
        #   U: [A10=1, A00=-1] -> indices 3, 0
        #   V: [B00=1, B01=1] -> indices 0, 1
        #   W: [C11=1] -> index 3
        #
        # M7 = (A11-A01)(B01+B11)
        #   U: [A11=1, A01=-1] -> indices 4, 1
        #   V: [B01=1, B11=1] -> indices 1, 3
        #   W: [C00=1, C11=1] -> indices 0, 3
        #
        # Additional terms for row 2 of B:
        # M8 = A02*B20
        #   U: [A02=1] -> index 2
        #   V: [B20=1] -> index 4
        #   W: [C00=1] -> index 0
        #
        # M9 = A02*B21
        #   U: [A02=1] -> index 2
        #   V: [B21=1] -> index 5
        #   W: [C01=1] -> index 1
        #
        # M10 = A12*B20
        #   U: [A12=1] -> index 5
        #   V: [B20=1] -> index 4
        #   W: [C10=1] -> index 2
        #
        # M11 = A12*B21
        #   U: [A12=1] -> index 5
        #   V: [B21=1] -> index 5
        #   W: [C11=1] -> index 3
        #
        # Total: 11 terms per block.

        terms = []
        # Each term: (U_indices_with_vals, V_indices_with_vals, W_indices_with_vals)
        # M1
        terms.append(([0, 4], [0, 3], [0, 1, 2, 3]))
        # M2
        terms.append(([3, 4], [0], [2, 3]))
        # M3
        terms.append(([0], [2, 1], [1, 3]))  # B10=1, B01=-1
        # M4
        terms.append(([4], [3], [0, 2, 3]))
        # M5
        terms.append(([0, 1], [3], [0, 1]))
        # M6
        terms.append(([3, 0], [0, 1], [3]))  # A10=1, A00=-1
        # M7
        terms.append(([4, 1], [1, 3], [0, 3]))  # A11=1, A01=-1
        # M8
        terms.append(([2], [4], [0]))
        # M9
        terms.append(([2], [5], [1]))
        # M10
        terms.append(([5], [4], [2]))
        # M11
        terms.append(([5], [5], [3]))

        # Set the values properly
        # M3: V has B10=1, B01=-1
        # M6: U has A10=1, A00=-1
        # M7: U has A11=1, A01=-1

        U = [[0]*11 for _ in range(6)]
        V = [[0]*11 for _ in range(6)]
        W = [[0]*11 for _ in range(4)]

        # M1
        U[0][0] = 1; U[4][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
        # M2
        U[3][1] = 1; U[4][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        # M3
        U[0][2] = 1
        V[2][2] = 1; V[1][2] = -1
        W[1][2] = 1; W[3][2] = 1
        # M4
        U[4][3] = 1
        V[3][3] = 1
        W[0][3] = 1; W[2][3] = 1; W[3][3] = 1
        # M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1; W[1][4] = 1
        # M6
        U[3][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # M7
        U[4][6] = 1; U[1][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        # M8
        U[2][7] = 1
        V[4][7] = 1
        W[0][7] = 1
        # M9
        U[2][8] = 1
        V[5][8] = 1
        W[1][8] = 1
        # M10
        U[5][9] = 1
        V[4][9] = 1
        W[2][9] = 1
        # M11
        U[5][10] = 1
        V[5][10] = 1
        W[3][10] = 1

        return U, V, W

    # Build the full 2x3 x 3x10 decomposition using 5 blocks
    sub_U, sub_V, sub_W = _strassen_block_2x3x3x2()
    sub_rank = 11

    full_rank = 5 * sub_rank
    U_full = [[0]*full_rank for _ in range(n*m)]
    V_full = [[0]*full_rank for _ in range(m*p)]
    W_full = [[0]*full_rank for _ in range(p*n)]

    for b in range(5):
        base = b * sub_rank
        # U: A is 2x3, so rows 0-5 map directly
        for i in range(6):
            for c in range(sub_rank):
                U_full[i][base + c] = sub_U[i][c]
        # V: B is 3x10. For block b, columns 2b, 2b+1.
        # V index in subproblem: j*2+k where k is 0 or 1 (local column).
        # In full: j*10 + (2b + k)
        for j in range(3):
            for k_local in range(2):
                sub_v_idx = j * 2 + k_local
                full_v_idx = j * 10 + (2 * b + k_local)
                for c in range(sub_rank):
                    V_full[full_v_idx][base + c] = sub_V[sub_v_idx][c]
        # W: C is 2x10. For block b, columns 2b, 2b+1.
        # W index in subproblem: i*2+k where k is 0 or 1 (local column).
        # In full: (2b + k)*2 + i
        for i in range(2):
            for k_local in range(2):
                sub_w_idx = i * 2 + k_local
                full_w_idx = (2 * b + k_local) * 2 + i
                for c in range(sub_rank):
                    W_full[full_w_idx][base + c] = sub_W[sub_w_idx][c]

    # Verify
    if _verify(U_full, V_full, W_full):
        if full_rank < best_rank:
            best_rank = full_rank
            best_result = {'U': U_full, 'V': V_full, 'W': W_full}

    # Return the best result found
    if best_result is not None:
        return best_result
    return _schoolbook()
