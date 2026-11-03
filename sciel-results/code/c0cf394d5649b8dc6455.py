"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 5
    deadline = time.time() + 110.0  # 150s limit, leave safety margin

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, c] = 1.0
                    V[j*p + k, c] = 1.0
                    W[k*n + i, c] = 1.0
                    c += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, tol=1e-9):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Check: (A B)_{i,k} = sum_j A_{i,j} B_{j,k}
        # = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] V[j*p+k, c])
        # Equivalently, for each c, the outer product U_c (n*m) x V_c (m*p) x W_c (p*n)
        # must correspond to a rank-1 contribution to the matmul tensor.
        # Let's check: for all (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # And for any (i,j,k) where the matmul tensor entry is 0, the sum should be 0.
        # The matmul tensor is T_{i,j,k} = 1 if (i,j,k) valid, i.e., it's always 1 for valid indices.
        # Wait, the matmul tensor M_{i,j,k} = A_{i,j} * B_{j,k} summed over j... no.
        # The tensor is T_{i,j,k} = delta (for matmul, T_{i,j,k} = 1 if we multiply (i,j) of A with (j,k) of B)
        # Actually the multiplication tensor is M where M[i,j,k] = 1 for all valid i,j,k (it's the identity in some sense)
        # No: the multiplication tensor mu satisfies (A*B)_{i,k} = sum_{j} A_{i,j} B_{j,k}
        # So the tensor is mu_{i,j,k} = 1 (it's the constant tensor that picks out the j-th element)
        # Wait no. mu_{i,j,k} = 1 for all i, j, k where the index j is shared. It's literally all ones.
        # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all valid (i,j,k)
        nm = n*m
        mp = m*p
        pn = p*n
        rank = U.shape[1]
        # Compute all entries
        # Efficient: build the tensor via einsum or loops
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        return ok

    def try_strassen_block():
        # Split A (3x5) into rows: A_top (2x5), A_bot (1x5)
        # Split B (5x5) into cols: B_left (5x2), B_right (5x3)
        # AB = [A_top; A_bot] * [B_left B_right]
        # = [[A_top B_left, A_top B_right], [A_bot B_left, A_bot B_right]]
        # Each block product:
        # A_top (2x5) * B_left (5x2): use 2x2x2 Strassen? No, middle dim is 5.
        # We need to compute (2x5)(5x2) = 2x2 result. Rank of 2x5x2 matmul?
        # Known: rank of 2xnxn matmul for n>=2... For 2x2x2, rank is 7 (Strassen).
        # For 2x5x2, we can use block Strassen on the 2x2 "outer" but inner dim is 5.
        # Actually, m_{2,5,2}: split the 5 into 2+3? No, it's a single dimension.
        # m_{2,p,2} rank: we can compute it as sum over p of m_{1,1,1}? No.
        # (2x5)(5x2): result is 2x2. Each entry (i,k) = sum_{j=1}^{5} A_{i,j} B_{j,k}.
        # This is equivalent to computing 4 independent bilinear forms of length 5.
        # Rank of computing 4 dot products of length 5: at least 5 (since each dot product needs rank 5... no)
        # Actually, the tensor for (2x5)(5x2) has rank at least max(2*2, ...) 
        # Known result: rank of m_{2,n,2} for n >= 2 is at most... 
        # Let's just use schoolbook for these: 2*5*2 = 20 for each block.
        # That's worse. Let me try a different split.
        
        # Alternative: Use the fact that m_{2,2,2} has rank 7.
        # Split A (3x5) into 3 blocks of (1x5)? Or 2x5 + 1x5?
        # Split into A1 (2x5) and A2 (1x5).
        # Split B (5x5) into B1 (5x3) and B2 (5x2)? 
        # Then:
        # A1*B1: (2x5)(5x3) -> 2x3. Rank of m_{2,5,3}?
        # A1*B2: (2x5)(5x2) -> 2x2. 
        # A2*B1: (1x5)(5x3) -> 1x3.
        # A2*B2: (1x5)(5x2) -> 1x2.
        # This doesn't help much.

        # Let's try: Split 3x5 * 5x5 using 3 = 1+1+1 and 5 = 2+3, 5 = 2+3?
        # 9 block products, many of rank 1 (1x2x2 = 4, 1x3x2 = 6, etc.)
        # 9 products: three of (1x2)(2x2) = 1x2x2 (rank 4 each? no, 1x2x2 = 4)
        # This is getting complicated. Let's just return schoolbook.
        return schoolbook()

    # Return schoolbook immediately for correctness
    return schoolbook()
# EVOLVE-BLOCK-END
