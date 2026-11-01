"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = 2, 2, 13
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (26, rank)
    # W shape: (p*n, rank) = (26, rank)

    U_list = []
    V_list = []
    W_list = []

    def add(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    def time_check():
        return (time.time() - start_time) > DEADLINE

    # Strategy: Use Strassen for pairs of columns, handle odd column separately.
    # For 2x2 * 2x2, Strassen uses rank 7.
    # We have 13 columns. 13 = 2*6 + 1.
    # So 6 blocks of 2x2 using Strassen (6 * 7 = 42 rank) + 1 column using naive (2 mults? no, 2x2*2x1 is 4 mults, but can be done in 2? No, 2x1 * 2x1 is 2 mults. Wait.
    # A (2x2) * B (2x1).
    # Result (2x1).
    # r1 = a*b0 + b*b1
    # r2 = c*b0 + d*b1
    # This is equivalent to multiplying 2x2 by a column vector.
    # Can we do this in rank < 4?
    # Tensor is T[i,k,j] where i in 0..1 (row of result), k in 0..1 (col of B), j in 0..3 (entries of A).
    # Actually, the multiplication tensor for (2x2)@(2x1) is a 2x2x4 tensor.
    # Rank of matrix multiplication 2x1x1 is 2? No, 1x1x1 is 1.
    # 2x2x1 (A=2x2, B=1x1? No B is 2x1).
    # Let's check the rank of 2x2 * 2x1.
    # The operations are:
    # y0 = a*b0 + b*b1
    # y1 = c*b0 + d*b1
    # This is two dot products.
    # Can we share computations?
    # If we use (a+b)*b0 ... no.
    # Let's assume naive 4 multiplications for the last column?
    # a*b0, b*b1, c*b0, d*b1.
    # Wait, if I just treat it as 4 scalar multiplications, that's rank 4.
    # But maybe I can combine?
    # Actually, for the last column, I can just use 4 simple terms.
    # Total rank = 6 * 7 + 4 = 42 + 4 = 46.
    
    # Is there a better way for the tail?
    # 2x2 * 2x1.
    # y0 = a*b0 + b*b1
    # y1 = c*b0 + d*b1
    # Let's try to find a rank 3 or 4 decomposition.
    # It's a bilinear map from F^4 x F^2 -> F^2.
    # Tensor T in F^2 x F^2 x F^4.
    # T[y0, b0, a] = 1, T[y0, b1, b] = 1
    # T[y1, b0, c] = 1, T[y1, b1, d] = 1
    # This is the sum of 4 rank-1 tensors:
    # e_{y0} \otimes e_{b0} \otimes e_{a}
    # e_{y0} \otimes e_{b1} \otimes e_{b}
    # e_{y1} \otimes e_{b0} \otimes e_{c}
    # e_{y1} \otimes e_{b1} \otimes e_{d}
    # Can we reduce rank?
    # If we had y0 = (a+b)*(b0+b1) - ... 
    # Let's stick to 4 for safety and simplicity, or try to find a better one.
    # Actually, 2x2 * 2x1 is just two independent dot products of length 2.
    # Each dot product of length 2 has rank 2 (x0*y0 + x1*y1).
    # So total rank is 2 + 2 = 4.
    # So 4 is minimal for the tail if they are independent.
    
    # So base rank = 46.

    # Now, can we do better than Strassen for the 2x2 blocks?
    # Strassen is optimal for 2x2x2 (rank 7).
    # So 6*7 = 42 is optimal for the first 12 columns if we group them into 2x2 blocks.
    
    # Is there a way to mix columns?
    # e.g. process 4 columns at once?
    # 2x2 * 2x4.
    # Rank of 2x2*2x4?
    # Lower bound: 2*2*4 / 2 = 8? No.
    # Rank of M_{2,2,4}.
    # Using Strassen on 2x2 blocks gives 14.
    # Can we do better?
    # M_{2,2,4} can be seen as M_{2,2,2} * 2.
    # Maybe there is a rank 13 or 12 decomposition for 2x2*2x4?
    # If we can do 2x2*2x4 in rank 13, then 3 blocks of 4 cols + 1 col?
    # 13 cols = 4 + 4 + 4 + 1.
    # 3 * 13 + 4 = 39 + 4 = 43.
    # If 2x2*2x4 has rank 12, then 3*12+4 = 40.
    # If 2x2*2x4 has rank 11? Unlikely.
    
    # Let's check known ranks.
    # M_{2,2,2} = 7.
    # M_{2,2,4} ?
    # A 2x2 * 2x4 matrix multiplication.
    # We can use Strassen's algorithm recursively? No, Strassen is for square.
    # But we can use the Strassen decomposition for the 2x2 part and apply it to each 2x2 sub-block of B.
    # That gives 7 + 7 = 14.
    # Can we do better?
    # The tensor is T in F^4 x F^8 x F^8.
    # Rank >= 4*8/4 = 8? No, lower bound is max(n*m, m*p, p*n)/something?
    # Lower bound for M_{n,m,p} is n*m*p / (n+m+p)? No.
    # Simple lower bound: n*m for fixed B? No.
    # Let's assume 14 is hard to beat for 2x2*2x4.
    
    # What about 2x2 * 2x13 directly?
    # Maybe there is a specific decomposition for 2x2*2x13?
    # Unlikely to be significantly better than block-wise Strassen.
    
    # So 46 seems like a solid baseline.
    # Can we improve the tail?
    # Tail is 2x2 * 2x1. Rank 4.
    # Total 46.
    
    # Let's implement the 46-rank solution.
    
    # Strassen for 2x2 * 2x2 block:
    # A = [[a, b], [c, d]], B = [[b0, b1], [b2, b3]]
    # m1 = (a+d)(b0+b3)
    # m2 = (c+d)(b1)
    # m3 = (a-b)(b2+b3)
    # m4 = (c+d)(b0)
    # m5 = (a+c)(b3)
    # m6 = (a-c)(b0+b2)
    # m7 = (b-d)(b1+b3)
    # P00 = m1 + m4 - m5 + m7
    # P01 = m3 + m5
    # P10 = m2 + m4
    # P11 = m1 - m2 + m3 + m6
    
    # Let's verify indices.
    # U is (n*m, rank) -> (4, rank). Indices 0..3 correspond to a,b,c,d.
    # V is (m*p, rank) -> (26, rank). For block k (0..5), cols are 2k, 2k+1.
    #   V index for b0 (row 0, col 2k) is 2k.
    #   V index for b1 (row 0, col 2k+1) is 2k+1.
    #   V index for b2 (row 1, col 2k) is 2k + 13? No.
    #   V is (m*p, rank). m=2, p=13.
    #   Index = row * p + col.
    #   Row 0: 0..12.
    #   Row 1: 13..25.
    #   So for block k (cols 2k, 2k+1):
    #   b0 = V[2k]
    #   b1 = V[2k+1]
    #   b2 = V[13+2k]
    #   b3 = V[13+2k+1]
    
    # W is (p*n, rank) -> (26, rank).
    # Output is 2x13.
    # Index = col * n + row?
    # "W[k*n+i,r]" where k is column, i is row.
    # So for column c, row r: index = c*n + r = 2c + r.
    # P00 is row 0, col 2k. Index 2*(2k) + 0 = 4k.
    # P01 is row 0, col 2k+1. Index 2*(2k+1) + 0 = 4k+2.
    # P10 is row 1, col 2k. Index 2*(2k) + 1 = 4k+1.
    # P11 is row 1, col 2k+1. Index 2*(2k+1) + 1 = 4k+3.
    
    # Let's build the Strassen terms for one block.
    def strassen_block(k):
        """Add 7 terms for Strassen decomposition of 2x2 * 2x2 block k."""
        # U indices: 0:a, 1:b, 2:c, 3:d
        # V indices for block k:
        # v_b0 = 2*k
        # v_b1 = 2*k + 1
        # v_b2 = 13 + 2*k
        # v_b3 = 13 + 2*k + 1
        
        # W indices for block k:
        # w_p00 = 4*k
        # w_p01 = 4*k + 2
        # w_p10 = 4*k + 1
        # w_p11 = 4*k + 3
        
        terms = []
        
        # m1 = (a+d)(b0+b3)
        # U: [1, 0, 0, 1]
        # V: [b0=1, b3=1]
        # W: [p00=1, p11=1]
        u1 = [1, 0, 0, 1]
        v1 = [0]*26
        v1[2*k] = 1
        v1[13+2*k+1] = 1
        w1 = [0]*26
        w1[4*k] = 1
        w1[4*k+3] = 1
        terms.append((u1, v1, w1))
        
        # m2 = (c+d)(b1)
        # U: [0, 0, 1, 1]
        # V: [b1=1]
        # W: [p10=1, p11=-1]
        u2 = [0, 0, 1, 1]
        v2 = [0]*26
        v2[2*k+1] = 1
        w2 = [0]*26
        w2[4*k+1] = 1
        w2[4*k+3] = -1
        terms.append((u2, v2, w2))
        
        # m3 = (a-b)(b2+b3)
        # U: [1, -1, 0, 0]
        # V: [b2=1, b3=1]
        # W: [p01=1, p11=1]
        u3 = [1, -1, 0, 0]
        v3 = [0]*26
        v3[13+2*k] = 1
        v3[13+2*k+1] = 1
        w3 = [0]*26
        w3[4*k+2] = 1
        w3[4*k+3] = 1
        terms.append((u3, v3, w3))
        
        # m4 = (c+d)(b0)
        # U: [0, 0, 1, 1]
        # V: [b0=1]
        # W: [p00=1, p10=1]
        u4 = [0, 0, 1, 1]
        v4 = [0]*26
        v4[2*k] = 1
        w4 = [0]*26
        w4[4*k] = 1
        w4[4*k+1] = 1
        terms.append((u4, v4, w4))
        
        # m5 = (a+c)(b3)
        # U: [1, 0, 1, 0]
        # V: [b3=1]
        # W: [p00=-1, p01=1]
        u5 = [1, 0, 1, 0]
        v5 = [0]*26
        v5[13+2*k+1] = 1
        w5 = [0]*26
        w5[4*k] = -1
        w5[4*k+2] = 1
        terms.append((u5, v5, w5))
        
        # m6 = (a-c)(b0+b2)
        # U: [1, 0, -1, 0]
        # V: [b0=1, b2=1]
        # W: [p11=1]
        u6 = [1, 0, -1, 0]
        v6 = [0]*26
        v6[2*k] = 1
        v6[13+2*k] = 1
        w6 = [0]*26
        w6[4*k+3] = 1
        terms.append((u6, v6, w6))
        
        # m7 = (b-d)(b1+b3)
        # U: [0, 1, 0, -1]
        # V: [b1=1, b3=1]
        # W: [p00=1]
        u7 = [0, 1, 0, -1]
        v7 = [0]*26
        v7[2*k+1] = 1
        v7[13+2*k+1] = 1
        w7 = [0]*26
        w7[4*k] = 1
        terms.append((u7, v7, w7))
        
        return terms

    # Add 6 Strassen blocks for columns 0-11
    for k in range(6):
        if time_check():
            break
        terms = strassen_block(k)
        for u, v, w in terms:
            add(u, v, w)

    # Tail: column 12 (2x2 * 2x1)
    # B col = [b0, b1] where b0 = B[0,12], b1 = B[1,12]
    # V indices:
    # b0 = V[12]
    # b1 = V[13+12] = V[25]
    
    # Result:
    # P0 = a*b0 + b*b1
    # P1 = c*b0 + d*b1
    # W indices:
    # P0 (row 0, col 12) = W[12*2 + 0] = W[24]
    # P1 (row 1, col 12) = W[12*2 + 1] = W[25]
    
    # Term 1: a*b0
    add([1, 0, 0, 0], [0]*26 + [1] if False else [0]*12 + [1] + [0]*13, [0]*24 + [1] + [0])
    # Wait, V[12] is index 12.
    v_t1 = [0]*26
    v_t1[12] = 1
    w_t1 = [0]*26
    w_t1[24] = 1
    add([1, 0, 0, 0], v_t1, w_t1)
    
    # Term 2: b*b1
    v_t2 = [0]*26
    v_t2[25] = 1
    w_t2 = [0]*26
    w_t2[24] = 1
    add([0, 1, 0, 0], v_t2, w_t2)
    
    # Term 3: c*b0
    v_t3 = [0]*26
    v_t3[12] = 1
    w_t3 = [0]*26
    w_t3[25] = 1
    add([0, 0, 1, 0], v_t3, w_t3)
    
    # Term 4: d*b1
    v_t4 = [0]*26
    v_t4[25] = 1
    w_t4 = [0]*26
    w_t4[25] = 1
    add([0, 0, 0, 1], v_t4, w_t4)

    # Return the constructed matrices
    # U: list of 4-vectors
    # V: list of 26-vectors
    # W: list of 26-vectors
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
