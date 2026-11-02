# EVOLVE-BLOCK-START
import time

def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = 4, 4, 6
    
    # Use rank-8 decomposition for 2x2*2x3, giving total rank 7*8=56
    inner_rank = 8
    rank = 7 * inner_rank  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Outer Strassen Steps for 4x4 * 4x6
    outer_steps = [
        (([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])),
        (([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])),
        (([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])),
        (([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])),
        (([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])),
        (([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])),
        (([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]))
    ]
    
    def get_global_A(block_id, r, s):
        I = (block_id // 2) * 2 + r
        J = (block_id % 2) * 2 + s
        return I, J

    def get_global_B(block_id, s, k):
        J = (block_id // 2) * 2 + s
        K = (block_id % 2) * 3 + k
        return J, K

    def get_global_C(block_id, r, k):
        I = (block_id // 2) * 2 + r
        K = (block_id % 2) * 3 + k
        return I, K

    # Rank-8 decomposition for 2x2 * 2x3
    # A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]], C = A*B
    # C = [[ae+ch, af+ci, ag+cj],
    #      [de+dh, df+di, dg+dj]]
    #
    # 8-term decomposition:
    # M1 = (a+d)(e+h)  -> C00+=M1, C10+=M1
    # M2 = (a+c)(e+h)  -> C00+=M2, C10+=M2  (wait, this doesn't work cleanly)
    #
    # Let me use a known rank-8 scheme:
    # We can get rank 8 by using 7 Strassen terms for cols 0,1 and 1 shared term for col 2.
    # Actually, here's a valid rank-8 decomposition:
    #
    # Let's use the following 8 bilinear forms:
    # T1: (a+b)(e+f)     -> C00 += T1, C01 += T1
    # T2: (c+d)(e+f)     -> C10 += T2, C11 += T2
    # T3: (a+d)(e+h)     -> C00 += T3, C10 += T3
    # T4: (a+c)(e+h)     -> C00 += T4, C10 += T4
    # T5: (a+b)(g+i)     -> C02 += T5
    # T6: (c+d)(g+j)     -> C12 += T6
    # T7: (a-d)(e-h)     -> C00 -= T7, C10 -= T7
    # T8: (a-c)(e-h)     -> C00 -= T8, C10 -= T8
    #
    # This is getting complicated. Let me just use the straightforward 7+1=8 approach:
    # 7 Strassen terms for the 2x2*2x2 part (cols 0,1)
    # 1 term that somehow handles col 2 -- but 1 term can't handle 4 products.
    #
    # Actually rank of [2,2,3] is 8, not 7+1. Let me look up the correct decomposition.
    #
    # Correct rank-8 decomposition for [2,2,3]:
    # The 12 entries of C need to be expressed as 8 bilinear terms.
    #
    # Here's one valid construction:
    # M1 = (a+d)(e+h)      -> contributes to C[0,0], C[1,0]
    # M2 = (a+c)(e+h)      -> contributes to C[0,0], C[1,0]  
    # M3 = (b+d)(f+i)      -> contributes to C[0,1], C[1,1]
    # M4 = (b+c)(f+i)      -> contributes to C[0,1], C[1,1]
    # M5 = (a-d)(e-h)      -> contributes to C[0,0], C[1,0]
    # M6 = (b-d)(f-i)      -> contributes to C[0,1], C[1,1]
    # M7 = (a+b+c+d)(g+i+j) -> too complex
    #
    # I'll use a simpler verified approach: keep 7 Strassen + 4 schoolbook = 11,
    # but try to reduce the outer rank. Alternatively, just verify the current rank-77 works
    # and attempt a different outer decomposition.
    #
    # For now, let me use a clean rank-8 decomposition that I can verify:
    # Use the fact that [2,2,3] can be embedded in [2,2,2]x[2,2,1] but that gives 7+4=11.
    #
    # After careful consideration, the known minimal rank is 8. Here's a valid one:
    # 
    # M1 = (a+d)(e+h)       C[0,0]+=M1, C[1,0]+=M1
    # M2 = (a+c)(e+h)       C[0,0]+=M2, C[1,0]+=M2
    # M3 = (b+d)(f+i)       C[0,1]+=M3, C[1,1]+=M3
    # M4 = (b+c)(f+i)       C[0,1]+=M4, C[1,1]+=M4
    # M5 = (a-d)(e-h)       C[0,0]-=M5, C[1,0]-=M5
    # M6 = (b-d)(f-i)       C[0,1]-=M6, C[1,1]-=M6
    # M7 = (a+b)(g+i)       C[0,2]+=M7
    # M8 = (c+d)(g+j)       C[1,2]+=M8
    #
    # Let me verify: C[0,0] = M1+M2-M5 = (a+d)(e+h)+(a+c)(e+h)-(a-d)(e-h)
    # = (e+h)[a+d+a+c] - (a-d)(e-h)
    # = (e+h)(2a+c+d) - (ae-ah-de+dh)
    # = 2ae+ae+ce+de+2ah+ch+dh - ae+ah+de-dh
    # = 2ae+ce+3ah+ch  -- this is wrong, should be ae+ch
    #
    # This decomposition is incorrect. Let me use a verified one.
    #
    # I'll stick with rank 77 (7*11) which is verified to work, but try to find
    # a better outer decomposition. For now, return the working rank-77 solution.
    
    # Revert to the working rank-77 approach
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    inner_strassen = [
        (([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])),
        (([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])),
        (([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])),
        (([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])),
        (([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (0,0,-1)])),
        (([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])),
        (([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]))
    ]
    
    inner_schoolbook = [
        (([(0,0,1)], [(0,0,1)], [(0,0,1)])),
        (([(0,1,1)], [(1,0,1)], [(0,0,1)])),
        (([(1,0,1)], [(0,0,1)], [(1,0,1)])),
        (([(1,1,1)], [(1,0,1)], [(1,0,1)]))
    ]
    
    for i, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
        base_rank = i * 11
        
        for j, (A_inner, B_inner, C_inner) in enumerate(inner_strassen):
            c_global_rank = base_rank + j
            
            for (r, s, coeff) in A_inner:
                for (block_id, b_coeff) in A_outer:
                    I, J = get_global_A(block_id, r, s)
                    U[I * m + J][c_global_rank] += coeff * b_coeff
                    
            for (s, k, coeff) in B_inner:
                for (block_id, b_coeff) in B_outer:
                    J, K = get_global_B(block_id, s, k)
                    V[J * p + K][c_global_rank] += coeff * b_coeff
                    
            for (r, k, coeff) in C_inner:
                for (block_id, b_coeff) in C_outer:
                    I, K = get_global_C(block_id, r, k)
                    W[K * n + I][c_global_rank] += coeff * b_coeff

        for j, (A_inner, B_inner, C_inner) in enumerate(inner_schoolbook):
            c_global_rank = base_rank + 7 + j
            
            for (r, s, coeff) in A_inner:
                for (block_id, b_coeff) in A_outer:
                    I, J = get_global_A(block_id, r, s)
                    U[I * m + J][c_global_rank] += coeff * b_coeff
                    
            for (s, k_local, coeff) in B_inner:
                k_global_in_block = 2 
                for (block_id, b_coeff) in B_outer:
                    J, K = get_global_B(block_id, s, k_global_in_block)
                    V[J * p + K][c_global_rank] += coeff * b_coeff
                    
            for (r, k_local, coeff) in C_inner:
                k_global_in_block = 2
                for (block_id, b_coeff) in C_outer:
                    I, K = get_global_C(block_id, r, k_global_in_block)
                    W[K * n + I][c_global_rank] += coeff * b_coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
