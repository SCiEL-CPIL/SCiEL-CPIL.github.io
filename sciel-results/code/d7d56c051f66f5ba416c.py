"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_TIMEOUT = 140  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Strategy: split j into {0,1,2} (2x3x6) and {3,4} (2x2x6)
    # For 2x3x6: split k into 3 pairs, each using 10-term 2x3x2 algorithm -> 30
    # For 2x2x6: split k into 3 pairs, each using 7-term Strassen 2x2x2 -> 21
    # Total rank = 30 + 21 = 51
    
    rank = 51
    
    U = [[0] * rank for _ in range(n * m)]  # 10 x 51
    V = [[0] * rank for _ in range(m * p)]  # 30 x 51
    W = [[0] * rank for _ in range(p * n)]  # 12 x 51
    
    col = 0
    
    # Part 1: j in {0, 1, 2} -> 2x3x6, split k into pairs (0,1), (2,3), (4,5)
    # Each pair uses 10-term algorithm for 2x3x2
    # A is 2x3 (rows i=0,1; cols j=0,1,2), B is 3x2 (rows j=0,1,2; cols k=k0,k1)
    # C is 2x2 (rows i=0,1; cols k=k0,k1)
    # 
    # 10-term algorithm for 2x3x2:
    # Let A = [[a,b,c],[d,e,f]], B = [[g,h],[i,j],[k,l]]
    # C[0,0] = ag + bi + ck
    # C[0,1] = ah + bj + cl
    # C[1,0] = dg + ei + fk
    # C[1,1] = dh + ej + fl
    #
    # M1 = (a+b)(g+i)
    # M2 = c*k
    # M3 = (d+e)(g+i)
    # M4 = f*l
    # M5 = a*(h-j)
    # M6 = b*(i-j)  -- wait, let me re-derive
    #
    # Actually, let me use a verified 10-term scheme:
    # M1 = (a+b)(g+i)
    # M2 = c*k
    # M3 = (d+e)(g+i)  
    # M4 = f*l
    # M5 = a*(h-j)
    # M6 = b*(j-i)
    # M7 = d*(h-j)
    # M8 = e*(j-i)
    # M9 = (a+b)*(h-j+k)
    # M10 = (d+e)*(h-j+k)
    #
    # C[0,0] = M1 - M5 - M9 + M2  ... this is getting complex.
    # Let me just use a known correct 10-term decomposition.
    
    # Verified 10-term for 2x3x2 (from literature):
    # A = [[a,b,c],[d,e,f]], B = [[g,h],[i,j],[k,l]]
    # M1 = a*g, M2 = b*i, M3 = c*k, M4 = d*g, M5 = e*i, M6 = f*l
    # M7 = (a+b)*(h-j), M8 = (d+e)*(h-j), M9 = (b+c)*(j-i), M10 = (e+f)*(j-i)
    # C[0,0] = M1 + M2 + M3
    # C[0,1] = M7 - M9 + (ah + bj + cl - (M7 - M9)) ... 
    # This isn't clean either.
    
    # Let me use a simpler approach: just do 2x3x6 with 12 terms per k-pair (naive)
    # That gives 3*12 = 36, plus 21 = 57. Worse than 54.
    
    # Alternative: Use the current 54-rank approach but fix the Strassen terms properly.
    # Actually, let me try a different split: j = {0} (2x1x6=12) + j={1,2,3,4} (2x4x6)
    # For 2x4x6: split into two 2x2x6 blocks, each with Strassen (7 per k-pair, 3 pairs) = 21 each
    # Total: 12 + 21 + 21 = 54. Same as before.
    
    # Let me try yet another approach: Use known results.
    # R(2,5,6) <= 50 is known. Let me try to construct rank 50.
    # 
    # Split: j={0,1} (2x2x6=21) + j={2,3,4} (2x3x6)
    # For 2x3x6: If we can do it in 29, total = 50.
    # 2x3x6: split k into pairs. Each 2x3x2 needs 10 terms. 3*10=30.
    # So 21+30=51.
    #
    # Or: j={0} (12) + j={1,2,3,4} (2x4x6)
    # 2x4x6: split into j={1,2} and j={3,4}, each 2x2x6=21. Total 12+21+21=54.
    
    # I'll go with the 51-rank approach using 10-term 2x3x2.
    # Let me carefully define the 10-term algorithm.
    
    # 10-term algorithm for 2x3x2:
    # A = [[a,b,c],[d,e,f]], B = [[g,h],[i,j],[k,l]]
    # C = A*B = [[ag+bi+ck, ah+bj+cl], [dg+ei+fk, dh+ej+fl]]
    #
    # M1 = (a+b+c)*(g+i+k)
    # M2 = a*(h-j)
    # M3 = b*(i-k)  
    # M4 = d*(h-j)
    # M5 = e*(i-k)
    # M6 = c*(j-i)
    # M7 = f*(j-i)
    # M8 = (a+d)*(g+h)
    # M9 = (b+e)*(i+j)
    # M10 = (c+f)*(k+l)
    #
    # This is getting too complicated to verify mentally. Let me just use the naive 
    # 12-term per 2x3x2 block and accept rank 54, but implement it correctly.
    
    # Actually, the simplest correct approach: use rank 54 with proper Strassen.
    # Let me implement it correctly this time.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=0, all k=0..5. 2x1x6 = 12 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: j in {1,2,3,4}, split into two 2x2x6 blocks
    # Block A: j={1,2}, Block B: j={3,4}
    # Each 2x2x6: split k into 3 pairs, each using Strassen 2x2x2 (7 terms)
    
    for block in range(2):
        j0 = 1 + block * 2  # 1 or 3
        j1 = j0 + 1         # 2 or 4
        
        for k_pair in range(3):
            k0 = k_pair * 2
            k1 = k_pair + 1
            
            # Strassen 2x2x2:
            # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
            # C = A*B
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-e)
            # M5 = (a+b)h
            # M6 = (c-a)(e+f)
            # M7 = (b-d)(g+h)
            #
            # C[0,0] = M1 + M4 - M5 + M7
            # C[0,1] = M3 + M5
            # C[1,0] = M2 + M4
            # C[1,1] = M1 - M2 + M3 - M7
            
            strassen = [
                # (u_coeffs, v_coeffs, w_coeffs)
                # u: [(i_local, j_local, val)], v: [(j_local, k_local, val)], w: [(i_local, k_local, val)]
                # M1: (a+d)(e+h)
                ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
                # M2: (c+d)e
                ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
                # M3: a(f-h)
                ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
                # M4: d(g-e)
                ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
                # M5: (a+b)h
                ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
                # M6: (c-a)(e+f)
                ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
                # M7: (b-d)(g+h)
                ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
            ]
            
            for term in strassen:
                u_list, v_list, w_list = term
                
                for (li, lj, val) in u_list:
                    gi = li
                    gj = j0 + lj
                    U[gi * m + gj][col] += val
                
                for (lj, lk, val) in v_list:
                    gj = j0 + lj
                    gk = k0 + lk
                    V[gj * p + gk][col] += val
                
                for (li, lk, val) in w_list:
                    gi = li
                    gk = k0 + lk
                    W[gk * n + gi][col] += val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
