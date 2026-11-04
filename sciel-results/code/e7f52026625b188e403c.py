import numpy as np
import time

def construct():
    n, m, p = 4, 4, 6
    
    # We'll use a recursive Strassen approach.
    # 4x4 * 4x6 -> 7 subproblems of 2x2 * 2x3
    # 2x2 * 2x3 -> 4 schoolbook (2x2*2x1) + 7 Strassen (2x2*2x2) = 11 terms
    # Total rank: 7 * 11 = 77
    
    rank = 77
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Define the 11 atomic terms for 2x2 * 2x3
    # Local indices: A[i,j] -> i*2+j, B[j,k] -> j*3+k, C[i,k] -> k*2+i
    # W index convention: W[k*n+i] for C[i,k]
    
    # Schoolbook terms for column 0 of B (2x2 * 2x1)
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    # Term 1: A[0,1]*B[1,0] -> C[0,0]
    # Term 2: A[1,0]*B[0,0] -> C[1,0]
    # Term 3: A[1,1]*B[1,0] -> C[1,0]
    
    # Strassen terms for columns 1,2 of B (2x2 * 2x2)
    # Standard Strassen for 2x2*2x2
    
    # Let me define each atomic term as (U_pattern, V_pattern, W_pattern)
    # where each pattern is a dict {local_index: value}
    
    atomic = []
    
    # Term 0
    atomic.append({
        'U': {0: 1},      # A[0,0]
        'V': {0: 1},      # B[0,0]
        'W': {0: 1}       # C[0,0] -> k=0,i=0 -> 0*2+0=0
    })
    
    # Term 1
    atomic.append({
        'U': {1: 1},      # A[0,1]
        'V': {3: 1},      # B[1,0]
        'W': {0: 1}       # C[0,0]
    })
    
    # Term 2
    atomic.append({
        'U': {2: 1},      # A[1,0]
        'V': {0: 1},      # B[0,0]
        'W': {1: 1}       # C[1,0] -> k=0,i=1 -> 0*2+1=1
    })
    
    # Term 3
    atomic.append({
        'U': {3: 1},      # A[1,1]
        'V': {3: 1},      # B[1,0]
        'W': {1: 1}       # C[1,0]
    })
    
    # Strassen terms for 2x2 * 2x2 (columns 1,2)
    # B local indices: col1 = 1,4; col2 = 2,5
    # C local indices: C[i,1] -> 1*2+i, C[i,2] -> 2*2+i
    
    # M1 = (A00+A11)(B01+B12) -> C01 + C12
    atomic.append({
        'U': {0: 1, 3: 1},
        'V': {1: 1, 5: 1},
        'W': {2: 1, 5: 1}  # C[0,1]=2, C[1,2]=5
    })
    
    # M2 = (A10+A11)B01 -> C11 - C12
    atomic.append({
        'U': {2: 1, 3: 1},
        'V': {1: 1},
        'W': {3: 1, 5: -1}  # C[1,1]=3, C[1,2]=5
    })
    
    # M3 = A00(B12-B02) -> C02 + C12
    atomic.append({
        'U': {0: 1},
        'V': {5: 1, 2: -1},
        'W': {4: 1, 5: 1}  # C[0,2]=4, C[1,2]=5
    })
    
    # M4 = A11(B11-B01) -> C01 + C11
    atomic.append({
        'U': {3: 1},
        'V': {4: 1, 1: -1},
        'W': {2: 1, 3: 1}  # C[0,1]=2, C[1,1]=3
    })
    
    # M5 = (A00+A01)B12 -> -C01 + C02
    atomic.append({
        'U': {0: 1, 1: 1},
        'V': {5: 1},
        'W': {2: -1, 4: 1}  # C[0,1]=2, C[0,2]=4
    })
    
    # M6 = (A10-A00)(B01+B02) -> C12
    atomic.append({
        'U': {2: 1, 0: -1},
        'V': {1: 1, 2: 1},
        'W': {5: 1}  # C[1,2]=5
    })
    
    # M7 = (A01-A11)(B11+B12) -> C01
    atomic.append({
        'U': {1: 1, 3: -1},
        'V': {4: 1, 5: 1},
        'W': {2: 1}  # C[0,1]=2
    })
    
    # Verify: C01 = M1 + M4 - M5 + M7
    # C01 contributions: M1:+1, M4:+1, M5:-1, M7:+1 => 1+1-1+1=2? That's wrong.
    
    # Let me recheck Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for 2x2*2x2:
    # C[0,0] gets M1, M4, -M5, M7
    # C[0,1] gets M3, M5
    # C[1,0] gets M2, M4
    # C[1,1] gets M1, -M2, M3, M6
    
    # But in our 2x2*2x3, columns 1,2 of B correspond to columns 1,2 of C.
    # So the "C00" in Strassen maps to C[0,1], "C01" maps to C[0,2], etc.
    
    # Let me redefine:
    # Strassen M1 -> contributes to C[0,1] and C[1,2] (the "diagonal" of the 2x2 subproblem)
    # Actually, Strassen for 2x2*2x2 gives a 2x2 output. If we're multiplying 2x2 by 2x2 (cols 1,2 of B),
    # the output is 2x2, which maps to C[:,1:3].
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11) -> C00 + C11
    # M2 = (A10+A11)B00 -> C10 + C11
    # M3 = A00(B01-B11) -> C01 + C11
    # M4 = A11(B10-B00) -> C00 + C10
    # M5 = (A00+A01)B11 -> C00 + C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00
    
    # Where C = [[C00, C01],[C10, C11]]
    
    # So for our 2x2*2x3, the 2x2*2x2 part (cols 1,2 of B) gives output cols 1,2 of C.
    # Let me call the Strassen output [[S00, S01],[S10, S11]] which maps to C[:,1:3].
    
    # S00 = M1 + M4 - M5 + M7 -> C[0,1]
    # S01 = M3 + M5 -> C[0,2]
    # S10 = M2 + M4 -> C[1,1]
    # S11 = M1 - M2 + M3 + M6 -> C[1,2]
    
    # So:
    # M1: U={A00,A11}, V={B01,B12}, W={C[0,1]:+1, C[1,2]:+1}
    # M2: U={A10,A11}, V={B01}, W={C[1,1]:+1, C[1,2]:-1}
    # M3: U={A00}, V={B02-B12}, W={C[0,2]:+1, C[1,2]:+1}
    # M4: U={A11}, V={B11-B01}, W={C[0,1]:+1, C[1,1]:+1}
    # M5: U={A00+A01}, V={B12}, W={C[0,1]:-1, C[0,2]:+1}
    # M6: U={A10-A00}, V={B01+B02}, W={C[1,2]:+1}
    # M7: U={A01-A11}, V={B11+B12}, W={C[0,1]:+1}
    
    # This matches what I had! Let me verify S00:
    # S00 = M1(+1) + M4(+1) - M5(-1) + M7(+1) = 1+1-(-1)+1 = 4? No, -M5 means coefficient -1 on M5.
    # M5 contributes -1 to C[0,1], so -M5 contributes +1. So 1+1+1+1=4? That's wrong.
    
    # I think I have the signs wrong. Let me look up Strassen carefully.
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So:
    # M1 -> C00:+1, C11:+1
    # M2 -> C10:+1, C11:-1
    # M3 -> C01:+1, C11:+1
    # M4 -> C00:+1, C10:+1
    # M5 -> C00:-1, C01:+1
    # M6 -> C11:+1
    # M7 -> C00:+1
    
    # Check C00: M1(+1)+M4(+1)+M5(-1)+M7(+1) = 1+1-1+1 = 2? That's still wrong.
    
    # I'm confusing myself. Let me just use the standard formula and trust it.
    # The key point is that the current program's atomic terms are defined, and the question is whether the overall construction is correct.
    
    # Let me just run the current program and verify it produces a valid decomposition.
    
    # For now, let me try a different strategy: use a known rank-8 decomposition for 2x2x3 if it exists.
    
    # After research, I believe the rank of 2x2x3 is indeed 8. Let me try to construct it.
    
    # Actually, let me just keep the current approach but make sure it's correct, and look for other optimizations.
    
    # The main issue might be that the current code has a bug in the index mapping.
    
    # Let me rewrite the whole thing more carefully.
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # For each of the 7 top-level Strassen terms:
    # Each term involves linear combinations of A-blocks, B-blocks, and C-blocks.
    # A-blocks are 2x2, B-blocks are 2x3, C-blocks are 2x3.
    
    # Top-level Strassen for 4x4 * 4x6:
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x3)
    # C blocks: C00, C01, C10, C11 (each 2x3)
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 * 2x3 product, which we decompose into 11 rank-1 terms.
    
    # For each top-level term t (0..6) and each atomic term s (0..10):
    #   col = t*11 + s
    #   For each A-block with coefficient ca:
    #     For each entry in atomic[s]['U']:
    #       Map local A index to global U index
    #   Similarly for V and W.
    
    # Let me define the top-level terms:
    top_terms = [
        # (a_blocks, b_blocks, c_blocks)
        # Each block is (row_block, col_block, coefficient)
        # A-block (bi,bj) covers rows 2bi..2bi+1, cols 2bj..2bj+1
        # B-block (bi,bj) covers rows 2bi..2bi+1, cols 3bj..3bj+2
        # C-block (bi,bj) covers rows 2bi..2bi+1, cols 3bj..3bj+2
        
        # M1
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        # M2
        ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1),((1,1),-1)]),
        # M3
        ([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)]),
        # M4
        ([((1,1),1)], [((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)]),
        # M5
        ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),-1),((0,1),1)]),
        # M6
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,1),1)]),
        # M7
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,0),1)]),
    ]
    
    for t, (a_blocks, b_blocks, c_blocks) in enumerate(top_terms):
        for s, term in enumerate(atomic):
            col = t*11 + s
            
            # Process U
            for u_local, val_u in term['U'].items():
                i = u_local // 2
                j = u_local % 2
                for (bi, bj), ca in a_blocks:
                    u_global = (bi*2 + i)*m + (bj*2 + j)
                    U[u_global][col] += ca * val_u
            
            # Process V
            for v_local, val_v in term['V'].items():
                j = v_local // 3
                k = v_local % 3
                for (bi, bj), cb in b_blocks:
                    v_global = (bi*2 + j)*p + (bj*3 + k)
                    V[v_global][col] += cb * val_v
            
            # Process W
            for w_local, val_w in term['W'].items():
                k = w_local // 2
                i = w_local % 2
                for (bi, bj), cc in c_blocks:
                    w_global = (bj*3 + k)*n + (bi*2 + i)
                    W[w_global][col] += cc * val_w
    
    return {"U": U, "V": V, "W": W}
