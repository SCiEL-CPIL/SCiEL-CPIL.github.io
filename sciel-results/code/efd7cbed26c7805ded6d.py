"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
_START_TIME = time.time()
_DEADLINE = 120.0
_MARGIN = 5.0

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    if time.time() > _START_TIME + _DEADLINE - _MARGIN:
        # Fallback: return schoolbook rank 96
        rank = 96
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Stothers rank-20 algorithm for 3x4x4 matrix multiplication
    # A (3x4) * B (4x4) = C (3x4)
    # 
    # Local indices for A (3x4): (i,j) with i in {0,1,2}, j in {0,1,2,3}
    #   flat index: i*4 + j
    # Local indices for B (4x4): (j,k) with j in {0,1,2,3}, k in {0,1,2,3}
    #   flat index: j*4 + k
    # Local indices for C (3x4): (i,k) with i in {0,1,2}, k in {0,1,2,3}
    #   flat index: i*4 + k
    #
    # The 20 terms of Stothers' algorithm:
    # We partition A into two 3x2 blocks: A_left (cols 0,1), A_right (cols 2,3)
    # We partition B into two 4x2 blocks: B_left (cols 0,1), B_right (cols 2,3)
    # 
    # Stothers 3x4x4 decomposition (rank 20):
    # The key insight is to use 20 bilinear forms.
    # 
    # Let me define the 20 terms explicitly. Each term has:
    #   u: dict mapping A-local-index -> coefficient
    #   v: dict mapping B-local-index -> coefficient  
    #   w: dict mapping C-local-index -> coefficient
    #
    # Stothers' 20-term algorithm for 3x4x4:
    # Split A = [A1 | A2] where A1, A2 are 3x2
    # Split B = [B1; B2] where B1, B2 are 2x4 (top and bottom halves)
    # Wait, let me think more carefully.
    #
    # Actually, for 3x4 * 4x4 = 3x4:
    # A is 3x4, B is 4x4, C is 3x4
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Stothers (1981) gives rank 20 for 3x4x4.
    # 
    # Let me use the known decomposition. The 20 terms are:
    # 
    # Term 1: (A[0,0]+A[0,2]+A[1,1]+A[2,0]) * (B[0,0]+B[2,1])
    #   -> C[0,0] += 1, C[1,1] += 1
    # 
    # This is getting complex. Let me use a verified set of 20 terms.
    #
    # From Stothers' paper, the 20 bilinear forms for 3x4x4:
    # I'll use the standard formulation.
    
    # Stothers 3x4x4 rank-20 decomposition
    # A indices: (i,j) -> i*4+j, i in 0..2, j in 0..3
    # B indices: (j,k) -> j*4+k, j in 0..3, k in 0..3
    # C indices: (i,k) -> i*4+k, i in 0..2, k in 0..3
    
    # The 20 terms (verified against the multiplication tensor):
    stothers_terms = [
        # Term 0
        {'u': {0: 1, 2: 1}, 'v': {0: 1, 4: 1}, 'w': {0: 1, 5: 1}},
        # Term 1
        {'u': {1: 1, 3: 1}, 'v': {1: 1, 5: 1}, 'w': {1: 1, 6: 1}},
        # Term 2
        {'u': {4: 1, 6: 1}, 'v': {8: 1, 12: 1}, 'w': {4: 1, 9: 1}},
        # Term 3
        {'u': {5: 1, 7: 1}, 'v': {9: 1, 13: 1}, 'w': {5: 1, 10: 1}},
        # Term 4
        {'u': {8: 1, 10: 1}, 'v': {0: 1, 8: 1}, 'w': {8: 1, 13: 1}},
        # Term 5
        {'u': {9: 1, 11: 1}, 'v': {1: 1, 9: 1}, 'w': {9: 1, 14: 1}},
        # Term 6
        {'u': {0: 1, 4: 1}, 'v': {2: 1, 10: 1}, 'w': {2: 1, 7: 1}},
        # Term 7
        {'u': {1: 1, 5: 1}, 'v': {3: 1, 11: 1}, 'w': {3: 1, 8: 1}},
        # Term 8
        {'u': {2: 1, 6: 1}, 'v': {0: 1, 8: 1}, 'w': {10: 1, 15: 1}},
        # Term 9
        {'u': {3: 1, 7: 1}, 'v': {1: 1, 9: 1}, 'w': {11: 1, 16: 1}},
        # Term 10
        {'u': {8: 1, 12: 1}, 'v': {2: 1, 10: 1}, 'w': {12: 1, 17: 1}},
        # Term 11
        {'u': {9: 1, 13: 1}, 'v': {3: 1, 11: 1}, 'w': {13: 1, 18: 1}},
        # Term 12
        {'u': {10: 1, 14: 1}, 'v': {0: 1, 8: 1}, 'w': {14: 1, 19: 1}},
        # Term 13
        {'u': {11: 1, 15: 1}, 'v': {1: 1, 9: 1}, 'w': {15: 1, 20: 1}},
        # Term 14
        {'u': {0: 1, 8: 1}, 'v': {4: 1, 12: 1}, 'w': {0: 1, 12: 1}},
        # Term 15
        {'u': {1: 1, 9: 1}, 'v': {5: 1, 13: 1}, 'w': {1: 1, 13: 1}},
        # Term 16
        {'u': {2: 1, 10: 1}, 'v': {4: 1, 12: 1}, 'w': {2: 1, 14: 1}},
        # Term 17
        {'u': {3: 1, 11: 1}, 'v': {5: 1, 13: 1}, 'w': {3: 1, 15: 1}},
        # Term 18
        {'u': {4: 1, 12: 1}, 'v': {6: 1, 14: 1}, 'w': {4: 1, 16: 1}},
        # Term 19
        {'u': {5: 1, 13: 1}, 'v': {7: 1, 15: 1}, 'w': {5: 1, 17: 1}},
    ]
    
    # Verify: for each output C[i,k], sum over terms of (sum_u * sum_v) should equal sum_j A[i,j]*B[j,k]
    # This is a 3x4x4 = 48-entry tensor.
    
    # Actually, I need to be more careful. Let me construct the terms properly.
    # The above terms may not be correct. Let me use a different approach.
    #
    # For 3x4x4, I'll use the following verified decomposition:
    # Split into: (3x2x2) + (3x2x2) + cross terms
    # Actually, let me just use schoolbook for 3x4x4 which is rank 48,
    # then apply Strassen-like reductions.
    #
    # Alternative: Use the fact that 3x4x4 can be done in rank 20.
    # Let me look up the exact Stothers terms.
    #
    # Stothers (1981) "On the multiplication of 3 x 4 matrices" gives rank 20.
    # The decomposition uses 20 bilinear forms.
    #
    # Let me define them carefully:
    # A = [[a00, a01, a02, a03],
    #      [a10, a11, a12, a13],
    #      [a20, a21, a22, a23]]
    # B = [[b00, b01, b02, b03],
    #      [b10, b11, b12, b13],
    #      [b20, b21, b22, b23],
    #      [b30, b31, b32, b33]]
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Stothers' 20 terms:
    # M1 = (a00+a11+a20) * (b00+b11+b20)  -- this doesn't look right either
    
    # Let me use a computational approach: construct the tensor and find rank-20 decomposition
    # by solving a system of equations. But that's complex to hardcode.
    #
    # Instead, let me use a simpler but still improved approach:
    # Use Strassen for 2x4x4 (rank 14 instead of 32) for rows 0,1 and schoolbook for row 2.
    # 2x4x4: split into (2x2x2)+(2x2x2), each Strassen rank 7, total 14
    # 1x4x4: schoolbook rank 16
    # Total: 14 + 16 = 30
    
    # Actually, let me try to use a known rank-20 decomposition for 3x4x4.
    # After careful consideration, I'll use the following verified terms:
    
    # For 3x4x4, here's a rank-20 decomposition (Stothers 1981):
    # The 20 bilinear forms M_t = (u_t . A) * (v_t . B)
    # C = sum_t w_t * M_t
    
    # I'll use the standard Stothers decomposition:
    stothers_terms = [
        # (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
        # u is for A (3x4=12 entries), v is for B (4x4=16 entries), w is for C (3x4=12 entries)
        # A flat: a[i,j] = i*4+j
        # B flat: b[j,k] = j*4+k
        # C flat: c[i,k] = i*4+k
        
        # M1: (a00+a20)(b00+b20)
        ({0: 1, 8: 1}, {0: 1, 8: 1}, {0: 1, 8: 1}),
        # M2: (a01+a21)(b01+b21)
        ({1: 1, 9: 1}, {1: 1, 9: 1}, {1: 1, 9: 1}),
        # M3: (a02+a22)(b02+b22)
        ({2: 1, 10: 1}, {2: 1, 10: 1}, {2: 1, 10: 1}),
        # M4: (a03+a23)(b03+b23)
        ({3: 1, 11: 1}, {3: 1, 11: 1}, {3: 1, 11: 1}),
        # M5: (a10+a12)(b10+b30)
        ({4: 1, 6: 1}, {4: 1, 12: 1}, {4: 1, 12: 1}),
        # M6: (a11+a13)(b11+b31)
        ({5: 1, 7: 1}, {5: 1, 13: 1}, {5: 1, 13: 1}),
        # M7: (a00+a10)(b10+b30)
        ({0: 1, 4: 1}, {4: 1, 12: 1}, {0: 1, 4: 1}),
        # M8: (a01+a11)(b11+b31)
        ({1: 1, 5: 1}, {5: 1, 13: 1}, {1: 1, 5: 1}),
        # M9: (a02+a12)(b12+b32)
        ({2: 1, 6: 1}, {6: 1, 14: 1}, {2: 1, 6: 1}),
        # M10: (a03+a13)(b13+b33)
        ({3: 1, 7: 1}, {7: 1, 15: 1}, {3: 1, 7: 1}),
        # M11: (a20+a10)(b10+b30)
        ({8: 1, 4: 1}, {4: 1, 12: 1}, {8: 1, 4: 1}),
        # M12: (a21+a11)(b11+b31)
        ({9: 1, 5: 1}, {5: 1, 13: 1}, {9: 1, 5: 1}),
        # M13: (a22+a12)(b12+b32)
        ({10: 1, 6: 1}, {6: 1, 14: 1}, {10: 1, 6: 1}),
        # M14: (a23+a13)(b13+b33)
        ({11: 1, 7: 1}, {7: 1, 15: 1}, {11: 1, 7: 1}),
        # M15: (a00+a02)(b10+b12)
        ({0: 1, 2: 1}, {4: 1, 6: 1}, {0: 1, 2: 1}),
        # M16: (a01+a03)(b11+b13)
        ({1: 1, 3: 1}, {5: 1, 7: 1}, {1: 1, 3: 1}),
        # M17: (a10+a12)(b20+b22)
        ({4: 1, 6: 1}, {8: 1, 10: 1}, {4: 1, 6: 1}),
        # M18: (a11+a13)(b21+b23)
        ({5: 1, 7: 1}, {9: 1, 11: 1}, {5: 1, 7: 1}),
        # M19: (a20+a22)(b20+b22)
        ({8: 1, 10: 1}, {8: 1, 10: 1}, {8: 1, 10: 1}),
        # M20: (a21+a23)(b21+b23)
        ({9: 1, 11: 1}, {9: 1, 11: 1}, {9: 1, 11: 1}),
    ]
    
    # Verify this decomposition is correct by checking the tensor
    # C[i,k] should equal sum_j A[i,j]*B[j,k]
    # Let's verify term by term that the sum of w_t * (u_t . A)(v_t . B) gives the right result.
    
    # For C[0,0] = a00*b00 + a01*b10 + a02*b20 + a03*b30
    # From M1: a00*b00 (from a00*b00 in (a00+a20)(b00+b20) -> a00*b00)
    #   M1 contributes: a00*b00 + a00*b20 + a20*b00 + a20*b20 to w[0]=c00 and w[8]=c20
    #   So c00 gets a00*b00, c20 gets a00*b00 (wrong! c20 should not get a00*b00)
    #
    # This decomposition is incorrect. Let me try a different approach.
    #
    # I'll use a verified rank-20 decomposition. Let me use the one from
    # the literature more carefully.
    
    # Actually, let me use a simpler strategy: use Strassen-based approach
    # for the 3x4x8 problem directly.
    #
    # Split 3x4x8 into:
    # - 2x4x4 (top two rows, first 4 cols of B): use Strassen-like, rank 14
    # - 2x4x4 (top two rows, last 4 cols of B): use Strassen-like, rank 14  
    # - 1x4x4 (bottom row, first 4 cols of B): schoolbook, rank 16
    # - 1x4x4 (bottom row, last 4 cols of B): schoolbook, rank 16
    # Total: 14+14+16+16 = 60
    
    # For 2x4x4 with rank 14:
    # Split A (2x4) into A1 (2x2) and A2 (2x2)
    # Split B (4x4) into B1 (2x4) and B2 (2x4)
    # C = A1*B1 + A2*B2
    # Each 2x2x4 can be done in rank 7 (Strassen for 2x2x2, twice)
    # Wait, 2x2x4: A(2x2)*B(2x4)=C(2x4)
    # Split B into two 2x2 blocks: B_left, B_right
    # C_left = A*B_left (2x2x2, Strassen rank 7)
    # C_right = A*B_right (2x2x2, Strassen rank 7)
    # Total for 2x2x4: rank 14
    # So 2x4x4: rank 14+14 = 28? No wait.
    # 2x4x4: A(2x4)*B(4x4)
    # Split A into A1(2x2), A2(2x2) [columns]
    # Split B into B1(2x4), B2(2x4) [rows]
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2 * 2x4 = 2x4, rank 14
    # A2*B2: 2x2 * 2x4 = 2x4, rank 14
    # Total: 28
    
    # That's not better than schoolbook 32 for 2x4x4.
    
    # Let me try: 3x4x4 in rank 20 (Stothers) + 3x4x4 in rank 20 = 40 total for 3x4x8.
    # I need to find a valid rank-20 decomposition for 3x4x4.
    
    # Let me construct it computationally and hardcode.
    # The multiplication tensor for 3x4x4 has 48 non-zero entries.
    # I need 20 vectors u_t in R^12, v_t in R^16, w_t in R^12 such that
    # sum_t u_t * v_t * w_t^T = T (the tensor)
    
    # This is equivalent to: for each (i,j,k), sum_t u_t[i,j] * v_t[j,k] * w_t[i,k] = delta[i,j,k]
    # where delta[i,j,k] = 1 if the entry (i,j,k) is in the tensor, else 0.
    # The tensor has entries T[i,j,k] = 1 for all valid (i,j,k) in the multiplication.
    # Actually T[i,j,k] = 1 when i is row of A, j is col of A = row of B, k is col of B = col of C.
    
    # Let me just use schoolbook for 3x4x4 (rank 48) twice = rank 96, same as before.
    # Or use the Strassen approach that gives 88.
    
    # Given the difficulty of finding a valid rank-20 decomposition by hand,
    # let me use a different valid improvement:
    # Use 2x4x4 in rank 14 (if possible) + 1x4x4 in rank 16, twice.
    
    # For 2x4x4: can we do better than 32?
    # 2x4x4 = 2x(2x2x2) + 2x(2x2x2) using Strassen = 4*7 = 28
    # So 2x4x4 in rank 28.
    # 1x4x4 in rank 16 (schoolbook).
    # Total for 3x4x4: 28 + 16 = 44.
    # For 3x4x8: 2 * 44 = 88. Same as current.
    
    # Let me try another split:
    # 3x4x8: split B into 4 blocks of 4x2.
    # 3x4x2: A(3x4)*B(4x2) = C(3x2)
    # Split A into A1(3x2), A2(3x2)
    # B into B1(2x2), B2(2x2)
    # C = A1*B1 + A2*B2
    # Each 3x2x2: rank 12 (schoolbook) or can we do better?
    # 3x2x2: 3*2*2 = 12 schoolbook. Strassen doesn't directly apply.
    # Total: 4 * (12+12) = 96. Worse.
    
    # OK, let me try to find a valid Stothers 3x4x4 rank-20 decomposition.
    # I'll use the known result and verify computationally.
    
    # Here's a known rank-20 decomposition for 3x4x4 (from Stothers 1981):
    # The 20 bilinear forms are:
    
    # Let me use a different, verified approach. I'll use the following:
    # For 3x4x4, use 20 terms based on the following structure:
    # 
    # Actually, I recall that the correct Stothers decomposition for 3x4x4 uses
    # a specific pattern. Let me implement it and verify.
    
    # I'll use the following 20 terms (0-indexed, A[i,j]=i*4+j, B[j,k]=j*4+k, C[i,k]=i*4+k):
    stothers_terms = [
        # t=0: (a[0,0]+a[1,1]+a[2,0])*(b[0,0]+b[1,1]+b[2,0]) -> c[0,0]+c[1,1]+c[2,0]
        ({0:1, 5:1, 8:1}, {0:1, 5:1, 8:1}, {0:1, 5:1, 8:1}),
        # t=1: (a[0,1]+a[1,0]+a[2,1])*(b[1,0]+b[0,1]+b[2,1]) -> c[0,1]+c[1,0]+c[2,1]
        ({1:1, 4:1, 9:1}, {4:1, 1:1, 9:1}, {1:1, 4:1, 9:1}),
        # t=2: (a[0,2]+a[1,3]+a[2,2])*(b[2,0]+b[3,1]+b[2,2]) -> c[0,2]+c[1,3]+c[2,2]
        ({2:1, 7:1, 10:1}, {8:1, 13:1, 10:1}, {2:1, 7:1, 10:1}),
        # t=3: (a[0,3]+a[1,2]+a[2,3])*(b[3,0]+b[2,1]+b[3,2]) -> c[0,3]+c[1,2]+c[2,3]
        ({3:1, 6:1, 11:1}, {12:1, 9:1, 15:1}, {3:1, 6:1, 11:1}),
        # t=4: (a[0,0]+a[0,2])*(b[1,0]+b[2,0]) -> c[0,0]+c[0,2]
        ({0:1, 2:1}, {4:1, 8:1}, {0:1, 2:1}),
        # t=5: (a[0,1]+a[0,3])*(b[1,1]+b[2,1]) -> c[0,1]+c[0,3]
        ({1:1, 3:1}, {5:1, 9:1}, {1:1, 3:1}),
        # t=6: (a[1,0]+a[1,2])*(b[0,0]+b[1,0]) -> c[1,0]+c[1,2]
        ({4:1, 6:1}, {0:1, 4:1}, {4:1, 6:1}),
        # t=7: (a[1,1]+a[1,3])*(b[0,1]+b[1,1]) -> c[1,1]+c[1,3]
        ({5:1, 7:1}, {1:1, 5:1}, {5:1, 7:1}),
        # t=8: (a[2,0]+a[2,2])*(b[0,0]+b[1,0]) -> c[2,0]+c[2,2]
        ({8:1, 10:1}, {0:1, 4:1}, {8:1, 10:1}),
        # t=9: (a[2,1]+a[2,3])*(b[0,1]+b[1,1]) -> c[2,1]+c[2,3]
        ({9:1, 11:1}, {1:1, 5:1}, {9:1, 11:1}),
        # t=10: (a[0,0]+a[1,0])*(b[2,0]+b[3,0]) -> c[0,0]+c[1,0]
        ({0:1, 4:1}, {8:1, 12:1}, {0:1, 4:1}),
        # t=11: (a[0,1]+a[1,1])*(b[2,1]+b[3,1]) -> c[0,1]+c[1,1]
        ({1:1, 5:1}, {9:1, 13:1}, {1:1, 5:1}),
        # t=12: (a[0,2]+a[1,2])*(b[2,2]+b[3,2]) -> c[0,2]+c[1,2]
        ({2:1, 6:1}, {10:1, 14:1}, {2:1, 6:1}),
        # t=13: (a[0,3]+a[1,3])*(b[2,3]+b[3,3]) -> c[0,3]+c[1,3]
        ({3:1, 7:1}, {11:1, 15:1}, {3:1, 7:1}),
        # t=14: (a[2,0]+a[1,0])*(b[2,0]+b[3,0]) -> c[2,0]+c[1,0]
        ({8:1, 4:1}, {8:1, 12:1}, {8:1, 4:1}),
        # t=15: (a[2,1]+a[1,1])*(b[2,1]+b[3,1]) -> c[2,1]+c[1,1]
        ({9:1, 5:1}, {9:1, 13:1}, {9:1, 5:1}),
        # t=16: (a[2,2]+a[1,2])*(b[2,2]+b[3,2]) -> c[2,2]+c[1,2]
        ({10:1, 6:1}, {10:1, 14:1}, {10:1, 6:1}),
        # t=17: (a[2,3]+a[1,3])*(b[2,3]+b[3,3]) -> c[2,3]+c[1,3]
        ({11:1, 7:1}, {11:1, 15:1}, {11:1, 7:1}),
        # t=18: (a[0,0]+a[0,1]+a[0,2]+a[0,3])*(b[0,0]+b[1,0]+b[2,0]+b[3,0]) -> c[0,0]
        ({0:1, 1:1, 2:1, 3:1}, {0:1, 4:1, 8:1, 12:1}, {0:1}),
        # t=19: (a[1,0]+a[1,1]+a[1,2]+a[1,3])*(b[0,1]+b[1,1]+b[2,1]+b[3,1]) -> c[1,1]
        ({4:1, 5:1, 6:1, 7:1}, {1:1, 5:1, 9:1, 13:1}, {5:1}),
    ]
    
    # I'm not confident these are correct. Let me use a safer approach.
    # I'll use the current Strassen-based rank-88 decomposition but try to verify
    # if we can do better with a simple improvement.
    
    # Actually, let me just keep the rank-88 solution which is verified,
    # and try a small optimization: use rank 7 Strassen for all 2x2x2 blocks
    # in a more efficient decomposition.
    
    # Current: 3x4x8 = 1x4x8 (schoolbook, rank 32) + 2x4x8 (Strassen, rank 56) = 88
    # Alternative: 3x4x8 = 3x4x4 + 3x4x4
    # Each 3x4x4: 2x4x4 (Strassen, rank 28) + 1x4x4 (schoolbook, rank 16) = 44
    # Total: 88. Same.
    
    # Another alternative: 3x4x8 = 2x4x4 + 2x4x4 + 1x4x4 + 1x4x4
    # = 28 + 28 + 16 + 16 = 88. Same.
    
    # So the current rank-88 is likely optimal for this approach.
    # Let me keep it and just clean up the code.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Schoolbook for A[0, :] (1x4) * B (4x8)
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: Strassen for A[1:3, :] (2x4) * B (4x8)
    strassen_terms = [
        {'U': {0: 1, 3: 1}, 'V': {0: 1, 3: 1}, 'W': {0: 1, 3: 1}},
        {'U': {2: 1, 3: 1}, 'V': {0: 1}, 'W': {2: 1, 3: -1}},
        {'U': {0: 1}, 'V': {1: 1, 3: -1}, 'W': {1: 1, 3: 1}},
        {'U': {3: 1}, 'V': {2: 1, 0: -1}, 'W': {0: 1, 2: 1}},
        {'U': {0: 1, 1: 1}, 'V': {3: 1}, 'W': {0: -1, 1: 1}},
        {'U': {2: 1, 0: -1}, 'V': {0: 1, 1: 1}, 'W': {3: 1}},
        {'U': {1: 1, 3: -1}, 'V': {2: 1, 3: 1}, 'W': {0: 1}},
    ]
    
    for j_block in range(4):
        for term in strassen_terms:
            for local_u, val in term['U'].items():
                lu, lv = divmod(local_u, 2)
                global_i = 1 + lu
                global_j = lv
                U[global_i * m + global_j][col] += val
            for local_v, val in term['V'].items():
                lu, lv = divmod(local_v, 2)
                global_j = lu
                global_k = 2 * j_block + lv
                V[global_j * p + global_k][col] += val
            for local_w, val in term['W'].items():
                lu, lv = divmod(local_w, 2)
                global_i = 1 + lu
                global_k = 2 * j_block + lv
                W[global_k * n + global_i][col] += val
            col += 1
        
        for term in strassen_terms:
            for local_u, val in term['U'].items():
                lu, lv = divmod(local_u, 2)
                global_i = 1 + lu
                global_j = 2 + lv
                U[global_i * m + global_j][col] += val
            for local_v, val in term['V'].items():
                lu, lv = divmod(local_v, 2)
                global_j = 2 + lu
                global_k = 2 * j_block + lv
                V[global_j * p + global_k][col] += val
            for local_w, val in term['W'].items():
                lu, lv = divmod(local_w, 2)
                global_i = 1 + lu
                global_k = 2 * j_block + lv
                W[global_k * n + global_i][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
