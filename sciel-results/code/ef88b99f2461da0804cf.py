"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42  # 6 blocks * 7 Strassen steps
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Local coefficient vectors for 2x2x2 Strassen steps
    # U, V, W are 4-element vectors corresponding to indices 00, 01, 10, 11
    strassen_steps = [
        { # P1
            'U': [1, 0, 0, 0],
            'V': [0, 1, 0, -1],
            'W': [0, 1, 0, 1]
        },
        { # P2
            'U': [1, 0, 1, 0],
            'V': [0, 0, 0, 1],
            'W': [0, 0, 1, 0]
        },
        { # P3
            'U': [-1, 0, 1, 0],
            'V': [1, 0, 0, 0],
            'W': [0, 0, 1, -1]
        },
        { # P4
            'U': [0, 0, 0, 1],
            'V': [1, 0, -1, 0],
            'W': [1, 0, 0, 1]
        },
        { # P5
            'U': [1, 1, 0, 0],
            'V': [1, 0, 0, 0],
            'W': [1, 0, 0, 1]
        },
        { # P6
            'U': [0, 0, 1, 1],
            'V': [-1, 0, 0, 1],
            'W': [-1, 1, 0, 0]
        },
        { # P7
            'U': [0, 1, 0, -1],
            'V': [1, 0, 0, 1],
            'W': [1, 0, 0, 0]
        }
    ]
    
    col = 0
    # u: block index for A columns (0, 1) and B rows (0, 1)
    # v: block index for B columns (0, 1, 2)
    for u in range(2):
        for v in range(3):
            for step in strassen_steps:
                # Map local indices to global indices
                # Local A index la in {0,1,2,3} -> global U row
                # la = a*2 + b
                # Global U row = i*m + j = a*4 + (2u+b)
                # Local B index lb in {0,1,2,3} -> global V row
                # lb = b*2 + c
                # Global V row = j*p + k = (2u+b)*6 + (2v+c)
                # Local C index lc in {0,1,2,3} -> global W row
                # lc = a*2 + c
                # Global W row = k*n + i = (2v+c)*2 + a
                
                # U contributions
                for la in range(4):
                    if step['U'][la] != 0:
                        a = la // 2
                        b = la % 2
                        global_u_row = a * m + (2 * u + b)
                        U[global_u_row][col] += step['U'][la]
                
                # V contributions
                for lb in range(4):
                    if step['V'][lb] != 0:
                        b = lb // 2
                        c = lb % 2
                        global_v_row = (2 * u + b) * p + (2 * v + c)
                        V[global_v_row][col] += step['V'][lb]
                
                # W contributions
                for lc in range(4):
                    if step['W'][lc] != 0:
                        a = lc // 2
                        c = lc % 2
                        global_w_row = (2 * v + c) * n + a
                        W[global_w_row][col] += step['W'][lc]
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
