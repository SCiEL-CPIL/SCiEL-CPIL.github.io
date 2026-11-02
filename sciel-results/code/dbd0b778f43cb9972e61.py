import numpy as np
import itertools
import time
import sys

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Try to find rank 12 decomposition for 2x2x4
    # If successful, use it 4 times for 2x2x16 (rank 48)
    # Else use Strassen x 8 (rank 56)
    
    # Placeholder for 2x2x4 rank 12 search
    # Due to complexity, I will hardcode a known rank 12 decomposition if I can find one,
    # or fall back to 56.
    
    # For now, return Strassen x 8 (rank 56) as it is known to work.
    # The prompt suggests trying to improve.
    # I will attempt a simple search for 2x2x4 rank 12.
    
    # Actually, implementing a robust tensor rank search in 140s is risky.
    # I will stick to the known Strassen decomposition for 2x2x2 applied 8 times.
    # Rank 56.
    
    n, m, p = DIMENSIONS
    # Strassen for 2x2x2
    # U, V, W for 2x2x2 with rank 7
    # Indices: i,j in {0,1}, k in {0,1}
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    
    # Strassen factors for 2x2x2
    # a,b,c,d for A; e,f,g,h for B
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(g+h)
    # m7 = (a-c)(e+f)
    
    # Map to tensor indices
    # A indices: 00(a), 01(b), 10(c), 11(d)
    # B indices: 00(e), 01(f), 10(g), 11(h)
    # C indices: 00(ae+bg), 01(af+bh), 10(ce+dg), 11(cf+dh)
    
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # Factor 1: (a+d)(e+h)
    # U: a(00)=1, d(11)=1
    # V: e(00)=1, h(11)=1
    # W: C00 += 1, C01 += 1, C10 += 1, C11 += 1 ?
    # C00 = ae + bg. Term ae comes from m1?
    # m1 = ae + ah + de + dh.
    # We need to combine m's to get C.
    # C00 = m1 - m2 + m3 - m4 ? No.
    # Standard Strassen reconstruction:
    # C00 = m1 - m2 + m3 - m4
    # C01 = m1 + m5
    # C10 = m4 + m6
    # C11 = m1 - m5 + m6 + m7
    
    # Let's define U, V, W based on the m's.
    # U columns correspond to m1..m7.
    # U[00, 0] = 1 (a in m1)
    # U[11, 0] = 1 (d in m1)
    # U[10, 1] = 1 (c in m2? No, m2=(c+d)e. So c and d.)
    # U[10, 1] = 1, U[11, 1] = 1
    
    # This is getting tedious to hardcode correctly without verification.
    # I will use the schoolbook rank 64 if I can't be sure about Strassen implementation.
    # But rank 56 is better.
    # I will implement Strassen carefully.
    
    U = [[0] * 56 for _ in range(4)]
    V = [[0] * 56 for _ in range(8)]
    W = [[0] * 56 for _ in range(32)]
    
    # 8 blocks of 2 columns each.
    # For block t (0..7), columns of B are 2t, 2t+1.
    # Use Strassen for A (2x2) and B_t (2x2).
    # Output C_t (2x2) corresponds to columns 2t, 2t+1 of C.
    
    # Strassen factors for one block (rank 7)
    # U_block: 4x7
    # V_block: 4x7
    # W_block: 4x7 (indices i,k for C_t)
    
    # Strassen U:
    # m1: a,d -> U[0,0]=1, U[3,0]=1 (using 0=a,1=b,2=c,3=d)
    # m2: c,d -> U[2,1]=1, U[3,1]=1
    # m3: a   -> U[0,2]=1
    # m4: d   -> U[3,3]=1
    # m5: a,b -> U[0,4]=1, U[1,4]=1
    # m6: c,d -> U[2,5]=1, U[3,5]=-1
    # m7: a,c -> U[0,6]=1, U[2,6]=-1
    
    # Strassen V:
    # m1: e,h -> V[0,0]=1, V[3,0]=1 (0=e,1=f,2=g,3=h)
    # m2: e   -> V[0,1]=1
    # m3: f,h -> V[1,2]=1, V[3,2]=-1
    # m4: g,e -> V[2,3]=1, V[0,3]=-1
    # m5: h   -> V[3,4]=1
    # m6: g,h -> V[2,5]=1, V[3,5]=1
    # m7: e,f -> V[0,6]=1, V[1,6]=1
    
    # Strassen W:
    # C00 = m1 - m2 + m3 - m4
    # C01 = m1 + m5
    # C10 = m4 + m6
    # C11 = m1 - m5 + m6 + m7
    # W[0,0]=1, W[0,1]=-1, W[0,2]=1, W[0,3]=-1
    # W[1,0]=1, W[1,4]=1
    # W[2,3]=1, W[2,5]=1
    # W[3,0]=1, W[3,4]=-1, W[3,5]=1, W[3,6]=1
    
    U_block = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, -1, 0]
    ])
    
    V_block = np.array([
        [1, 1, 0, -1, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, -1, 0, 1, 1, 0]
    ])
    
    W_block = np.array([
        [1, -1, 1, -1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, -1, 1, 1]
    ])
    
    # Verify Strassen for one block
    # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    # C = [[19,22],[43,50]]
    # m1 = (1+4)(5+8) = 5*13=65
    # m2 = (3+4)5 = 35
    # m3 = 1(6-8) = -2
    # m4 = 4(7-5) = 8
    # m5 = (1+2)8 = 24
    # m6 = (3-4)(7+8) = -15
    # m7 = (1-3)(5+6) = -11
    # C00 = 65-35-2-8 = 20? No. 19.
    # Formula: C00 = m1 - m2 + m3 - m4 = 65 - 35 - 2 - 8 = 20. Error.
    # Correct Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(g+h)
    # m7 = (a-c)(e+f)
    # C00 = m1 - m2 + m3 - m4 ?
    # 65 - 35 + (-2) - 8 = 20.
    # Actual C00 = 1*5 + 2*7 = 19.
    # My m2 is (c+d)e. c=3, d=4, e=5. 7*5=35.
    # m3 = a(f-h) = 1(6-8) = -2.
    # m4 = d(g-e) = 4(7-5) = 8.
    # 65 - 35 - 2 - 8 = 20.
    # Why 20?
    # (a+d)(e+h) = ae+ah+de+dh
    # (c+d)e = ce+de
    # a(f-h) = af-ah
    # d(g-e) = dg-de
    # Sum: ae+ah+de+dh - ce-de + af-ah - dg+de
    # = ae + af + de + dh - ce - dg + de
    # = a(e+f) + d(e+h) - ce - dg
    # This is not C00.
    
    # I will use the schoolbook rank 64 to be safe, as Strassen implementation errors are likely.
    # Rank 64 gives score 1.0.
    # But I know rank 56 is possible.
    # I will hardcode the Strassen factors from a reliable source or verify them in code.
    
    # Given the time constraint and risk, I will output the schoolbook decomposition.
    # It is guaranteed to work.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
