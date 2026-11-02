import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS

    # Schoolbook 2x2 (rank 4)
    U2 = np.eye(4, dtype=int)
    V2 = np.eye(4, dtype=int)
    W2 = np.eye(4, dtype=int)

    # Schoolbook 3x3 (rank 9)
    U3 = np.eye(9, dtype=int)
    V3 = np.eye(9, dtype=int)
    W3 = np.eye(9, dtype=int)

    rank6 = 4 * 9  # 36
    U6 = np.zeros((36, rank6), dtype=int)
    V6 = np.zeros((36, rank6), dtype=int)
    W6 = np.zeros((36, rank6), dtype=int)

    for t2 in range(4):
        for t3 in range(9):
            t = t2 * 9 + t3
            
            # U6
            u2_val = U2[0, t2] # Since U2 is identity, u2_col is 1 at t2
            # Actually, U2[:, t2] is a vector with 1 at t2.
            # u2_col[i2*2+k2] is 1 if i2*2+k2 == t2.
            
            # We can just fill U6, V6, W6 based on the identity structure.
            # U6 row (i,k) corresponds to t2 = i2*2+k2, t3 = i3*3+k3.
            # So U6[u_row, t] = 1 if u_row matches the indices for t.
            
            pass

    # Simpler construction:
    # The Kronecker product of two identity matrices is an identity matrix.
    # The rows of U6 are indexed by (i,k).
    # i = 2*i2 + i3, k = 2*k2 + k3.
    # t = t2*9 + t3, where t2 = i2*2+k2, t3 = i3*3+k3.
    # So for a given row u_row, we can find i, k.
    # Then i2 = i//2, i3 = i%2? No.
    # i = 2*i2 + i3. i in 0..5.
    # i2 = i // 2? No.
    # The indexing in the previous code was:
    # i = 2*i2 + i3.
    # So i2 = i // 2, i3 = i % 2?
    # If i=0, i2=0, i3=0.
    # If i=1, i2=0, i3=1.
    # If i=2, i2=1, i3=0.
    # If i=3, i2=1, i3=1.
    # If i=4, i2=2, i3=0.
    # If i=5, i2=2, i3=1.
    # This works if i2 in 0..2? No, i2 in 0..1 for 2x2.
    # Wait, 2x2 means i2 in 0..1.
    # But i goes up to 5.
    # So i = 2*i2 + i3 is NOT a bijection for i in 0..5 and i2 in 0..1, i3 in 0..2.
    # i2 in 0..1, i3 in 0..2.
    # i = 2*i2 + i3.
    # i2=0: i=0,1,2.
    # i2=1: i=2,3,4.
    # Collision at i=2.
    # So the indexing in the previous code `i = 2*i2 + i3` is INVALID for 2x3 split.
    # It should be `i = 3*i2 + i3`? No.
    # We are splitting 6 into 2 and 3.
    # So $i = i_2 \cdot 3 + i_3$?
    # If $i_2 \in \{0,1\}$ and $i_3 \in \{0,1,2\}$.
    # $i = i_2 \cdot 3 + i_3$.
    # $i_2=0: 0,1,2$.
    # $i_2=1: 3,4,5$.
    # This is a valid bijection.
    
    # The previous code used `i = 2*i2 + i3`.
    # This is wrong for dimensions 2 and 3.
    # It should be `i = 3*i2 + i3`.
    
    # Let's rebuild with correct indexing.
    # i = 3*i2 + i3
    # k = 3*k2 + k3
    # j = 3*j2 + j3
    
    # U row: i*6+k = (3*i2+i3)*6 + (3*k2+k3)
    # V row: k*6+j = (3*k2+k3)*6 + (3*j2+j3)
    # W row: j*6+i = (3*j2+j3)*6 + (3*i2+i3)
    
    # t = t2*9 + t3
    # t2 = i2*2+k2 (index in 2x2)
    # t3 = i3*3+k3 (index in 3x3)
    
    for i2 in range(2):
        for i3 in range(3):
            for k2 in range(2):
                for k3 in range(3):
                    i = 3*i2 + i3
                    k = 3*k2 + k3
                    u_row = i*6 + k
                    t2 = i2*2 + k2
                    t3 = i3*3 + k3
                    t = t2*9 + t3
                    U6[u_row, t] = 1
                    
    for k2 in range(2):
        for k3 in range(3):
            for j2 in range(2):
                for j3 in range(3):
                    k = 3*k2 + k3
                    j = 3*j2 + j3
                    v_row = k*6 + j
                    t2 = k2*2 + j2
                    t3 = k3*3 + j3
                    t = t2*9 + t3
                    V6[v_row, t] = 1
                    
    for j2 in range(2):
        for j3 in range(3):
            for i2 in range(2):
                for i3 in range(3):
                    j = 3*j2 + j3
                    i = 3*i2 + i3
                    w_row = j*6 + i
                    t2 = j2*2 + i2
                    t3 = j3*3 + i3
                    t = t2*9 + t3
                    W6[w_row, t] = 1

    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
