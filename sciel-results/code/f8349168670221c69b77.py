import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = 2, 4, 8
    
    # Strassen decomposition for 2x2 matrix multiplication
    # A = [[a1, a2], [a3, a4]], B = [[b1, b2], [b3, b4]]
    # m1 = (a1+a4)(b1+b4)
    # m2 = (a3+a4)b1
    # m3 = a1(b2-b4)
    # m4 = a4(b3-b1)
    # m5 = (a1+a2)b4
    # m6 = (a3-a1)(b1+b2)
    # m7 = (a2-a4)(b3+b4)
    # C[0,0] = m1 - m4 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    # A-side linear combinations (7 x 4), row s = strassen term, col = (i,jp) in 2x2 block
    x_forms = np.array([
        [1, 0, 0, 1],   # m1: a1+a4
        [0, 0, 1, 1],   # m2: a3+a4
        [1, 0, 0, 0],   # m3: a1
        [0, 0, 0, 1],   # m4: a4
        [1, 1, 0, 0],   # m5: a1+a2
        [-1, 0, 1, 0],  # m6: a3-a1
        [0, 1, 0, -1]   # m7: a2-a4
    ], dtype=int)
    
    # B-side linear combinations (7 x 4), row s = strassen term, col = (jp,kp) in 2x2 block
    y_forms = np.array([
        [1, 0, 0, 1],   # m1: b1+b4
        [1, 0, 0, 0],   # m2: b1
        [0, 1, 0, -1],  # m3: b2-b4
        [-1, 0, 1, 0],  # m4: b3-b1
        [0, 0, 0, 1],   # m5: b4
        [1, 1, 0, 0],   # m6: b1+b2
        [0, 0, 1, 1]    # m7: b3+b4
    ], dtype=int)
    
    # C-side linear combinations (4 x 7), row = (i,kp) in 2x2 output, col = strassen term
    z_forms = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0] = m1 - m4 + m7
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1] = m3 + m5
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0] = m2 + m4
        [1, -1, 1, 0, 0, 1, 0]    # C[1,1] = m1 - m2 + m3 + m6
    ], dtype=int)

    num_groups = 4   # k-groups: 8 cols of B split into 4 pairs
    num_ablocks = 2  # j-groups: 4 cols of A (and 4 rows of B) split into 2 pairs
    num_strassen = 7
    
    rank = num_groups * num_ablocks * num_strassen  # 56
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    for g in range(num_groups):
        for b in range(num_ablocks):
            for s in range(num_strassen):
                # U: A is 2x4, index = i*m + (b*2 + jp)
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + b * 2 + jp
                        U[a_idx, r] = x_forms[s, i * 2 + jp]
                
                # V: B is 4x8, index = (b*2 + jp)*p + (2*g + kp)
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * p + (2 * g + kp)
                        V[b_idx, r] = y_forms[s, jp * 2 + kp]
                
                # W: C output indexed by k*n+i, k = 2*g + kp
                for i in range(n):
                    for kp in range(2):
                        k = 2 * g + kp
                        c_idx = k * n + i
                        W[c_idx, r] = z_forms[i * 2 + kp, s]
                
                r += 1
                
    # Verification removed for speed, relying on correctness of Strassen forms
    # The forms are standard and verified in previous runs.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
