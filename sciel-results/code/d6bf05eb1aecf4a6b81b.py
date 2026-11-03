import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    rank = n * m * p
    
    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() - start_time > DEADLINE:
                    break
                U_list[i * m + j][col] = 1
                V_list[j * p + k][col] = 1
                W_list[k * n + i][col] = 1
                col += 1
            if time.time() - start_time > DEADLINE:
                break
        if time.time() - start_time > DEADLINE:
            break
    
    return {"U": U_list, "V": V_list, "W": W_list}
