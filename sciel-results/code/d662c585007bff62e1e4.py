import numpy as np
import time

N = 400

def compute_autocorr(x):
    R = np.zeros(N)
    for k in range(N):
        R[k] = np.dot(x[k:], x[:N-k])
    return R

def thue_morse(n):
    x = np.zeros(n, dtype=np.int8)
    for i in range(n):
        x[i] = 1 - 2 * (bin(i).count('1') % 2)
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    x = thue_morse(N)
    R = compute_autocorr(x)
    best_x = x.copy()
    best_R = R.copy()
    best_score = np.max(np.abs(R[1:]))
    
    while time.time() < deadline:
        best_j = -1
        best_new_score = best_score
        best_new_R = None
        
        for j in range(N):
            xj = x[j]
            left = np.zeros(N, dtype=np.int8)
            right = np.zeros(N, dtype=np.int8)
            left[:j+1] = x[j::-1]
            right[:N-j] = x[j:]
            delta = -2 * xj * (left + right)
            R_new = R + delta
            score = np.max(np.abs(R_new[1:]))
            if score < best_new_score:
                best_new_score = score
                best_j = j
                best_new_R = R_new
            if time.time() > deadline:
                break
        
        if best_j >= 0:
            x[best_j] = -x[best_j]
            R = best_new_R
            best_x = x.copy()
            best_R = R.copy()
            best_score = best_new_score
        else:
            break
    
    return best_x
