import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for block in range(4):
        base_c = block * 7
        b00_idx = 0 * p + (block * 2)      
        b01_idx = 0 * p + (block * 2 + 1)  
        b10_idx = 1 * p + (block * 2)      
        b11_idx = 1 * p + (block * 2 + 1)  
        
        c00_w_idx = (block * 2) * n + 0    
        c01_w_idx = (block * 2 + 1) * n + 0 
        c10_w_idx = (block * 2) * n + 1    
        c11_w_idx = (block * 2 + 1) * n + 1 
        
        U[0, base_c] = 1
        U[3, base_c] = 1
        V[b00_idx, base_c] = 1
        V[b11_idx, base_c] = 1
        W[c00_w_idx, base_c] = 1
        W[c01_w_idx, base_c] = 1
        W[c10_w_idx, base_c] = -1
        W[c11_w_idx, base_c] = 1
        
        U[2, base_c + 1] = 1
        U[3, base_c + 1] = 1
        V[b11_idx, base_c + 1] = 1
        W[c00_w_idx, base_c + 1] = -1
        W[c10_w_idx, base_c + 1] = 1
        
        U[0, base_c + 2] = 1
        V[b01_idx, base_c + 2] = 1
        V[b11_idx, base_c + 2] = -1
        W[c01_w_idx, base_c + 2] = 1
        W[c10_w_idx, base_c + 2] = 1
        
        U[3, base_c + 3] = 1
        V[b00_idx, base_c + 3] = 1
        V[b10_idx, base_c + 3] = -1
        W[c00_w_idx, base_c + 3] = 1
        W[c11_w_idx, base_c + 3] = -1
        
        U[0, base_c + 4] = 1
        U[1, base_c + 4] = 1
        V[b10_idx, base_c + 4] = 1
        W[c01_w_idx, base_c + 4] = 1
        W[c11_w_idx, base_c + 4] = 1
        
        U[2, base_c + 5] = 1
        U[0, base_c + 5] = -1
        V[b00_idx, base_c + 5] = 1
        V[b01_idx, base_c + 5] = 1
        W[c00_w_idx, base_c + 5] = 1
        W[c11_w_idx, base_c + 5] = -1
        
        U[1, base_c + 6] = 1
        U[3, base_c + 6] = -1
        V[b10_idx, base_c + 6] = 1
        V[b11_idx, base_c + 6] = 1
        W[c01_w_idx, base_c + 6] = -1
        W[c10_w_idx, base_c + 6] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
