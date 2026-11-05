import numpy as np

def construct():
    n, m, p = 6, 6, 6
    dim = 6
    
    # Fallback: Schoolbook 6x6 (Rank 216)
    def _schoolbook_6x6():
        rank = 216
        U = [[0] * rank for _ in range(36)]
        V = [[0] * rank for _ in range(36)]
        W = [[0] * rank for _ in range(36)]
        col = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U[i * 6 + j][col] = 1
                    V[j * 6 + k][col] = 1
                    W[k * 6 + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Hybrid: 2x2 Strassen (Outer) x 3x3 BCLR/Schoolbook (Inner)
    def _hybrid():
        # Inner Rank 21 (BCLR)
        # ... define bclr tensors ...
        # If BCLR is valid, use it. Else use Rank 27.
        
        # Let's use Rank 27 (Schoolbook) for inner to be safe, 
        # but the prompt wants improvement.
        # I will use Rank 21.
        
        pass

    # ...
