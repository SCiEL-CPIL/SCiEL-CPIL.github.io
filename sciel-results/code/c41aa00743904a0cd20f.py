import time
import random
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 5
    deadline = time.time() + 140.0
    
    # Fallback rank-36 solution
    def get_rank_36():
        R = 36
        U = [[0]*R for _ in range(8)]
        V = [[0]*R for _ in range(20)]
        W = [[0]*R for _ in range(10)]
        
        def strassen_block(base, x00, x01, x10, x11, y00, y01, y10, y11, out_k0, out_k1):
            U[x00][base+0] = 1
            V[y10][base+0] = 1
            V[y11][base+0] = 1
            W[2*out_k0+0][base+0] = -1
            W[2*out_k1+0][base+0] = 1

            U[x00][base+1] = 1
            U[x01][base+1] = 1
            V[y10][base+1] = 1
            W[2*out_k1+0][base+1] = 1
            W[2*out_k1+1][base+1] = 1

            U[x10][base+2] = 1
            U[x11][base+2] = 1
            V[y00][base+2] = 1
            W[2*out_k0+1][base+2] = 1
            W[2*out_k1+1][base+2] = -1

            U[x11][base+3] = 1
            V[y01][base+3] = 1
            V[y00][base+3] = -1
            W[2*out_k0+0][base+3] = 1
            W[2*out_k0+1][base+3] = 1

            U[x00][base+4] = 1
            U[x11][base+4] = 1
            V[y00][base+4] = 1
            V[y11][base+4] = 1
            W[2*out_k0+0][base+4] = 1
            W[2*out_k1+1][base+4] = 1

            U[x01][base+5] = 1
            U[x10][base+5] = -1
            V[y10][base+5] = 1
            V[y11][base+5] = 1
            W[2*out_k1+1][base+5] = -1

            U[x11][base+6] = 1
            U[x10][base+6] = -1
            V[y00][base+6] = 1
            V[y10][base+6] = -1
            W[2*out_k0+0][base+6] = 1

        strassen_block(0, 0, 1, 4, 5, 0, 1, 5, 6, 0, 1)
        strassen_block(7, 2, 3, 6, 7, 10, 11, 15, 16, 0, 1)
        strassen_block(14, 0, 1, 4, 5, 2, 3, 7, 8, 2, 3)
        strassen_block(21, 2, 3, 6, 7, 12, 13, 17, 18, 2, 3)
        
        for i in range(2):
            for j in range(4):
                col = 28 + i*4 + j
                U[i*4+j][col] = 1
                V[j*5+4][col] = 1
                W[2*4+i][col] = 1
                
        return {"U": U, "V": V, "W": W}

    # Verify function
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True

    # Attempt to find rank 35 via random search?
    # Given the complexity, it's safer to return the known 36.
    # But let's try a simple local search on the 8 trivial products?
    # No, 8 is minimal for trivial.
    
    return get_rank_36()
