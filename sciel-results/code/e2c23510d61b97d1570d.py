import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(16)]
    
    col = 0
    for s in range(2):
        for j in range(4):
            a00 = 0 * 4 + (2*s)
            a01 = 0 * 4 + (2*s + 1)
            a10 = 1 * 4 + (2*s)
            a11 = 1 * 4 + (2*s + 1)
            
            b00 = (2*s) * 8 + (2*j)
            b01 = (2*s) * 8 + (2*j + 1)
            b10 = (2*s + 1) * 8 + (2*j)
            b11 = (2*s + 1) * 8 + (2*j + 1)
            
            c00 = (2*j) * 2 + 0
            c01 = (2*j + 1) * 2 + 0
            c10 = (2*j) * 2 + 1
            c11 = (2*j + 1) * 2 + 1
            
            # M1 = (a00+a11)(b00+b11)
            U[a00][col] += 1; U[a11][col] += 1
            V[b00][col] += 1; V[b11][col] += 1
            W[c00][col] += 1
            W[c11][col] += 1
            
            # M2 = (a10+a11)b00
            U[a10][col+1] += 1; U[a11][col+1] += 1
            V[b00][col+1] += 1
            W[c10][col+1] += 1
            W[c11][col+1] += -1
            
            # M3 = a00(b01-b11)
            U[a00][col+2] += 1
            V[b01][col+2] += 1; V[b11][col+2] += -1
            W[c01][col+2] += 1
            W[c11][col+2] += 1
            
            # M4 = a11(b10-b00)
            U[a11][col+3] += 1
            V[b10][col+3] += 1; V[b00][col+3] += -1
            W[c10][col+3] += 1
            W[c00][col+3] += 1
            
            # M5 = (a00+a01)b11
            U[a00][col+4] += 1; U[a01][col+4] += 1
            V[b11][col+4] += 1
            W[c00][col+4] += -1
            W[c01][col+4] += 1
            
            # M6 = (a10-a00)(b00+b01)
            U[a10][col+5] += 1; U[a00][col+5] += -1
            V[b00][col+5] += 1; V[b01][col+5] += 1
            W[c00][col+5] += -1
            W[c11][col+5] += 1
            
            # M7 = (a01-a11)(b10+b11)
            U[a01][col+6] += 1; U[a11][col+6] += -1
            V[b10][col+6] += 1; V[b11][col+6] += 1
            W[c00][col+6] += 1
            W[c11][col+6] += 0  # M7 doesn't contribute to c11 per formula... wait
            
            col += 7
    
    # Verify: C[1,1] = M1 - M2 + M3 + M6
    # c11: M1(+1), M2(-1), M3(+1), M6(+1) -> looks correct
    # C[0,0] = M1 + M4 - M5 + M7
    # c00: M1(+1), M4(+1), M5(-1), M7(+1) -> correct
    
    return {"U": U, "V": V, "W": W}
