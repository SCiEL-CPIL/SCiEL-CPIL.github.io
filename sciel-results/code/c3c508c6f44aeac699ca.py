import time

def construct():
    n = 10
    S1 = [[0]*n for _ in range(n)]
    S2 = [[0]*n for _ in range(n)]
    
    for r in range(n):
        r1 = r // 2
        r2 = r % 2
        for c in range(n):
            c1 = c // 2
            c2 = c % 2
            
            # S1
            u1 = (r1 + c1) % 5
            u2 = (r2 + c2) % 2
            S1[r][c] = 2 * u1 + u2
            
            # S2
            v1 = (r1 + 2 * c1) % 5
            v2 = c2 % 2
            S2[r][c] = 2 * v1 + v2
            
    return [S1, S2]
