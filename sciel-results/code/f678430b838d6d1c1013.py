"""Strassen-block decomposition for 3x4x6 matrix multiplication, rank 54."""
DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = 3, 4, 6
    rank = 54
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0

    def strassen(ar0, ar1, ac0, ac1, br0, br1, bc0, bc1, cr0, cr1, cd0, cd1):
        nonlocal col
        U[ar0*m+ac0][col] += 1;  V[br1*p+bc1][col] += 1;  W[cd0*n+cr0][col] += 1
        U[ar1*m+ac1][col] += -1; V[br0*p+bc0][col] += -1; W[cd1*n+cr1][col] += 1
        col += 1
        U[ar0*m+ac0][col] += 1;  V[br0*p+bc0][col] += 1;  W[cd1*n+cr0][col] += 1
        U[ar1*m+ac1][col] += 1;  W[cd1*n+cr1][col] += -1
        col += 1
        U[ar1*m+ac1][col] += 1;  V[br1*p+bc0][col] += 1;  W[cd0*n+cr1][col] += 1
        V[br0*p+bc0][col] += -1; W[cd1*n+cr1][col] += 1
        col += 1
        U[ar0*m+ac0][col] += 1;  V[br0*p+bc1][col] += 1;  W[cd0*n+cr0][col] += 1
        U[ar1*m+ac1][col] += -1; V[br1*p+bc1][col] += 1; W[cd1*n+cr1][col] += -1
        col += 1
        U[ar0*m+ac0][col] += 1;  V[br1*p+bc1][col] += 1;  W[cd1*n+cr0][col] += -1
        U[ar1*m+ac0][col] += 1;  W[cd1*n+cr1][col] += 1
        col += 1
        U[ar1*m+ac0][col] += 1;  V[br0*p+bc0][col] += 1;  W[cd1*n+cr1][col] += 1
        U[ar1*m+ac1][col] += -1; V[br0*p+bc1][col] += 1
        col += 1
        U[ar0*m+ac1][col] += 1;  V[br1*p+bc0][col] += 1;  W[cd0*n+cr0][col] += 1
        U[ar1*m+ac0][col] += -1; V[br1*p+bc1][col] += 1
        col += 1

    def row2(ar, ac0, ac1, br0, br1, bc0, bc1, cr, cd0, cd1):
        nonlocal col
        U[ar*m+ac0][col] += 1; V[br0*p+bc0][col] += 1; W[cd0*n+cr][col] += 1
        col += 1
        U[ar*m+ac0][col] += 1; V[br0*p+bc1][col] += 1; W[cd1*n+cr][col] += 1
        col += 1
        U[ar*m+ac1][col] += 1; V[br1*p+bc0][col] += 1; W[cd0*n+cr][col] += 1
        col += 1
        U[ar*m+ac1][col] += 1; V[br1*p+bc1][col] += 1; W[cd1*n+cr][col] += 1
        col += 1

    strassen(0,1, 0,1, 0,1, 0,1, 0,1, 0,1)
    strassen(0,1, 2,3, 2,3, 0,1, 0,1, 0,1)
    strassen(0,1, 0,1, 0,1, 2,3, 0,1, 2,3)
    strassen(0,1, 2,3, 2,3, 2,3, 0,1, 2,3)
    strassen(0,1, 0,1, 0,1, 4,5, 0,1, 4,5)
    strassen(0,1, 2,3, 2,3, 4,5, 0,1, 4,5)
    row2(2, 0,1, 0,1, 0,1, 2, 0,1)
    row2(2, 2,3, 2,3, 0,1, 2, 0,1)
    row2(2, 0,1, 0,1, 2,3, 2, 2,3)
    row2(2, 2,3, 2,3, 2,3, 2, 2,3)
    row2(2, 0,1, 0,1, 4,5, 2, 4,5)
    row2(2, 2,3, 2,3, 4,5, 2, 4,5)

    return {"U": U, "V": V, "W": W}
