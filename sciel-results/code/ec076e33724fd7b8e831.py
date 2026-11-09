import time
import sys

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
class TensorDecomposition:
    def __init__(self):
        self.cache = {}
        self.deadline = time.time() + 110

    def get_rank(self, n, m, p):
        if (n, m, p) in self.cache:
            return self.cache[(n, m, p)]
        
        best_rank = n * m * p
        
        # Try splitting n
        for i in range(1, n):
            if time.time() > self.deadline:
                break
            r1 = self.get_rank(i, m, p)
            r2 = self.get_rank(n-i, m, p)
            best_rank = min(best_rank, r1 + r2)
            
        # Try splitting m
        for j in range(1, m):
            if time.time() > self.deadline:
                break
            r1 = self.get_rank(n, j, p)
            r2 = self.get_rank(n, m-j, p)
            best_rank = min(best_rank, r1 + r2)
            
        # Try splitting p
        for k in range(1, p):
            if time.time() > self.deadline:
                break
            r1 = self.get_rank(n, m, k)
            r2 = self.get_rank(n, m, p-k)
            best_rank = min(best_rank, r1 + r2)
            
        self.cache[(n, m, p)] = best_rank
        return best_rank

    def _strassen_2x2x2(self):
        """
        Returns U, V, W matrices for 2x2x2 matrix multiplication using Strassen's algorithm.
        U shape: (4, 7) - rows indexed by (i, j) with i in {0,1}, j in {0,1}, index = i*2+j
        V shape: (4, 7) - rows indexed by (j, k) with j in {0,1}, k in {0,1}, index = j*2+k
        W shape: (4, 7) - rows indexed by (k, i) with k in {0,1}, i in {0,1}, index = k*2+i
        
        Strassen's 7 products:
        M1 = (A00 + A11)(B00 + B11)
        M2 = (A10 + A11)B00
        M3 = A00(B10 - B00)
        M4 = A11(B01 - B11)
        M5 = (A00 + A01)B11
        M6 = (A10 - A00)(B00 + B01)
        M7 = (A01 - A11)(B10 + B11)
        
        Output:
        C00 = M1 - M3 + M5 - M7
        C01 = M4 + M5
        C10 = M2 + M3
        C11 = M1 - M2 + M4 + M6
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        # A indices: 00=0, 01=1, 10=2, 11=3
        # B indices: 00=0, 01=1, 10=2, 11=3
        # C indices: 00=0, 01=1, 10=2, 11=3
        U[0][0] = 1; U[3][0] = 1  # A00, A11
        V[0][0] = 1; V[3][0] = 1  # B00, B11
        W[0][0] = 1; W[3][0] = 1  # C00, C11
        
        # M2: (A10+A11)B00 -> C10 += 1, C11 += -1
        U[2][1] = 1; U[3][1] = 1  # A10, A11
        V[0][1] = 1  # B00
        W[2][1] = 1; W[3][1] = -1  # C10, C11
        
        # M3: A00(B10-B00) -> C00 += -1, C10 += 1
        U[0][2] = 1  # A00
        V[2][2] = 1; V[0][2] = -1  # B10, B00
        W[0][2] = -1; W[2][2] = 1  # C00, C10
        
        # M4: A11(B01-B11) -> C01 += 1, C11 += 1
        U[3][3] = 1  # A11
        V[1][3] = 1; V[3][3] = -1  # B01, B11
        W[1][3] = 1; W[3][3] = 1  # C01, C11
        
        # M5: (A00+A01)B11 -> C00 += -1, C01 += 1
        U[0][4] = 1; U[1][4] = 1  # A00, A01
        V[3][4] = 1  # B11
        W[0][4] = -1; W[1][4] = 1  # C00, C01
        
        # M6: (A10-A00)(B00+B01) -> C11 += 1
        U[2][5] = 1; U[0][5] = -1  # A10, A00
        V[0][5] = 1; V[1][5] = 1  # B00, B01
        W[3][5] = 1  # C11
        
        # M7: (A01-A11)(B10+B11) -> C00 += -1
        U[1][6] = 1; U[3][6] = -1  # A01, A11
        V[2][6] = 1; V[3][6] = 1  # B10, B11
        W[0][6] = -1  # C00
        
        return U, V, W, 7

    def construct(self, n, m, p):
        if time.time() > self.deadline:
             rank = n * m * p
             U = [[0] * rank for _ in range(n * m)]
             V = [[0] * rank for _ in range(m * p)]
             W = [[0] * rank for _ in range(p * n)]
             col = 0
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         U[i*m+j][col] = 1
                         V[j*p+k][col] = 1
                         W[k*n+i][col] = 1
                         col += 1
             return U, V, W, rank
        
        # Base case: 2x2x2 uses Strassen
        if n == 2 and m == 2 and p == 2:
            return self._strassen_2x2x2()
        
        # Base case: 1x1x1
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W, 1
        
        best_rank = n * m * p
        best_split = None
        
        for i in range(1, n):
            r1 = self.get_rank(i, m, p)
            r2 = self.get_rank(n-i, m, p)
            if r1 + r2 < best_rank:
                best_rank = r1 + r2
                best_split = ('n', i)
                
        for j in range(1, m):
            r1 = self.get_rank(n, j, p)
            r2 = self.get_rank(n, m-j, p)
            if r1 + r2 < best_rank:
                best_rank = r1 + r2
                best_split = ('m', j)
                
        for k in range(1, p):
            r1 = self.get_rank(n, m, k)
            r2 = self.get_rank(n, m, p-k)
            if r1 + r2 < best_rank:
                best_rank = r1 + r2
                best_split = ('p', k)

        if best_split is None:
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
            return U, V, W, rank
        else:
            dim, idx = best_split
            if dim == 'n':
                n1, n2 = idx, n - idx
                U1, V1, W1, r1 = self.construct(n1, m, p)
                U2, V2, W2, r2 = self.construct(n2, m, p)
                
                U = [[0] * (r1 + r2) for _ in range(n * m)]
                V = [[0] * (r1 + r2) for _ in range(m * p)]
                W = [[0] * (r1 + r2) for _ in range(p * n)]
                
                for i in range(n1):
                    for j in range(m):
                        for c in range(r1):
                            U[i*m+j][c] = U1[i*m+j][c]
                for i in range(n2):
                    for j in range(m):
                        for c in range(r2):
                            U[(n1+i)*m+j][r1+c] = U2[i*m+j][c]
                            
                for j in range(m):
                    for k in range(p):
                        for c in range(r1):
                            V[j*p+k][c] = V1[j*p+k][c]
                        for c in range(r2):
                            V[j*p+k][r1+c] = V2[j*p+k][c]
                            
                for k in range(p):
                    for i in range(n1):
                        for c in range(r1):
                            W[k*n+i][c] = W1[k*n1+i][c]
                    for i in range(n2):
                        for c in range(r2):
                            W[k*n + (n1+i)][r1+c] = W2[k*n2+i][c]
                            
                return U, V, W, r1 + r2
            elif dim == 'm':
                m1, m2 = idx, m - idx
                U1, V1, W1, r1 = self.construct(n, m1, p)
                U2, V2, W2, r2 = self.construct(n, m2, p)
                
                U = [[0] * (r1 + r2) for _ in range(n * m)]
                V = [[0] * (r1 + r2) for _ in range(m * p)]
                W = [[0] * (r1 + r2) for _ in range(p * n)]
                
                for i in range(n):
                    for j in range(m1):
                        for c in range(r1):
                            U[i*m+j][c] = U1[i*m1+j][c]
                    for j in range(m2):
                        for c in range(r2):
                            U[i*m + (m1+j)][r1+c] = U2[i*m2+j][c]
                            
                for j in range(m1):
                    for k in range(p):
                        for c in range(r1):
                            V[j*p+k][c] = V1[j*p+k][c]
                for j in range(m2):
                    for k in range(p):
                        for c in range(r2):
                            V[(m1+j)*p+k][r1+c] = V2[j*p+k][c]
                            
                for k in range(p):
                    for i in range(n):
                        for c in range(r1):
                            W[k*n+i][c] = W1[k*n+i][c]
                        for c in range(r2):
                            W[k*n+i][r1+c] = W2[k*n+i][c]
                            
                return U, V, W, r1 + r2
            else:
                p1, p2 = idx, p - idx
                U1, V1, W1, r1 = self.construct(n, m, p1)
                U2, V2, W2, r2 = self.construct(n, m, p2)
                
                U = [[0] * (r1 + r2) for _ in range(n * m)]
                V = [[0] * (r1 + r2) for _ in range(m * p)]
                W = [[0] * (r1 + r2) for _ in range(p * n)]
                
                for i in range(n):
                    for j in range(m):
                        for c in range(r1):
                            U[i*m+j][c] = U1[i*m+j][c]
                        for c in range(r2):
                            U[i*m+j][r1+c] = U2[i*m+j][c]
                            
                for j in range(m):
                    for k in range(p1):
                        for c in range(r1):
                            V[j*p+k][c] = V1[j*p1+k][c]
                    for k in range(p2):
                        for c in range(r2):
                            V[j*p + (p1+k)][r1+c] = V2[j*p2+k][c]
                            
                for k in range(p1):
                    for i in range(n):
                        for c in range(r1):
                            W[k*n+i][c] = W1[k*n+i][c]
                for k in range(p2):
                    for i in range(n):
                        for c in range(r2):
                            W[(p1+k)*n+i][r1+c] = W2[k*n+i][c]
                            
                return U, V, W, r1 + r2

def construct():
    n, m, p = DIMENSIONS
    solver = TensorDecomposition()
    U, V, W, rank = solver.construct(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
