"""Matrix multiplication 4x5 x 5x5 rank-minimal tensor decomposition."""
import time

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2():
    """Strassen rank-7 decomposition for 2x2 * 2x2 -> 2x2.
    Returns U(4,7), V(4,7), W(4,7).
    U[i*2+j], V[j*2+k], W[k*2+i]
    C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # A: 0=(0,0),1=(0,1),2=(1,0),3=(1,1)
    # B: 0=(0,0),1=(0,1),2=(1,0),3=(1,1)
    # C: 0=(0,0),1=(0,1),2=(1,0),3=(1,1) [k*2+i]
    
    # M1 = (a+d)(e+h)
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1  # C00 += M1, C11 += M1
    
    # M2 = (c+d)e
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[2][1]=1; W[3][1]=-1  # C10 += M2, C11 -= M2
    
    # M3 = a(f-h)
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[1][2]=1; W[3][2]=1  # C01 += M3, C11 += M3
    
    # M4 = d(g-e)
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1  # C00 += M4, C10 += M4
    
    # M5 = (a+b)h
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][4]=-1; W[1][4]=1  # C00 -= M5, C01 += M5
    
    # M6 = (c-a)(e+f)
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1  # C11 += M6
    
    # M7 = (b-d)(g+h)
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1  # C00 += M7
    
    return U, V, W

def _schoolbook(a, b, c):
    """Schoolbook for a x b * b x c -> a x c."""
    rank = a*b*c
    U = [[0]*rank for _ in range(a*b)]
    V = [[0]*rank for _ in range(b*c)]
    W = [[0]*rank for _ in range(c*a)]
    col = 0
    for i in range(a):
        for j in range(b):
            for k in range(c):
                U[i*b+j][col] = 1
                V[j*c+k][col] = 1
                W[k*a+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    n_parts = [2, 2]
    m_parts = [2, 3]
    p_parts = [2, 3]
    
    n_off = [0]
    for x in n_parts[:-1]:
        n_off.append(n_off[-1] + x)
    m_off = [0]
    for x in m_parts[:-1]:
        m_off.append(m_off[-1] + x)
    p_off = [0]
    for x in p_parts[:-1]:
        p_off.append(p_off[-1] + x)
    
    block_info = []
    total_rank = 0
    for bi in range(len(n_parts)):
        for bj in range(len(m_parts)):
            a = n_parts[bi]
            b = m_parts[bj]
            c = p_parts[bj]
            if a == 2 and b == 2 and c == 2:
                alg = 'strassen'
                brank = 7
            else:
                alg = 'schoolbook'
                brank = a*b*c
            block_info.append((bi, bj, a, b, c, alg, brank))
            total_rank += brank
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_off = 0
    for (bi, bj, a, b, c, alg, brank) in block_info:
        r0 = n_off[bi]
        c0 = m_off[bj]
        d0 = p_off[bj]
        
        if alg == 'strassen':
            Ub, Vb, Wb = _strassen_2x2x2()
        else:
            Ub, Vb, Wb = _schoolbook(a, b, c)
        
        for li in range(a):
            for lj in range(b):
                lrow = li*b + lj
                grow = (r0+li)*m + (c0+lj)
                for col in range(brank):
                    if Ub[lrow][col] != 0:
                        U[grow][col_off+col] = Ub[lrow][col]
        
        for lj in range(b):
            for lk in range(c):
                lrow = lj*c + lk
                grow = (c0+lj)*p + (d0+lk)
                for col in range(brank):
                    if Vb[lrow][col] != 0:
                        V[grow][col_off+col] = Vb[lrow][col]
        
        for lk in range(c):
            for li in range(a):
                lrow = lk*a + li
                grow = (d0+lk)*n + (r0+li)
                for col in range(brank):
                    if Wb[lrow][col] != 0:
                        W[grow][col_off+col] = Wb[lrow][col]
        
        col_off += brank
    
    return {"U": U, "V": V, "W": W}
