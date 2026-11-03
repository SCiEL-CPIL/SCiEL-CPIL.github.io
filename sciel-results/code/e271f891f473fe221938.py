import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    rank = 0
    for ni in n_splits:
        for mj in m_splits:
            for pk in p_splits:
                if ni == 2 and mj == 2 and pk == 2:
                    rank += 7
                else:
                    rank += ni * mj * pk
                    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    n_offset = 0
    m_offset = 0
    p_offset = 0
    
    for ni in n_splits:
        for mj in m_splits:
            for pk in p_splits:
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen
                    # M1
                    U[(n_offset+0)*m + (m_offset+0)][col] = 1
                    U[(n_offset+1)*m + (m_offset+1)][col] = 1
                    V[(m_offset+0)*p + (p_offset+0)][col] = 1
                    V[(m_offset+1)*p + (p_offset+1)][col] = 1
                    W[(p_offset+0)*n + (n_offset+0)][col] = 1
                    W[(p_offset+1)*n + (n_offset+1)][col] = -1
                    col += 1
                    
                    # M2
                    U[(n_offset+1)*m + (m_offset+0)][col] = 1
                    U[(n_offset+1)*m + (m_offset+1)][col] = 1
                    V[(m_offset+0)*p + (p_offset+0)][col] = 1
                    W[(p_offset+0)*n + (n_offset+1)][col] = 1
                    col += 1
                    
                    # M3
                    U[(n_offset+0)*m + (m_offset+0)][col] = 1
                    V[(m_offset+1)*p + (p_offset+0)][col] = 1
                    V[(m_offset+1)*p + (p_offset+1)][col] = -1
                    W[(p_offset+1)*n + (n_offset+0)][col] = 1
                    col += 1
                    
                    # M4
                    U[(n_offset+1)*m + (m_offset+1)][col] = 1
                    V[(m_offset+1)*p + (p_offset+1)][col] = 1
                    V[(m_offset+0)*p + (p_offset+0)][col] = -1
                    W[(p_offset+0)*n + (n_offset+0)][col] = 1
                    W[(p_offset+1)*n + (n_offset+1)][col] = 1
                    col += 1
                    
                    # M5
                    U[(n_offset+0)*m + (m_offset+0)][col] = 1
                    U[(n_offset+0)*m + (m_offset+1)][col] = 1
                    V[(m_offset+1)*p + (p_offset+1)][col] = 1
                    W[(p_offset+0)*n + (n_offset+0)][col] = -1
                    W[(p_offset+1)*n + (n_offset+0)][col] = 1
                    col += 1
                    
                    # M6
                    U[(n_offset+1)*m + (m_offset+0)][col] = 1
                    U[(n_offset+0)*m + (m_offset+0)][col] = -1
                    V[(m_offset+0)*p + (p_offset+0)][col] = 1
                    V[(m_offset+1)*p + (p_offset+0)][col] = 1
                    W[(p_offset+1)*n + (n_offset+0)][col] = 1
                    col += 1
                    
                    # M7
                    U[(n_offset+0)*m + (m_offset+1)][col] = 1
                    U[(n_offset+1)*m + (m_offset+1)][col] = -1
                    V[(m_offset+1)*p + (p_offset+0)][col] = 1
                    V[(m_offset+1)*p + (p_offset+1)][col] = 1
                    W[(p_offset+0)*n + (n_offset+0)][col] = -1
                    W[(p_offset+1)*n + (n_offset+1)][col] = 1
                    col += 1
                else:
                    for i in range(ni):
                        for j in range(mj):
                            for k in range(pk):
                                U[(n_offset+i)*m + (m_offset+j)][col] = 1
                                V[(m_offset+j)*p + (p_offset+k)][col] = 1
                                W[(p_offset+k)*n + (n_offset+i)][col] = 1
                                col += 1
                p_offset += pk
            m_offset += mj
        n_offset += ni
            
    return {"U": U, "V": V, "W": W}
