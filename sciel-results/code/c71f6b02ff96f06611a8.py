import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
    def init_vec(dim):
        return [0.0] * dim
        
    def add_strassen(i0, j0, k0):
        # M1 = (a+d)(e+i)
        u1 = init_vec(n*m)
        v1 = init_vec(m*p)
        w1 = init_vec(p*n)
        u1[i0*m+j0] = 1.0
        u1[(i0+1)*m+(j0+1)] = 1.0
        v1[j0*p+k0] = 1.0
        v1[(j0+1)*p+(k0+1)] = 1.0
        w1[k0*n+i0] = 1.0
        w1[(k0+1)*n+(i0+1)] = 1.0
        add_term(u1, v1, w1)
        
        # M2 = (c+d)e
        u2 = init_vec(n*m)
        v2 = init_vec(m*p)
        w2 = init_vec(p*n)
        u2[(i0+1)*m+j0] = 1.0
        u2[(i0+1)*m+(j0+1)] = 1.0
        v2[j0*p+k0] = 1.0
        w2[k0*n+(i0+1)] = 1.0
        w2[(k0+1)*n+(i0+1)] = -1.0
        add_term(u2, v2, w2)
        
        # M3 = a(f-i)
        u3 = init_vec(n*m)
        v3 = init_vec(m*p)
        w3 = init_vec(p*n)
        u3[i0*m+j0] = 1.0
        v3[(j0+1)*p+k0] = 1.0
        v3[(j0+1)*p+(k0+1)] = -1.0
        w3[(k0+1)*n+i0] = 1.0
        w3[(k0+1)*n+(i0+1)] = 1.0
        add_term(u3, v3, w3)
        
        # M4 = d(g-h)
        u4 = init_vec(n*m)
        v4 = init_vec(m*p)
        w4 = init_vec(p*n)
        u4[(i0+1)*m+(j0+1)] = 1.0
        v4[j0*p+(k0+1)] = 1.0
        v4[(j0+1)*p+(k0+1)] = -1.0
        w4[k0*n+i0] = 1.0
        w4[k0*n+(i0+1)] = 1.0
        add_term(u4, v4, w4)
        
        # M5 = (a+b)h
        u5 = init_vec(n*m)
        v5 = init_vec(m*p)
        w5 = init_vec(p*n)
        u5[i0*m+j0] = 1.0
        u5[i0*m+(j0+1)] = 1.0
        v5[(j0+1)*p+k0] = 1.0
        w5[k0*n+i0] = -1.0
        w5[(k0+1)*n+i0] = 1.0
        add_term(u5, v5, w5)
        
        # M6 = (c-b)(e+f)
        u6 = init_vec(n*m)
        v6 = init_vec(m*p)
        w6 = init_vec(p*n)
        u6[(i0+1)*m+j0] = 1.0
        u6[i0*m+(j0+1)] = -1.0
        v6[j0*p+k0] = 1.0
        v6[j0*p+(k0+1)] = 1.0
        w6[(k0+1)*n+(i0+1)] = 1.0
        add_term(u6, v6, w6)
        
        # M7 = (a-d)(i+j)
        u7 = init_vec(n*m)
        v7 = init_vec(m*p)
        w7 = init_vec(p*n)
        u7[i0*m+j0] = 1.0
        u7[(i0+1)*m+(j0+1)] = -1.0
        v7[(j0+1)*p+k0] = 1.0
        v7[(j0+1)*p+(k0+1)] = 1.0
        w7[k0*n+i0] = 1.0
        add_term(u7, v7, w7)

    def add_221(i0, j0, k0):
        for idx in range(4):
            u = init_vec(n*m)
            v = init_vec(m*p)
            w = init_vec(p*n)
            if idx == 0:
                u[i0*m+j0] = 1.0
                v[j0*p+k0] = 1.0
                w[k0*n+i0] = 1.0
            elif idx == 1:
                u[i0*m+(j0+1)] = 1.0
                v[(j0+1)*p+k0] = 1.0
                w[k0*n+i0] = 1.0
            elif idx == 2:
                u[(i0+1)*m+j0] = 1.0
                v[j0*p+k0] = 1.0
                w[k0*n+(i0+1)] = 1.0
            elif idx == 3:
                u[(i0+1)*m+(j0+1)] = 1.0
                v[(j0+1)*p+k0] = 1.0
                w[k0*n+(i0+1)] = 1.0
            add_term(u, v, w)

    def add_122(i0, j0, k0):
        for idx in range(4):
            u = init_vec(n*m)
            v = init_vec(m*p)
            w = init_vec(p*n)
            if idx == 0:
                u[i0*m+j0] = 1.0
                v[j0*p+k0] = 1.0
                w[k0*n+i0] = 1.0
            elif idx == 1:
                u[i0*m+(j0+1)] = 1.0
                v[(j0+1)*p+k0] = 1.0
                w[k0*n+i0] = 1.0
            elif idx == 2:
                u[i0*m+j0] = 1.0
                v[j0*p+(k0+1)] = 1.0
                w[(k0+1)*n+i0] = 1.0
            elif idx == 3:
                u[i0*m+(j0+1)] = 1.0
                v[(j0+1)*p+(k0+1)] = 1.0
                w[(k0+1)*n+i0] = 1.0
            add_term(u, v, w)

    add_strassen(0, 0, 0)
    add_strassen(0, 2, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 2)
    add_221(0, 0, 4)
    add_221(0, 2, 4)
    add_122(2, 0, 0)
    add_122(2, 2, 0)
    add_122(2, 0, 2)
    add_122(2, 2, 2)
    
    for j in [0, 2]:
        u = init_vec(n*m)
        v = init_vec(m*p)
        w = init_vec(p*n)
        u[2*m+j] = 1.0
        v[j*p+4] = 1.0
        w[4*n+2] = 1.0
        add_term(u, v, w)
        
        u = init_vec(n*m)
        v = init_vec(m*p)
        w = init_vec(p*n)
        u[2*m+(j+1)] = 1.0
        v[(j+1)*p+4] = 1.0
        w[4*n+2] = 1.0
        add_term(u, v, w)

    rank = len(U_list)
    U_mat = np.zeros((n*m, rank))
    V_mat = np.zeros((m*p, rank))
    W_mat = np.zeros((p*n, rank))
    
    for r in range(rank):
        for i in range(n*m):
            U_mat[i, r] = U_list[r][i]
        for i in range(m*p):
            V_mat[i, r] = V_list[r][i]
        for i in range(p*n):
            W_mat[i, r] = W_list[r][i]
            
    return {"U": U_mat, "V": V_mat, "W": W_mat}
