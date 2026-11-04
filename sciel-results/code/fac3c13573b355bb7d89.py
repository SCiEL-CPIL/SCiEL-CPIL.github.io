import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_terms.append(u_vec)
        V_terms.append(v_vec)
        W_terms.append(w_vec)
    
    def make_vec(length):
        return [0] * length
    
    def strassen_2x2x2(i0, j0, k0):
        # M1 = (A00+A11)*(B00+B11) -> C[0,0]+=1, C[1,1]+=1
        u = make_vec(n*m); v = make_vec(m*p); w = make_vec(p*n)
        u[i0*m+j0] = 1; u[(i0+1)*m+(j0+1)] = 1
        v[j0*p+k0] = 1; v[(j0+1)*p+(k0+1)] = 1
        w[k0*n+i0] = 1; w[(k0+1)*n+(i0+1)] = 1
        add_term(u, v, w)
        
        # M2 = (A10+A11)*B00 -> C[1,0]+=1, C[1,1]+=1
        u = make_vec(n*m); v = make_vec(m*p); w = make_vec(p*n)
        u[(i0+1)*m+j0] = 1; u[(i0+1)*m+(j0+1)] = 1
        v[j0*p+k0] = 1
        w[k0*n+(i0+1)] = 1; w[(k0+1)*n+(i0+1)] = 1
        add_term(u, v, w)
        
        # M3 = A00*(B01-B11) -> C[0,1]+=1, C[1,1]+=1
        u = make_vec(n*m); v = make_vec(m*p); w = make_vec(p*n)
        u[i0*m+j0] = 1
        v[j0*p+(k0+1)] = 1; v[(j0+1)*p+(k0+1)] = -1
        w[(k0+1)*n+i0] = 1; w[(k0+1)*n+(i0+1)] = 1
        add_term(u, v, w)
        
        # M4 = A11*(B10-B00) -> C[0,0]+=1, C[1,0]+=1
        u = make_vec(n*m); v = make_vec(m*p); w = make_vec(p*n)
        u[(i0+1)*m+(j0+1)] = 1
        v[(j0+1)*p+k0] = 1; v[j0*p+k0] = -1
        w[k0*n+i0] = 1; w[k0*n+(i0+1)] = 1
        add_term(u, v, w)
        
        # M5 = (A00+A01)*B11 -> C[0,1]+=1, C[1,1]+=1
        u = make_vec(n*m); v = make_vec(m*p); w = make_vec(p*n)
        u[i0*m+j0] = 1; u[i0*m+(j0+1)] = 1
        v[(j0+1)*p+(k0+1)] = 1
        w[(k0+1)*n+i0] = 1; w[(k0+1)*n+(i0+1)] = 1
        add_term(u, v, w)
        
        # M6 = (A10-A00)*(B00+B01) -> C[1,1]+=1
        u = make_vec(n*m); v = make_vec(m*p); w = make_vec(p*n)
        u[(i0+1)*m+j0] = 1; u[i0*m+j0] = -1
        v[j0*p+k0] = 1; v[j0*p+(k0+1)] = 1
        w[(k0+1)*n+(i0+1)] = 1
        add_term(u, v, w)
        
        # M7 = (A01-A11)*(B10+B11) -> C[0,0]-=1
        u = make_vec(n*m); v = make_vec(m*p); w = make_vec(p*n)
        u[i0*m+(j0+1)] = 1; u[(i0+1)*m+(j0+1)] = -1
        v[(j0+1)*p+k0] = 1; v[(j0+1)*p+(k0+1)] = 1
        w[k0*n+i0] = -1
        add_term(u, v, w)
    
    # 2x5x6 block: rows 0,1 of A
    # j=0,1 with k=0,1: Strassen
    strassen_2x2x2(0, 0, 0)
    # j=0,1 with k=2,3: Strassen
    strassen_2x2x2(0, 0, 2)
    # j=0,1 with k=4,5: Strassen
    strassen_2x2x2(0, 0, 4)
    # j=2,3 with k=0,1: Strassen
    strassen_2x2x2(0, 2, 0)
    # j=2,3 with k=2,3: Strassen
    strassen_2x2x2(0, 2, 2)
    # j=2,3 with k=4,5: Strassen
    strassen_2x2x2(0, 2, 4)
    
    # j=4 (single column) with k=0..5: rank-2 decomposition
    for i in [0, 1]:
        u = make_vec(n*m); v = make_vec(m*p); w = make_vec(p*n)
        u[i*m+4] = 1
        for k in range(6):
            v[4*p+k] = 1
            w[k*n+i] = 1
        add_term(u, v, w)
    
    # 1x5x6 block: row 2 of A, direct
    for j in range(5):
        for k in range(6):
            u = make_vec(n*m); v = make_vec(m*p); w = make_vec(p*n)
            u[2*m+j] = 1
            v[j*p+k] = 1
            w[k*n+2] = 1
            add_term(u, v, w)
    
    return {"U": U_terms, "V": V_terms, "W": W_terms}
