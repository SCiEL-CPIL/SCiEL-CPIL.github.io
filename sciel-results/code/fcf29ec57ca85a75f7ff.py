import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    u_list = []
    v_list = []
    w_list = []
    
    def add_term(u_vec, v_vec, w_vec):
        u_list.append(u_vec)
        v_list.append(v_vec)
        w_list.append(w_vec)
        
    def strassen_2x2x2(i0, j0, k0):
        def make_u(vec):
            u = np.zeros(n*m)
            for r in range(2):
                for c in range(2):
                    idx = (i0 + r) * m + (j0 + c)
                    u[idx] = vec[r*2+c]
            return u
            
        def make_v(vec):
            v = np.zeros(m*p)
            for r in range(2):
                for c in range(2):
                    idx = (j0 + r) * p + (k0 + c)
                    v[idx] = vec[r*2+c]
            return v
            
        def make_w(vec):
            w = np.zeros(p*n)
            for r in range(2):
                for c in range(2):
                    idx = (k0 + c) * n + (i0 + r)
                    w[idx] = vec[r*2+c]
            return w
            
        # 1
        add_term(make_u([1,0,0,1]), make_v([1,0,0,1]), make_w([1,0,0,1]))
        # 2
        add_term(make_u([0,0,1,1]), make_v([1,0,0,0]), make_w([0,0,1,0]))
        # 3
        add_term(make_u([1,0,0,0]), make_v([0,1,0,-1]), make_w([0,1,0,0]))
        # 4
        add_term(make_u([0,0,0,1]), make_v([-1,0,1,0]), make_w([-1,0,1,0]))
        # 5
        add_term(make_u([1,1,0,0]), make_v([0,0,0,1]), make_w([-1,1,0,0]))
        # 6
        add_term(make_u([-1,0,1,0]), make_v([1,1,0,0]), make_w([0,0,0,-1]))
        # 7
        add_term(make_u([0,1,0,-1]), make_v([0,0,1,1]), make_w([1,0,0,0]))
        
    def schoolbook_2x2x1(i0, j0, k0):
        def make_u(vec):
            u = np.zeros(n*m)
            for r in range(2):
                for c in range(2):
                    idx = (i0 + r) * m + (j0 + c)
                    u[idx] = vec[r*2+c]
            return u
            
        def make_v(vec):
            v = np.zeros(m*p)
            for r in range(2):
                for c in range(1): # Only 1 col
                    idx = (j0 + r) * p + (k0 + c)
                    v[idx] = vec[r*2+c]
            return v
            
        def make_w(vec):
            w = np.zeros(p*n)
            for r in range(2):
                for c in range(1): # Only 1 col
                    idx = (k0 + c) * n + (i0 + r)
                    w[idx] = vec[r*2+c]
            return w
            
        # a*e -> p
        add_term(make_u([1,0,0,0]), make_v([1,0]), make_w([1,0]))
        # b*f -> p
        add_term(make_u([0,1,0,0]), make_v([0,1]), make_w([1,0]))
        # c*e -> r
        add_term(make_u([0,0,1,0]), make_v([1,0]), make_w([0,1]))
        # d*f -> r
        add_term(make_u([0,0,0,1]), make_v([0,1]), make_w([0,1]))
        
    # Top block: i0=0
    # B cols 0,1
    strassen_2x2x2(0, 0, 0)
    # B cols 2,3
    strassen_2x2x2(0, 2, 2)
    # B col 4
    schoolbook_2x2x1(0, 4, 4)
    
    # Bottom block: i0=2
    # B cols 0,1
    strassen_2x2x2(2, 0, 0)
    # B cols 2,3
    strassen_2x2x2(2, 2, 2)
    # B col 4
    schoolbook_2x2x1(2, 4, 4)
    
    rank = len(u_list)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r in range(rank):
        U[:, r] = u_list[r]
        V[:, r] = v_list[r]
        W[:, r] = w_list[r]
        
    return {"U": U, "V": V, "W": W}
