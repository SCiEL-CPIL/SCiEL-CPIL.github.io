import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
        
    # ... (same strassen_4x4_head as before) ...
    
    # New tail: rank 24 for 4x4x2
    def strassen_4x4x2_tail_rank24():
        # B cols 4, 5
        # A blocks: A00, A01, A10, A11 (2x2)
        # B blocks: B0 (rows 0-1, cols 4-5), B1 (rows 2-3, cols 4-5)
        
        # 24 terms
        # 1. (A00+A11)(B0+B1) -> C00 + C11
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[0]=1; u[1]=1; u[4]=1; u[5]=1 # A00+A11? No, A00 is 0,1,4,5?
        # A is 4x4. A00 is rows 0-1, cols 0-1.
        # A00: 0,1,4,5. A11: 10,11,14,15.
        u[0]=1; u[1]=1; u[4]=1; u[5]=1
        u[10]=1; u[11]=1; u[14]=1; u[15]=1
        
        v[4*2+0]=1; v[4*2+1]=1 # B0 col 0? No, B is 4x7.
        # B0 is rows 0-1, cols 4-5.
        # v indices: row * p + col.
        # B0: rows 0,1; cols 4,5.
        # v[0*7+4]=1, v[0*7+5]=1, v[1*7+4]=1, v[1*7+5]=1
        v[4]=1; v[5]=1; v[11]=1; v[12]=1
        # B1: rows 2,3; cols 4,5.
        v[18]=1; v[19]=1; v[25]=1; v[26]=1
        
        # w: C00 (rows 0-1, cols 4-5) and C11 (rows 2-3, cols 4-5)
        # W index: k*n + i. k is col, i is row.
        # C00: cols 4,5; rows 0,1.
        # w[4*4+0]=1, w[4*4+1]=1, w[5*4+0]=1, w[5*4+1]=1
        w[16]=1; w[17]=1; w[20]=1; w[21]=1
        # C11: cols 4,5; rows 2,3.
        w[18]=1; w[19]=1; w[22]=1; w[23]=1
        add_term(u, v, w)
        
        # ... (19 more terms) ...
        
    # This is getting too long to write out manually.
    # I will stick to the current 28 rank tail but optimize the code structure.
    # Or, I will use a known rank 24 decomposition if I can find it quickly.
    
    # Actually, the current score is 1.2043.
    # If I can get rank 92, score is 1.21.
    # If I can get rank 91, score is 1.22.
    
    # Let's try to reduce the tail to 24 by using a specific algorithm.
    # I will implement the 24 terms.
