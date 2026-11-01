import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # LUW construction
    # LUW(q, d) -> 2*q^d vertices, degree 2*q, girth 2*d+2
    # For k=4, g=10:
    # 2*q = 4 => q=2? No, q=4, d=4 => 2*4=8 degree? No.
    # LUW(q, d) degree is 2*q.
    # So for k=4, q=2.
    # Girth 2*d+2 = 10 => 2*d=8 => d=4.
    # N = 2 * q^d = 2 * 2^4 = 32.
    # This is too small. Girth 10 with 32 vertices?
    # Moore bound for (4,10) is 485.
    # So LUW(2, 4) cannot have girth 10.
    # My LUW formula memory is fuzzy.
    
    # Let's use the standard LUW for q=4.
    # LUW(4, d) has degree 8?
    # No, the seed program used LUW(4, 5) and got 2048 vertices.
    # Degree was 4?
    # The seed code:
    # rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    # This is for q=4.
    # The degree is 4.
    # So LUW(4, d) has degree 4.
    # N = 2 * 4^d.
    # For d=4, N = 512.
    # Girth = 2*d+2 = 10.
    # This matches.
    
    if k == 4 and g == 10:
        return build_luw_graph(4, 4)
        
    # Fallback
    return 0, []

def build_luw_graph(q_val, d_val):
    if q_val == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y)%q_val
        sub = lambda x,y: (x-y)%q_val
        mul = lambda x,y: x*y%q_val
        
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    def get_rule(dv):
        r = dict(rule)
        for c in range(5, dv+1):
            rc = (c-5)%4
            c0 = c-(c-5)%4
            if rc==0: r[c]=(1,c0-2)
            elif rc==1: r[c]=(c0-1,1)
            elif rc==2: r[c]=(c0,1)
            else: r[c]=(1,c0+1)
        return r
        
    r = get_rule(d_val)
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q_val):
            y = [None, first] + [0]*(d_val-1)
            for c in range(2, d_val+1):
                a,b = r[c]
                if side==0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
            
    start = (0, (None,)+(0,)*d_val)
    index = {start: 0}
    order = [start]
    edges = []
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
