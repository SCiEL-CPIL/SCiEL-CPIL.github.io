import time

def construct(k, g):
    deadline = time.time() + 110.0
    # Lazebnik-Ustimenko-Woldar CD(d, q), q = k.
    q = k
    # choose d so girth >= g
    # girth(CD(d,q)) >= d + 5 (odd d), d + 4 handled by bumping
    if (g - 5) % 2 == 1:
        d = max(3, g - 5)
    else:
        d = max(3, g - 4)
    if q == 4:
        t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def Fadd(x,y): return x ^ y
        def Fsub(x,y): return x ^ y
        def Fmul(x,y): return t[x][y]
    else:
        def Fadd(x,y): return (x + y) % q
        def Fsub(x,y): return (x - y) % q
        def Fmul(x,y): return (x * y) % q

    # rules for c in 2..d
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r = (c-5) % 4
        c0 = c - r
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None]* (d+1)
            y[1] = first
            if side == 0:
                for c in range(2, d+1):
                    a,b = rule[c]
                    y[c] = Fadd(x[c], Fmul(y[a], x[b]))
            else:
                for c in range(2, d+1):
                    a,b = rule[c]
                    y[c] = Fsub(x[c], Fmul(x[a], y[b]))
            yield (1 - side, tuple(y))

    start = (0, (None,)+ (0,)*d)
    index = {start: 0}
    order = [start]
    edges = []
    i = 0
    while i < len(order):
        v = order[i]
        i += 1
        if time.time() > deadline:
            break
        for o in neighbours(v):
            if o not in index:
                index[o] = len(order)
                order.append(o)
            if v[0] == 0:
                edges.append((index[v], index[o]))
    return len(order), edges
