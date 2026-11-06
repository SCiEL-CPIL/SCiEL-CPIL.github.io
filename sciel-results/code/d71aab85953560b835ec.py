import time

def construct():
    start_time = time.time()
    TIMEOUT = 115  # seconds, leave 5s margin
    
    def solve(n):
        edges = [(i,j) for i in range(n) for j in range(i+1, n)]
        m = len(edges)
        color = [[-1]*n for _ in range(n)]
        adj0 = [0]*n
        adj1 = [0]*n
        
        def common_nb(i, j, c):
            if c == 0:
                return adj0[i] & adj0[j]
            else:
                return adj1[i] & adj1[j]
        
        def has_k22(S, c):
            adj_c = adj0 if c == 0 else adj1
            verts = []
            s = S
            while s:
                lsb = s & (-s)
                v = lsb.bit_length() - 1
                verts.append(v)
                s ^= lsb
            k = len(verts)
            if k < 4:
                return False
            for ai in range(k):
                a = verts[ai]
                for bi in range(ai+1, k):
                    b = verts[bi]
                    if not (adj_c[a] >> b) & 1:
                        continue
                    common = adj_c[a] & adj_c[b] & S
                    common &= ~(1 << a) & ~(1 << b)
                    if common == 0:
                        continue
                    # Check if common has a color-c edge
                    cv = common
                    while cv:
                        lsb = cv & (-cv)
                        u = lsb.bit_length() - 1
                        rest = common & ~(1 << u)
                        if adj_c[u] & rest:
                            return True
                        cv ^= lsb
            return False
        
        def backtrack(idx):
            if time.time() - start_time > TIMEOUT:
                return False
            if idx == m:
                return True
            i, j = edges[idx]
            for c in [0, 1]:
                S = common_nb(i, j, c)
                if has_k22(S, c):
                    continue
                # Assign
                color[i][j] = color[j][i] = c
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                
                if backtrack(idx + 1):
                    return True
                
                if c == 0:
                    adj0[i] &= ~(1 << j)
                    adj0[j] &= ~(1 << i)
                else:
                    adj1[i] &= ~(1 << j)
                    adj1[j] &= ~(1 << i)
                color[i][j] = color[j][i] = -1
            return False
        
        if backtrack(0):
            result = []
            for i in range(n):
                for j in range(i+1, n):
                    result.append(color[i][j])
            return result
        return None
    
    for n in [30, 29]:
        result = solve(n)
        if result is not None:
            return {'n': n, 'colors': result}
    
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}
