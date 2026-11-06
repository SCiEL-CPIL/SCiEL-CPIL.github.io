def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    adj = [0] * N
    for i in range(N):
        a = i + 1
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) != 0:
                mask |= (1 << j)
        adj[i] = mask
    
    best_clique = []
    best_size = 0
    
    # Greedy initialization
    P = (1 << N) - 1
    R = []
    while P:
        # Pick vertex with max degree in P
        # Degree of v in P is (P & adj[v]).bit_count()
        # This is O(N) per step, so O(N^2) total.
        max_deg = -1
        max_v = -1
        temp_P = P
        while temp_P:
            lsb = temp_P & -temp_P
            v = lsb.bit_length() - 1
            deg = (P & adj[v]).bit_count()
            if deg > max_deg:
                max_deg = deg
                max_v = v
            temp_P &= ~(1 << v)
        
        if max_v == -1:
            break
            
        R.append(max_v)
        P &= ~adj[max_v]
        P &= ~(1 << max_v)
        
    if len(R) > best_size:
        best_size = len(R)
        best_clique = R[:]
        
    def expand(R, P):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        
        if P == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R[:]
            return
        
        # Quick bound
        if len(R) + P.bit_count() <= best_size:
            return
        
        # Greedy Coloring
        uncolored = P
        color = 0
        bounds = {}
        
        while uncolored:
            color += 1
            independent_set = 0
            temp = uncolored
            
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                
                if (adj[v] & independent_set) == 0:
                    independent_set |= (1 << v)
                    temp &= ~(adj[v] | (1 << v))
                else:
                    temp &= ~(1 << v)
            
            temp_is = independent_set
            while temp_is:
                lsb = temp_is & -temp_is
                v = lsb.bit_length() - 1
                bounds[v] = color
                temp_is &= ~(1 << v)
                
            uncolored &= ~independent_set
        
        # Order vertices by bound
        verts = []
        temp_P = P
        while temp_P:
            lsb = temp_P & -temp_P
            v = lsb.bit_length() - 1
            verts.append((v, bounds[v]))
            temp_P &= ~(1 << v)
            
        verts.sort(key=lambda x: x[1], reverse=True)
        
        for v, b in verts:
            if len(R) + b <= best_size:
                break
            
            new_R = R + [v]
            new_P = P & adj[v]
            expand(new_R, new_P)
            
            P &= ~(1 << v)
            
            if time.time() > deadline:
                return

    init_P = (1 << N) - 1
    expand([], init_P)
    
    result = [v + 1 for v in best_clique]
    
    if not result:
        return list(range(1, 257, 2))
        
    return result
