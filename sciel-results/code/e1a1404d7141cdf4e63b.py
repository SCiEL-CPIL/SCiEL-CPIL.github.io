import time
import random

def construct():
    deadline = time.time() + 145.0  # Leave 5s margin
    
    def construct_32():
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r

        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
            
        n = 32
        M = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                M[i][j] = log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3
                M[j][i] = M[i][j]
        return M

    def local_search(n, time_limit):
        start_time = time.time()
        M = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                M[i][j] = random.randint(0, 3)
                M[j][i] = M[i][j]
                
        def init_adj(M, n):
            adj = [[0] * 4 for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    c = M[i][j]
                    adj[i][c] |= (1 << j)
                    adj[j][c] |= (1 << i)
            return adj

        def update_adj(M, adj, i, j, old_c, new_c, n):
            adj[i][old_c] &= ~(1 << j)
            adj[j][old_c] &= ~(1 << i)
            adj[i][new_c] |= (1 << j)
            adj[j][new_c] |= (1 << i)
            M[i][j] = new_c
            M[j][i] = new_c

        def count_triangles_adj(adj, i, j, c, n):
            mask = adj[i][c] & adj[j][c]
            return bin(mask).count('1')

        def find_triangle(M, adj, n):
            # Try up to 50 random edges
            for _ in range(50):
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i == j: continue
                if i > j: i, j = j, i
                c = M[i][j]
                mask = adj[i][c] & adj[j][c]
                if mask:
                    k = (mask & -mask).bit_length() - 1
                    return (i, j, k)
            # Full scan
            for i in range(n):
                for j in range(i+1, n):
                    c = M[i][j]
                    mask = adj[i][c] & adj[j][c]
                    if mask:
                        k = (mask & -mask).bit_length() - 1
                        return (i, j, k)
            return None

        adj = init_adj(M, n)
        
        while time.time() < time_limit:
            tri = find_triangle(M, adj, n)
            if not tri:
                return True, M
            
            i, j, k = tri
            c_ij = M[i][j]
            c_ik = M[i][k]
            c_jk = M[j][k]
            
            best_delta = float('inf')
            best_move = None
            
            # Try changing (i,j)
            for new_c in range(4):
                if new_c == c_ij: continue
                new_cost = count_triangles_adj(adj, i, j, new_c, n)
                old_cost = count_triangles_adj(adj, i, j, c_ij, n)
                delta = new_cost - old_cost
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j, c_ij, new_c)
            
            # Try changing (i,k)
            for new_c in range(4):
                if new_c == c_ik: continue
                new_cost = count_triangles_adj(adj, i, k, new_c, n)
                old_cost = count_triangles_adj(adj, i, k, c_ik, n)
                delta = new_cost - old_cost
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, k, c_ik, new_c)
            
            # Try changing (j,k)
            for new_c in range(4):
                if new_c == c_jk: continue
                new_cost = count_triangles_adj(adj, j, k, new_c, n)
                old_cost = count_triangles_adj(adj, j, k, c_jk, n)
                delta = new_cost - old_cost
                if delta < best_delta:
                    best_delta = delta
                    best_move = (j, k, c_jk, new_c)
            
            if not best_move:
                continue
                
            u, v, old_c, new_c = best_move
            
            if best_delta < 0:
                accept = True
            elif best_delta == 0:
                accept = random.random() < 0.5
            else:
                accept = random.random() < 0.1
                
            if accept:
                update_adj(M, adj, u, v, old_c, new_c, n)
            else:
                # Random walk step to escape local minimum
                u_rand = random.randint(0, n-1)
                v_rand = random.randint(0, n-1)
                if u_rand != v_rand:
                    if u_rand > v_rand: u_rand, v_rand = v_rand, u_rand
                    old_c_rand = M[u_rand][v_rand]
                    new_c_rand = random.choice([c for c in range(4) if c != old_c_rand])
                    update_adj(M, adj, u_rand, v_rand, old_c_rand, new_c_rand, n)

        return False, M

    # Try 50
    if time.time() < deadline - 5:
        success, M50 = local_search(50, time.time() + 130.0)
        if success:
            return {'n': 50, 'colors': [M50[i][j] for i in range(50) for j in range(i+1, 50)]}
            
    # Try 45
    if time.time() < deadline - 5:
        success, M45 = local_search(45, time.time() + 10.0)
        if success:
            return {'n': 45, 'colors': [M45[i][j] for i in range(45) for j in range(i+1, 45)]}
            
    # Try 40
    if time.time() < deadline - 2:
        success, M40 = local_search(40, time.time() + 2.0)
        if success:
            return {'n': 40, 'colors': [M40[i][j] for i in range(40) for j in range(i+1, 40)]}
            
    # Fallback to 32
    M32 = construct_32()
    return {'n': 32, 'colors': [M32[i][j] for i in range(32) for j in range(i+1, 32)]}
