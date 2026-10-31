# EVOLVE-BLOCK-START
def construct(k, g):
    import random
    import math
    import numpy as np

    if k != 3:
        raise ValueError("This construction only works for k=3")
    
    if g % 2 == 0:
        moore_bound = 3 * (2 ** (g // 2 - 1) - 1)
    else:
        moore_bound = 3 * (2 ** ((g - 1) // 2 - 1) - 1) if (g - 1) // 2 >= 1 else 3
    
    m_min = max(1, math.ceil(moore_bound / 6))
    
    def has_short_cycle(m, V, max_half):
        """Check if there's a cycle of length < 2*max_half in the lift.
        V is 3x3 numpy int array.
        Returns True if a short cycle exists.
        """
        # Check 4-cycles (L=2): A_i -> B_j -> A_k -> B_l -> A_i
        # Voltage: V[i][j] - V[k][j] + V[k][l] - V[i][l] ≡ 0 mod m
        # with i != k, j != l
        for i in range(3):
            for k in range(3):
                if i == k:
                    continue
                for j in range(3):
                    for l in range(3):
                        if j == l:
                            continue
                        vol = (V[i][j] - V[k][j] + V[k][l] - V[i][l]) % m
                        if vol == 0:
                            return True
        
        # Check 6-cycles (L=3): A_i -> B_j -> A_k -> B_l -> A_m -> B_n -> A_i
        # Voltage: V[i][j] - V[k][j] + V[k][l] - V[m][l] + V[m][n] - V[i][n] ≡ 0 mod m
        if max_half >= 3:
            for i in range(3):
                for k in range(3):
                    for m_idx in range(3):
                        for j in range(3):
                            for l in range(3):
                                for n in range(3):
                                    vol = (V[i][j] - V[k][j] + V[k][l] - V[m_idx][l] + V[m_idx][n] - V[i][n]) % m
                                    if vol == 0:
                                        # Check it's a simple cycle (no repeated vertices)
                                        # Vertices: A_i, B_j, A_k, B_l, A_m, B_n
                                        # Need all A's distinct and all B's distinct
                                        if len({i, k, m_idx}) < 3 or len({j, l, n}) < 3:
                                            continue
                                        return True
        
        # Check 8-cycles (L=4)
        if max_half >= 4:
            # A_i -> B_j -> A_k -> B_l -> A_m -> B_n -> A_p -> B_q -> A_i
            # Voltage: V[i][j] - V[k][j] + V[k][l] - V[m][l] + V[m][n] - V[p][n] + V[p][q] - V[i][q] ≡ 0
            for i in range(3):
                for k in range(3):
                    for m_idx in range(3):
                        for p in range(3):
                            if len({i, k, m_idx, p}) < 4:
                                continue
                            for j in range(3):
                                for l in range(3):
                                    for n in range(3):
                                        for q in range(3):
                                            if len({j, l, n, q}) < 4:
                                                continue
                                            vol = (V[i][j] - V[k][j] + V[k][l] - V[m_idx][l] + V[m_idx][n] - V[p][n] + V[p][q] - V[i][q]) % m
                                            if vol == 0:
                                                return True
        
        # Check 10-cycles (L=5)
        if max_half >= 5:
            # This gets very expensive. For now, skip and rely on the fact that
            # if no 4,6,8-cycles exist, the girth is likely >= 10.
            # For girth 16, we need to check up to L=7.
            # Let's do a BFS-based check for longer cycles.
            n = 6 * m
            adj = [[] for _ in range(n)]
            for a in range(3):
                for b in range(3):
                    v = int(V[a][b])
                    for s in range(m):
                        u = 0 * 3 * m + a * m + s
                        w = 1 * 3 * m + b * m + (s + v) % m
                        adj[u].append(w)
                        adj[w].append(u)
            
            from collections import deque
            max_depth = max_half
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    if dist[u] >= max_depth:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < 2 * max_half:
                                return True
            return False
        
        return False
    
    random.seed(99999)
    
    max_m = m_min + 200
    
    for m in range(m_min, max_m + 1):
        n = 6 * m
        if n > 1500:
            break
        
        num_trials = 40 if m < 80 else 20
        
        for trial in range(num_trials):
            V = np.random.randint(0, m, size=(3, 3))
            
            if not has_short_cycle(m, V, g // 2):
                edges = []
                for i in range(3):
                    for j in range(3):
                        v = int(V[i][j])
                        for s in range(m):
                            u = 0 * 3 * m + i * m + s
                            w = 1 * 3 * m + j * m + (s + v) % m
                            edges.append((u, w))
                return n, edges
    
    # Fallback
    n = 6
    edges = [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)]
    return n, edges
# EVOLVE-BLOCK-END
