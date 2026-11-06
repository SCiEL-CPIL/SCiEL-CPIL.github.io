import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def elapsed():
        return time.time() - start_time

    def add_edge_safe(adj, n, u, v):
        if adj[u][v]:
            return True
        common = 0
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                common += 1
                if common >= 2:
                    return False
        adj[u][v] = 1
        adj[v][u] = 1
        return True

    def has_independent_set_11(adj, n):
        # Branch and bound: find independent set of size 11
        # Independent set in G = clique in complement
        # Use bitmasks for speed
        # n <= 39, so use Python ints as bitmasks
        # comp_mask[v] = bitmask of vertices not adjacent to v in G (excluding v)
        comp_mask = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and not adj[i][j]:
                    comp_mask[i] |= (1 << j)

        found = [False]
        
        def bb(candidates_mask, current_size):
            if found[0]:
                return
            if current_size >= 11:
                found[0] = True
                return
            if current_size + bin(candidates_mask).count('1') < 11:
                return
            if elapsed() > deadline - 15:
                found[0] = True  # Conservative: assume exists
                return
            
            # Pick vertex with highest degree in candidate subgraph
            best_v = -1
            best_deg = -1
            temp = candidates_mask
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = bin(comp_mask[v] & candidates_mask).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v < 0:
                return
            
            # Include best_v
            new_mask = candidates_mask & comp_mask[best_v]
            bb(new_mask, current_size + 1)
            
            if found[0]:
                return
            
            # Exclude best_v
            new_mask = candidates_mask & ~(1 << best_v)
            bb(new_mask, current_size)

        bb((1 << n) - 1, 0)
        return found[0]

    best_n = 30
    best_edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        best_edges += [[a,b],[a,c],[b,c]]

    for target_n in [39, 38, 37, 36, 35, 34, 33, 32, 31]:
        if elapsed() > deadline - 50:
            break

        for attempt in range(200):
            if elapsed() > deadline - 50:
                break

            random.seed(attempt * 7919 + target_n * 104729)
            n = target_n
            adj = [[0]*n for _ in range(n)]

            edge_order = [(i,j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edge_order)

            for u, v in edge_order:
                add_edge_safe(adj, n, u, v)

            if not has_independent_set_11(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i][j]:
                            edges.append([i,j])
                best_n = n
                best_edges = edges
                break
        else:
            continue
        break

    return {'n': best_n, 'edges': best_edges}
