import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    BIT_IDX = {1 << i: i for i in range(20)}

    def solve(target_n):
        adj = [[0] * target_n for _ in range(4)]
        edges = []
        for i in range(target_n):
            for j in range(i + 1, target_n):
                edges.append((i, j))
        num_edges = len(edges)
        edge_color = [-1] * num_edges
        edge_to_idx = {p: i for i, p in enumerate(edges)}

        def _check_c6(u, v, c):
            nu = adj[c][u]
            nv = adj[c][v]
            a = nu
            while a:
                a_bit = a & -a
                a_idx = BIT_IDX[a_bit]
                a ^= a_bit
                e = nv
                while e:
                    e_bit = e & -e
                    e_idx = BIT_IDX[e_bit]
                    e ^= e_bit
                    na = adj[c][a_idx]
                    ne = adj[c][e_idx]
                    b = na
                    while b:
                        b_bit = b & -b
                        b_idx = BIT_IDX[b_bit]
                        b ^= b_bit
                        if adj[c][b_idx] & ne:
                            return True
            return False

        def _backtrack(idx):
            if idx == num_edges:
                return True
            u, v = edges[idx]
            e_idx = idx
            
            # Symmetry breaking
            if idx == 0:
                possible_colors = [0]
            elif u == 0:
                if v == 1: possible_colors = [0]
                elif v == 2: possible_colors = [0, 1]
                elif v == 3: possible_colors = [0, 1, 2]
                else: possible_colors = [0, 1, 2, 3]
            else:
                possible_colors = [0, 1, 2, 3]
                
            for c in possible_colors:
                if _check_c6(u, v, c):
                    continue
                
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                edge_color[e_idx] = c
                
                if _backtrack(idx + 1):
                    return True
                    
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                edge_color[e_idx] = -1
                
            return False

        if _backtrack(0):
            final_colors = []
            for idx in range(num_edges):
                final_colors.append(edge_color[idx])
            return {'n': target_n, 'colors': final_colors}
        return None

    for n in [20, 19, 18, 17]:
        if time.time() > deadline - 5:
            break
        res = solve(n)
        if res:
            return res

    # Fallback
    return {'n': 1, 'colors': []}
