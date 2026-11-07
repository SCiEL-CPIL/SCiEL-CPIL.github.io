"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    from random import Random
    sys.setrecursionlimit(10000)

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to bitmasks
        # Colors are 0..4, so bits 0..4
        init_domains = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            init_domains[v] = mask
            
        assigned = [-1] * n
        domains = init_domains[:]
        
        def backtrack():
            min_v = -1
            min_size = 100
            for v in range(n):
                if assigned[v] == -1:
                    size = domains[v].bit_count()
                    if size < min_size:
                        min_size = size
                        min_v = v
                        if size == 0:
                            return False
                        if size == 1:
                            break
            
            if min_v == -1:
                return True
            
            d = domains[min_v]
            while d:
                lsb = d & (-d)
                c = lsb.bit_length() - 1
                
                assigned[min_v] = c
                saved_domains = domains[:]
                
                conflict = False
                for u in adj[min_v]:
                    if assigned[u] == -1:
                        if domains[u] & lsb:
                            domains[u] &= ~lsb
                            if domains[u] == 0:
                                conflict = True
                                break
                
                if not conflict:
                    if backtrack():
                        return True
                
                # Undo
                assigned[min_v] = -1
                domains[:] = saved_domains[:]
                
                d &= (d - 1)
                
            return False
        
        if backtrack():
            return assigned
        else:
            # Fallback to greedy
            deg = [len(adj[i]) for i in range(n)]
            order = sorted(range(n), key=lambda x: -deg[x])
            greedy_color = [-1] * n
            for v in order:
                taken = 0
                for u in adj[v]:
                    if greedy_color[u] != -1:
                        taken |= (1 << greedy_color[u])
                mask = init_domains[v] & ~taken
                if mask:
                    lsb = mask & (-mask)
                    greedy_color[v] = lsb.bit_length() - 1
                else:
                    lsb = init_domains[v] & (-init_domains[v])
                    greedy_color[v] = lsb.bit_length() - 1
            return greedy_color

    out = []
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    instances = [(20, 701), (40, 702), (60, 703), (100, 704)]
    
    for i, (splits, seed) in enumerate(instances):
        # Check if we have enough time for a full solve.
        # If not, we still need to produce a result.
        # The backtracker is fast, so we likely have time.
        
        n, edges, lists = instance(splits, seed)
        
        # Determine if we should use the full backtracker or just greedy.
        # Given the small size, backtracker is preferred.
        res = solve_instance(n, edges, lists)
        
        # Ensure types are correct (int, list)
        final_res = []
        for c in res:
            if c is None:
                final_res.append(0)
            else:
                final_res.append(int(c))
                
        out.append(final_res)
        
        # If we are running out of time, we can't do much else but ensure we finish the loop.
        # The loop will complete all 4 iterations.
        
    return out

# EVOLVE-BLOCK-END
