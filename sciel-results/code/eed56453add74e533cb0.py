import time
import random

N = 40

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    base_edges = [(i, (i+1) % N) for i in range(N)]
    base_edge_set = set()
    for u, v in base_edges:
        base_edge_set.add((u, v))
        base_edge_set.add((v, u))

    def build_graph(chords):
        edges = set(base_edge_set)
        for u, v in chords:
            edges.add((u, v))
            edges.add((v, u))
        return edges

    def find_cycle(k, edges_set, adj):
        if k == N:
            return list(range(N))
        
        # For small k, we can use a faster check
        # But DFS is okay
        
        for start in range(N):
            visited = [False] * N
            path = [start]
            visited[start] = True
            
            def dfs(node, depth):
                if depth == k:
                    if (node, start) in edges_set:
                        return path[:]
                    return None
                
                # Pruning: if remaining depth > possible path length?
                # Hard to estimate.
                
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        path.append(neighbor)
                        
                        res = dfs(neighbor, depth + 1)
                        if res:
                            return res
                        
                        path.pop()
                        visited[neighbor] = False
                return None
            
            res = dfs(start, 1)
            if res:
                return res
        return None

    def is_pancyclic(edges_set):
        adj = {i: set() for i in range(N)}
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
            
        cycles = {}
        for k in range(3, N + 1):
            if time_left() < 2:
                return None
            c = find_cycle(k, edges_set, adj)
            if c is None:
                return None
            cycles[k] = c
        return cycles

    def try_remove_edges(edges_set):
        current_edges = set(edges_set)
        # Try to remove non-base edges first
        non_base = [e for e in current_edges if e not in base_edge_set]
        random.shuffle(non_base)
        
        for u, v in non_base:
            if (u, v) not in current_edges:
                continue
            if time_left() < 2:
                break
                
            new_edges = set(current_edges)
            new_edges.discard((u, v))
            new_edges.discard((v, u))
            
            if is_pancyclic(new_edges):
                current_edges = new_edges
                
        return current_edges

    best_edges = None
    best_cycles = None
    best_score = float('inf')
    
    # Try different strategies
    # 1. Random chords
    for num_chords in [8, 10, 12, 15]:
        if time_left() < 10:
            break
            
        for trial in range(100):
            if time_left() < 5:
                break
                
            chords = []
            used = set()
            # Bias towards short chords for small cycles
            for _ in range(num_chords):
                u = random.randint(0, N-1)
                # Pick distance
                dist = random.choice([2, 3, 4, 5, 10, 15, 20])
                v = (u + dist) % N
                if u == v: continue
                if (u, v) in base_edge_set: continue
                pair = (min(u,v), max(u,v))
                if pair in used: continue
                used.add(pair)
                chords.append((u, v))
                
            if len(chords) < num_chords:
                continue
                
            edges_set = build_graph(chords)
            cycles = is_pancyclic(edges_set)
            
            if cycles:
                pruned = try_remove_edges(edges_set)
                final_cycles = is_pancyclic(pruned)
                if final_cycles:
                    score = len(pruned) - N
                    if score < best_score:
                        best_score = score
                        best_edges = pruned
                        best_cycles = final_cycles
                        if best_score <= 5:
                            return {
                                'edges': list(best_edges),
                                'cycles': [best_cycles[k] for k in range(3, N+1)]
                            }
                            
    # Fallback to Seed
    if best_edges is None:
        seed_edges = [(0, i) for i in range(1, N)] + [(i, i+1) for i in range(1, N-1)]
        seed_edges_set = set()
        for u, v in seed_edges:
            seed_edges_set.add((u, v))
            seed_edges_set.add((v, u))
        seed_cycles = {k: list(range(k)) for k in range(3, N+1)}
        return {
            'edges': list(seed_edges_set),
            'cycles': [seed_cycles[k] for k in range(3, N+1)]
        }
    else:
        return {
            'edges': list(best_edges),
            'cycles': [best_cycles[k] for k in range(3, N+1)]
        }
