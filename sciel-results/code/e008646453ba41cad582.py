import random
import itertools

def construct():
    n = 20
    best_edges = []
    
    for _ in range(5): # 5 restarts
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                edges.append([u, v, random.randint(0, 19)])
        
        edge_to_idx = {}
        for i, (u, v, _) in enumerate(edges):
            edge_to_idx[(u, v)] = i
            edge_to_idx[(v, u)] = i
            
        cycles = []
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            
            e1 = edge_to_idx.get((a, b))
            e2 = edge_to_idx.get((b, c))
            e3 = edge_to_idx.get((c, d))
            e4 = edge_to_idx.get((d, a))
            if None not in (e1, e2, e3, e4):
                cycles.append((e1, e2, e3, e4))
                
            e1 = edge_to_idx.get((a, b))
            e2 = edge_to_idx.get((b, d))
            e3 = edge_to_idx.get((d, c))
            e4 = edge_to_idx.get((c, a))
            if None not in (e1, e2, e3, e4):
                cycles.append((e1, e2, e3, e4))
                
            e1 = edge_to_idx.get((a, c))
            e2 = edge_to_idx.get((c, b))
            e3 = edge_to_idx.get((b, d))
            e4 = edge_to_idx.get((d, a))
            if None not in (e1, e2, e3, e4):
                cycles.append((e1, e2, e3, e4))
                
        active = [True] * len(edges)
        conflict_count = [0] * len(edges)
        is_bad = [False] * len(cycles)
        
        def is_rainbow(cycle_edges):
            colors = [edges[e][2] for e in cycle_edges]
            return len(set(colors)) == 4
            
        # Precompute edge_to_cycles
        edge_to_cycles = [[] for _ in range(len(edges))]
        for i, cycle in enumerate(cycles):
            for e in cycle:
                edge_to_cycles[e].append(i)
                
        # Initialize
        for i, cycle in enumerate(cycles):
            if all(active[e] for e in cycle):
                if not is_rainbow(cycle):
                    is_bad[i] = True
                    for e in cycle:
                        conflict_count[e] += 1
                        
        # Greedy Removal
        while True:
            max_conf = -1
            edge_to_remove = -1
            for i in range(len(edges)):
                if active[i] and conflict_count[i] > max_conf:
                    max_conf = conflict_count[i]
                    edge_to_remove = i
                    
            if edge_to_remove == -1:
                break
                
            active[edge_to_remove] = False
            
            for cycle_idx in edge_to_cycles[edge_to_remove]:
                if is_bad[cycle_idx]:
                    cycle = cycles[cycle_idx]
                    for e in cycle:
                        if e != edge_to_remove and active[e]:
                            conflict_count[e] -= 1
                is_bad[cycle_idx] = False
                
        result = []
        for i, (u, v, c) in enumerate(edges):
            if active[i]:
                result.append([u, v, c])
                
        if len(result) > len(best_edges):
            best_edges = result
            
    return best_edges
