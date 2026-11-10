import random
import time

def construct():
    def make_grötzsch():
        edges = set()
        for i in range(5):
            edges.add((i, (i+1) % 5))
        for i in range(5):
            edges.add((5+i, i))
            edges.add((5+i, (i+1) % 5))
        for i in range(5):
            edges.add((10, 5+i))
        return edges
    
    edges = make_grötzsch()
    
    # Build adjacency set for Grötzsch graph (vertices 0-10)
    adj_set = set()
    for u, v in edges:
        adj_set.add((u, v))
        adj_set.add((v, u))
    
    # Find all independent sets of size >= 3 in Grötzsch graph
    def find_independent_sets():
        sets = []
        def backtrack(start, current):
            if len(current) >= 3:
                sets.append(frozenset(current))
            for i in range(start, 11):
                if all((i, j) not in adj_set for j in current):
                    current.append(i)
                    backtrack(i + 1, current)
                    current.pop()
        backtrack(0, [])
        return sets
    
    indep_sets = find_independent_sets()
    max_size = max(len(s) for s in indep_sets) if indep_sets else 0
    max_indep_sets = [s for s in indep_sets if len(s) == max_size]
    
    # Assign isolated vertices (11-15) to independent sets
    random.seed(42)
    assignments = [random.choice(max_indep_sets) for _ in range(5)]
    
    for i, iso in enumerate(range(11, 16)):
        for v in assignments[i]:
            edges.add((iso, v))
    
    # Verify triangle-free
    def is_triangle_free(edge_set):
        a = set()
        for u, v in edge_set:
            a.add((u, v))
            a.add((v, u))
        for u, v in edge_set:
            for w in range(16):
                if w != u and w != v and (u, w) in a and (v, w) in a:
                    return False
        return True
    
    assert is_triangle_free(edges)
    
    # Verify not 3-colorable
    def is_not_3_colorable(edge_set):
        adj_list = [[] for _ in range(16)]
        for u, v in edge_set:
            adj_list[u].append(v)
            adj_list[v].append(u)
        colors = [-1] * 16
        def backtrack(v):
            if v == 16:
                return True
            for c in range(3):
                if all(colors[u] != c for u in adj_list[v]):
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            return False
        colors[0] = 0
        return not backtrack(1)
    
    assert is_not_3_colorable(edges)
    
    # Local search to add more edges
    start_time = time.time()
    timeout = 110
    all_edges = [(i, j) for i in range(16) for j in range(i+1, 16)]
    
    best_edges = set(edges)
    best_count = len(best_edges)
    current_edges = set(edges)
    current_count = len(current_edges)
    
    def build_masks(es):
        masks = [0] * 16
        for u, v in es:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks
    
    current_masks = build_masks(current_edges)
    
    def is_not_3_colorable_masks(masks):
        colors = [-1] * 16
        def backtrack(v):
            if v == 16:
                return True
            forbidden = 0
            for u in range(16):
                if colors[u] >= 0 and (masks[v] >> u) & 1:
                    forbidden |= (1 << colors[u])
            for c in range(3):
                if not (forbidden >> c) & 1:
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            return False
        colors[0] = 0
        return not backtrack(1)
    
    temperature = 5.0
    cooling = 0.999
    
    while time.time() - start_time < timeout:
        if random.random() < 0.8:
            candidates = [e for e in all_edges if e not in current_edges]
            if not candidates:
                continue
            e = random.choice(candidates)
            u, v = e
            if current_masks[u] & current_masks[v]:
                continue
            new_masks = current_masks[:]
            new_masks[u] |= (1 << v)
            new_masks[v] |= (1 << u)
            if is_not_3_colorable_masks(new_masks):
                current_edges.add(e)
                current_masks = new_masks
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_edges = set(current_edges)
        else:
            if not current_edges:
                continue
            e = random.choice(list(current_edges))
            u, v = e
            new_masks = current_masks[:]
            new_masks[u] &= ~(1 << v)
            new_masks[v] &= ~(1 << u)
            current_edges.discard(e)
            current_masks = new_masks
            current_count -= 1
        temperature *= cooling
    
    return [list(e) for e in best_edges]
