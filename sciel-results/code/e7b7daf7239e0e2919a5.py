import time
import random
import itertools

def construct():
    # Fallback: Fano plane (7 edges)
    edges_fano = [
        [0, 1, 2], [0, 3, 4], [0, 5, 6],
        [1, 3, 5], [1, 4, 6], [2, 3, 6], [2, 4, 5]
    ]
    colors_fano = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Helper to check 2-colorability
    def is_2_colorable(edges):
        n = 9
        colors = [-1] * n
        
        def backtrack(v):
            if v == n:
                for e in edges:
                    if colors[e[0]] == colors[e[1]] == colors[e[2]]:
                        return False
                return True
            
            for c in [0, 1]:
                colors[v] = c
                valid = True
                for e in edges:
                    if v in e:
                        vals = [colors[x] for x in e]
                        if -1 not in vals and vals[0] == vals[1] == vals[2]:
                            valid = False
                            break
                if valid and backtrack(v + 1):
                    return True
            colors[v] = -1
            return False

        return backtrack(0)

    def find_3_chromatic(k):
        start = time.time()
        deadline = 10.0 # 10 seconds for search
        all_edges = list(itertools.combinations(range(9), 3))
        
        while time.time() < deadline:
            random.shuffle(all_edges)
            candidate = all_edges[:k]
            if not is_2_colorable(candidate):
                # Found a non-2-colorable hypergraph
                # We need a 3-coloring. 
                # Since it's not 2-colorable, it requires 3. 
                # We can just find any valid 3-coloring.
                # For now, we can use a simple greedy or backtracking for 3 colors.
                colors = [-1] * 9
                def backtrack_3(v):
                    if v == 9:
                        for e in candidate:
                            if colors[e[0]] == colors[e[1]] == colors[e[2]]:
                                return False
                        return True
                    for c in [0, 1, 2]:
                        colors[v] = c
                        valid = True
                        for e in candidate:
                            if v in e:
                                vals = [colors[x] for x in e]
                                if -1 not in vals and vals[0] == vals[1] == vals[2]:
                                    valid = False
                                    break
                        if valid and backtrack_3(v + 1):
                            return True
                    colors[v] = -1
                    return False
                
                if backtrack_3(0):
                    return [list(e) for e in candidate], colors
        return None

    # Try to find 5-edge solution
    res = find_3_chromatic(5)
    if res:
        edges, colors = res
        return {'edges': edges, 'colors': colors}
        
    # Try to find 6-edge solution
    res = find_3_chromatic(6)
    if res:
        edges, colors = res
        return {'edges': edges, 'colors': colors}
        
    # Fallback to Fano
    return {'edges': edges_fano, 'colors': colors_fano}
