import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Initial graph: K_{3,13}
    edges = set()
    for u in range(3):
        for v in range(3, 16):
            edges.add((u, v))
            
    # Precompute adjacency for speed? 
    # Or just use the set.
    
    def is_q3(vertices):
        if len(vertices) != 8:
            return False
        
        # Check 12 edges
        sub_edges = []
        for u, v in itertools.combinations(vertices, 2):
            if (u, v) in edges or (v, u) in edges:
                sub_edges.append((u, v))
        
        if len(sub_edges) != 12:
            return False
        
        # Check 3-regular
        degrees = {v: 0 for v in vertices}
        for u, v in sub_edges:
            degrees[u] += 1
            degrees[v] += 1
        for d in degrees.values():
            if d != 3:
                return False
        
        # Check bipartite and count 4-cycles
        # Since it's 3-regular and bipartite on 8 vertices, it must be Q3.
        # We just need to verify bipartiteness.
        color = {}
        for v in vertices:
            if v not in color:
                color[v] = 0
                stack = [v]
                while stack:
                    curr = stack.pop()
                    for u, w in sub_edges:
                        if u == curr:
                            nxt = w
                        elif w == curr:
                            nxt = u
                        else:
                            continue
                        if nxt not in color:
                            color[nxt] = 1 - color[curr]
                            stack.append(nxt)
                        else:
                            if color[nxt] == color[curr]:
                                return False
        
        # If we reach here, it is a 3-regular bipartite graph on 8 vertices.
        # This is unique: Q3.
        return True

    all_pairs = set(itertools.combinations(range(16), 2))
    non_edges = list(all_pairs - edges)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        current_non_edges = [e for e in non_edges if e not in edges]
        for e in current_non_edges:
            if time.time() > deadline:
                break
            u, v = e
            remaining = [x for x in range(16) if x != u and x != v]
            safe = True
            # Add edge temporarily
            edges.add((u, v))
            
            for combo in itertools.combinations(remaining, 6):
                verts = set([u, v] + list(combo))
                if is_q3(verts):
                    safe = False
                    break
            
            if safe:
                changed = True
            else:
                edges.remove((u, v))

    return [list(e) for e in edges]
