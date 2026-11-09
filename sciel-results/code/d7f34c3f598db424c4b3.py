import random
import time

def construct():
    n = 20
    vertices = list(range(n))
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    # Precompute 4-cycles for each edge
    # For edge (u, v), cycles are u-v-w-x-u
    # w, x are distinct, not u, not v
    # Other edges: (v, w), (w, x), (x, u)
    cycles_for_edge = {e: [] for e in edges}
    for u, v in edges:
        others = [x for x in vertices if x != u and x != v]
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                w = others[i]
                x = others[j]
                # Cycle u-v-w-x-u
                # Edges: (u,v), (v,w), (w,x), (x,u)
                # We are adding (u,v). The others are (v,w), (w,x), (x,u).
                # Ensure these edges exist in the graph (they do, it's K_n)
                e1 = (min(v, w), max(v, w))
                e2 = (min(w, x), max(w, x))
                e3 = (min(x, u), max(x, u))
                cycles_for_edge[(u, v)].append((e1, e2, e3))
    
    def try_greedy():
        random.shuffle(edges)
        color_map = {}
        for e in edges:
            u, v = e
            valid_colors = set(range(20))
            for (e1, e2, e3) in cycles_for_edge[e]:
                # Check if e1, e2, e3 are already colored
                if e1 in color_map and e2 in color_map and e3 in color_map:
                    c1 = color_map[e1]
                    c2 = color_map[e2]
                    c3 = color_map[e3]
                    # If the path has duplicate colors, it's a dead end for this cycle
                    # But we are adding the last edge. If the path has duplicates,
                    # NO color for the last edge will make the cycle rainbow.
                    if c1 == c2 or c2 == c3 or c1 == c3:
                        valid_colors = set()
                        break
                    else:
                        # The path has 3 distinct colors.
                        # The new color must be different from c1, c2, c3.
                        valid_colors.discard(c1)
                        valid_colors.discard(c2)
                        valid_colors.discard(c3)
            
            if not valid_colors:
                return None
            
            # Pick a random valid color
            c = random.choice(list(valid_colors))
            color_map[e] = c
        
        return color_map

    start_time = time.time()
    best_result = None
    
    # Try to find a full coloring
    while time.time() - start_time < 110:
        res = try_greedy()
        if res is not None:
            # Verify the solution
            valid = True
            # Check all 4-cycles
            # For speed, we can just trust the greedy if it didn't fail, 
            # but a final check is safer.
            # Actually, the greedy check is sufficient if implemented correctly.
            # Let's do a quick verification on a subset or all.
            # Number of 4-cycles is 4845.
            for u in range(n):
                for v in range(u + 1, n):
                    for w in range(v + 1, n):
                        for x in range(w + 1, n):
                            # Check if (u,v), (v,w), (w,x), (x,u) form a cycle
                            # In K_n, any 4 distinct vertices form 3 cycles.
                            # We just need to check if the edges exist. They do.
                            # Get colors
                            c1 = res.get((min(u, v), max(u, v)))
                            c2 = res.get((min(v, w), max(v, w)))
                            c3 = res.get((min(w, x), max(w, x)))
                            c4 = res.get((min(x, u), max(x, u)))
                            # Note: The cycle is u-v-w-x-u.
                            # Edges are (u,v), (v,w), (w,x), (x,u).
                            if len({c1, c2, c3, c4}) != 4:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if valid:
                best_result = res
                break
    
    if best_result is None:
        # Fallback to star
        return [[0, v, 0] for v in range(1, n)]
    
    # Format output
    out = []
    for e, c in best_result.items():
        u, v = e
        out.append([u, v, c])
    return out
