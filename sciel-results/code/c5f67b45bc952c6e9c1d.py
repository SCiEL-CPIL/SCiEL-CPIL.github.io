import numpy as np
import time
import random
from collections import Counter
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    n = 20
    num_colors = 20
    
    # Precompute all pairs
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    def get_valid_colors(adj, colors, u, v):
        """
        Returns a set of colors that can be assigned to edge (u, v)
        such that all new 4-cycles formed are rainbow.
        """
        forbidden = set()
        bad_path_exists = False
        
        # Check all paths u-a-b-v
        for a in range(n):
            if a == u or a == v:
                continue
            if not adj[u, a]:
                continue
            for b in range(n):
                if b == u or b == v or b == a:
                    continue
                if not adj[a, b]:
                    continue
                if not adj[b, v]:
                    continue
                
                # Path u-a-b-v exists
                c1 = colors[u, a]
                c2 = colors[a, b]
                c3 = colors[b, v]
                
                # For the cycle u-a-b-v-u to be rainbow, c1, c2, c3 must be distinct
                # and the new color c must be distinct from c1, c2, c3
                if c1 == c2 or c1 == c3 or c2 == c3:
                    bad_path_exists = True
                    break
                else:
                    forbidden.add(c1)
                    forbidden.add(c2)
                    forbidden.add(c3)
            if bad_path_exists:
                break
        
        if bad_path_exists:
            return set()
        
        valid = set(range(num_colors)) - forbidden
        return valid
    
    def count_edges(adj):
        return int(np.sum(adj) // 2)
    
    def is_valid(adj, colors):
        """
        Verify the entire graph is valid.
        """
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u, v]:
                    continue
                # Check all 4-cycles containing (u,v)
                # Cycle u-v-w-x-u
                # v-w-x-u is a path of length 3 from v to u
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if not adj[v, w]:
                        continue
                    for x in range(n):
                        if x == u or x == v or x == w:
                            continue
                        if not adj[w, x]:
                            continue
                        if not adj[x, u]:
                            continue
                        
                        c1 = colors[u, v]
                        c2 = colors[v, w]
                        c3 = colors[w, x]
                        c4 = colors[x, u]
                        
                        if len({c1, c2, c3, c4}) < 4:
                            return False
        return True

    best_edges = 0
    best_adj = np.zeros((n, n), dtype=np.uint8)
    best_colors = np.zeros((n, n), dtype=np.int32)
    
    # Multiple restarts
    while time.time() < deadline - 5:
        adj = np.zeros((n, n), dtype=np.uint8)
        colors = np.zeros((n, n), dtype=np.int32)
        
        # Random order of edges
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            valid_colors = get_valid_colors(adj, colors, u, v)
            if valid_colors:
                # Pick a random valid color
                c = random.choice(list(valid_colors))
                adj[u, v] = 1
                adj[v, u] = 1
                colors[u, v] = c
                colors[v, u] = c
        
        current_edges = count_edges(adj)
        
        # Local Search / Repair Phase
        # Try to add missing edges by recoloring
        missing_edges = [e for e in all_edges if not adj[e[0], e[1]]]
        random.shuffle(missing_edges)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try to add a few missing edges
            for u, v in missing_edges[:10]: # Limit work per iteration
                if adj[u, v]:
                    continue
                
                valid_colors = get_valid_colors(adj, colors, u, v)
                if valid_colors:
                    c = random.choice(list(valid_colors))
                    adj[u, v] = 1
                    adj[v, u] = 1
                    colors[u, v] = c
                    colors[v, u] = c
                    current_edges += 1
                    improved = True
                    # Remove from missing list
                    if (u, v) in missing_edges:
                        missing_edges.remove((u, v))
                    elif (v, u) in missing_edges:
                        missing_edges.remove((v, u))
                
                else:
                    # Try to fix by recoloring edges in conflicting paths
                    # Find a conflicting path
                    conflict_found = False
                    for a in range(n):
                        if a == u or a == v or not adj[u, a]:
                            continue
                        for b in range(n):
                            if b == u or b == v or b == a or not adj[a, b] or not adj[b, v]:
                                continue
                            
                            c1 = colors[u, a]
                            c2 = colors[a, b]
                            c3 = colors[b, v]
                            
                            # If colors are not distinct, we have a structural conflict for this path
                            # We need to recolor one of c1, c2, c3 to make them distinct
                            # Or if they are distinct, but forbidden set is full, we need to change one to free up a color?
                            # Actually, if they are distinct, they just contribute to forbidden set.
                            # If they are NOT distinct, the edge (u,v) can NEVER be added unless we recolor.
                            
                            if c1 == c2 or c1 == c3 or c2 == c3:
                                conflict_found = True
                                # Try to recolor one of the edges
                                # Pick random edge from (u,a), (a,b), (b,v)
                                target_edges = [(u, a), (a, b), (b, v)]
                                random.shuffle(target_edges)
                                recolor_success = False
                                for tu, tv in target_edges:
                                    old_c = colors[tu, tv]
                                    # Try all colors
                                    for new_c in range(num_colors):
                                        if new_c == old_c:
                                            continue
                                        # Check if recoloring (tu, tv) to new_c is valid for all cycles containing (tu, tv)
                                        # This is expensive. 
                                        # Simple check: does it create a non-rainbow cycle?
                                        # We can just apply and check locally? 
                                        # Better: use get_valid_colors logic for the neighbors?
                                        # Let's just try it and verify the specific cycle and its neighbors.
                                        
                                        colors[tu, tv] = new_c
                                        colors[tv, tu] = new_c
                                        
                                        # Check if this recoloring broke any existing 4-cycles
                                        valid_recolor = True
                                        # Check cycles involving (tu, tv)
                                        # (tu, tv) is part of cycles tu-tv-x-y-tu
                                        for x in range(n):
                                            if x == tu or x == tv or not adj[tv, x]:
                                                continue
                                            for y in range(n):
                                                if y == tu or y == tv or y == x or not adj[x, y] or not adj[y, tu]:
                                                    continue
                                                c_a = colors[tu, tv]
                                                c_b = colors[tv, x]
                                                c_c = colors[x, y]
                                                c_d = colors[y, tu]
                                                if len({c_a, c_b, c_c, c_d}) < 4:
                                                    valid_recolor = False
                                                    break
                                            if not valid_recolor:
                                                break
                                        
                                        if valid_recolor:
                                            recolor_success = True
                                            break
                                        else:
                                            colors[tu, tv] = old_c
                                            colors[tv, tu] = old_c
                                        
                                        if recolor_success:
                                            break
                                
                                if recolor_success:
                                    improved = True
                                    break # Restart the check for (u,v)
                            else:
                                # Colors are distinct, so (u,v) is blocked by full forbidden set
                                # This means we need to recolor an edge to change the forbidden set
                                # This is harder. Skip for now or try random recolor.
                                pass
                            
                            if conflict_found:
                                break
                        if conflict_found:
                            break
                
                # Update missing edges list if we added an edge
                if adj[u, v]:
                    if (u, v) in missing_edges:
                        missing_edges.remove((u, v))
                    elif (v, u) in missing_edges:
                        missing_edges.remove((v, u))
        
        # Final validation
        if is_valid(adj, colors):
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj.copy()
                best_colors = colors.copy()
        
        # If we found a very good solution, we can stop early
        if best_edges > 150: # Heuristic threshold
            break

    # Construct output
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v] == 1:
                c = int(best_colors[u, v])
                result.append([u, v, c])
    
    return result
