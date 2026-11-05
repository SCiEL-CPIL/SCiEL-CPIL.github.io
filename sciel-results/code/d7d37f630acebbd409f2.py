"""Strong edge colourings of three fixed bounded-degree graphs."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_bitmasks(edges, adj):
        """Build conflict as list of bitmasks for fast checking."""
        m = len(edges)
        if m == 0:
            return []
        
        conflict = [0] * m
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # Check shared vertices
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                    continue
                # Check if any endpoint of e1 is adjacent to any endpoint of e2
                if (u2 in adj.get(u1, set()) or v2 in adj.get(u1, set()) or
                    u2 in adj.get(v1, set()) or v2 in adj.get(v1, set())):
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        
        return conflict

    def dsatur_greedy(conflict, m):
        """
        DSATUR-inspired greedy coloring:
        Pick the uncolored edge with the most distinct conflicting colors (saturation),
        breaking ties by degree in the conflict graph.
        """
        color = [-1] * m
        num_colors = 0
        
        for _ in range(m):
            best_edge = -1
            best_sat = -1
            best_deg = -1
            
            for i in range(m):
                if color[i] != -1:
                    continue
                # Count distinct colors among neighbors
                sat_colors = set()
                for j in range(m):
                    if (conflict[i] >> j) & 1 and color[j] != -1:
                        sat_colors.add(color[j])
                sat = len(sat_colors)
                deg = bin(conflict[i]).count('1')
                
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_edge = i
            
            if best_edge == -1:
                break
            
            # Assign the smallest available color
            used_colors = set()
            for j in range(m):
                if (conflict[best_edge] >> j) & 1 and color[j] != -1:
                    used_colors.add(color[j])
            
            c = 0
            while c in used_colors:
                c += 1
            
            color[best_edge] = c
            if c >= num_colors:
                num_colors = c + 1
        
        return color, num_colors

    def try_remove_color(color, conflict, target_color, m):
        """
        Try to eliminate target_color by reassigning its edges.
        Returns True if successful (mutates color in place).
        """
        target_edges = [i for i in range(m) if color[i] == target_color]
        if not target_edges:
            return True
        
        # Sort by fewest conflicts first (easier to reassign)
        target_edges.sort(key=lambda i: bin(conflict[i]).count('1'))
        
        for idx in target_edges:
            # Find a valid color for idx
            neighbor_colors = set()
            for j in range(m):
                if (conflict[idx] >> j) & 1:
                    neighbor_colors.add(color[j])
            
            found = False
            for c in sorted(range(max(color) + 1), key=lambda x: abs(x - target_color)):
                if c == target_color:
                    continue
                if c not in neighbor_colors:
                    color[idx] = c
                    found = True
                    break
            
            if not found:
                return False
        
        return True

    def local_search(color, conflict, m, max_iters=5000):
        """
        Local search: repeatedly try to remove the highest color.
        Uses reassignment and swap moves.
        """
        best = color[:]
        best_nc = max(best) + 1 if best else 0
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            cur_nc = max(best) + 1 if best else 0
            if cur_nc <= 1:
                break
            
            target = cur_nc - 1
            target_edges = [i for i in range(m) if best[i] == target]
            
            if not target_edges:
                continue
            
            # Try direct reassignment
            trial = best[:]
            if try_remove_color(trial, conflict, target, m):
                best = trial
                continue
            
            # Try swap-based: move one target edge to another color, then retry
            swapped = False
            for idx in target_edges:
                # Try moving idx to each other color
                for c in range(target):
                    # Check if idx can take color c
                    neighbor_colors = set()
                    for j in range(m):
                        if (conflict[idx] >> j) & 1:
                            neighbor_colors.add(best[j])
                    if c not in neighbor_colors:
                        # Move idx to color c
                        best[idx] = c
                        swapped = True
                        break
                if swapped:
                    break
            
            if swapped:
                # Retry removing target color
                trial = best[:]
                if try_remove_color(trial, conflict, target, m):
                    best = trial
            else:
                # No swap found; try random perturbation
                if target_edges:
                    idx = random.choice(target_edges)
                    # Try all colors
                    neighbor_colors = set()
                    for j in range(m):
                        if (conflict[idx] >> j) & 1:
                            neighbor_colors.add(best[j])
                    available = [c for c in range(max(best) + 1) if c != target and c not in neighbor_colors]
                    if available:
                        best[idx] = random.choice(available)
                        trial = best[:]
                        if try_remove_color(trial, conflict, target, m):
                            best = trial
        
        return best

    def polish(color, conflict, m):
        """
        Final polish: multiple passes trying to remove colors from high to low.
        """
        best = color[:]
        
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            
            cur_nc = max(best) + 1 if best else 0
            
            for target in range(cur_nc - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                
                target_edges = [i for i in range(m) if best[i] == target]
                if not target_edges:
                    continue
                
                # Try reassignment
                trial = best[:]
                if try_remove_color(trial, conflict, target, m):
                    best = trial
                    break
        
        return best

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        
        if m == 0:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflict = build_conflict_bitmasks(edges, adj)
        
        # DSATUR greedy initialization
        color, nc = dsatur_greedy(conflict, m)
        
        # Local search
        color = local_search(color, conflict, m, max_iters=5000)
        
        # Polish
        color = polish(color, conflict, m)
        
        return color

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline - 40:
            break
        
        n, d, seed = spec
        out.append(solve_instance(n, d, seed))
    
    while len(out) < 3:
        out.append([])
    
    return out
# EVOLVE-BLOCK-END
