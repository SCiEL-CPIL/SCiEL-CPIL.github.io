"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def build_conflict(edges, adj):
        """Build conflict graph: two edges conflict if they share a vertex or are connected by an edge."""
        m = len(edges)
        edge_idx = {e: i for i, e in enumerate(edges)}
        conflict = [set() for _ in range(m)]
        
        for i, (u, v) in enumerate(edges):
            # Edges sharing a vertex
            for w in adj.get(u, ()):
                for e2 in [(u, w), (w, u)]:
                    e2 = (min(e2), max(e2))
                    if e2 in edge_idx and e2 != (u, v):
                        j = edge_idx[e2]
                        conflict[i].add(j)
                        conflict[j].add(i)
            for w in adj.get(v, ()):
                for e2 in [(v, w), (w, v)]:
                    e2 = (min(e2), max(e2))
                    if e2 in edge_idx and e2 != (u, v):
                        j = edge_idx[e2]
                        conflict[i].add(j)
                        conflict[j].add(i)
            # Edges connected by an edge (u-x, u-y, v-x, v-y where x-y is an edge)
            for x in adj.get(u, ()):
                for y in adj.get(x, ()):
                    if y == u:
                        continue
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_idx and e2 != (u, v):
                        j = edge_idx[e2]
                        conflict[i].add(j)
                        conflict[j].add(i)
            for x in adj.get(v, ()):
                for y in adj.get(x, ()):
                    if y == v:
                        continue
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_idx and e2 != (u, v):
                        j = edge_idx[e2]
                        conflict[i].add(j)
                        conflict[j].add(i)
        
        return conflict
    
    def dsatur_greedy(conflict, m):
        """DSATUR greedy colouring."""
        colour = [-1] * m
        colour_count = 0
        # Track saturation degree and degree for tie-breaking
        sat_deg = [0] * m  # number of different colours in neighbourhood
        deg = [len(conflict[i]) for i in range(m)]
        
        for step in range(m):
            # Find uncoloured vertex with max saturation, then max degree
            best = -1
            best_sat = -1
            best_deg = -1
            for i in range(m):
                if colour[i] == -1:
                    # Compute saturation degree
                    neighbours = conflict[i]
                    colours_in_neighbours = set()
                    for j in neighbours:
                        if colour[j] != -1:
                            colours_in_neighbours.add(colour[j])
                    s = len(colours_in_neighbours)
                    if s > best_sat or (s == best_sat and deg[i] > best_deg):
                        best_sat = s
                        best_deg = deg[i]
                        best = i
            if best == -1:
                break
            # Assign smallest available colour
            used = set()
            for j in conflict[best]:
                if colour[j] != -1:
                    used.add(colour[j])
            c = 0
            while c in used:
                c += 1
            colour[best] = c
            if c >= colour_count:
                colour_count = c + 1
        
        return colour, colour_count
    
    def local_optimise(conflict, m, colour, colour_count):
        """Try to reduce colour count by reassigning edges."""
        # Try to eliminate the highest colour by reassigning its edges
        for attempt in range(100):
            if colour_count <= 1:
                break
            # Find all edges with the highest colour
            max_c = colour_count - 1
            to_reassign = [i for i in range(m) if colour[i] == max_c]
            if not to_reassign:
                colour_count -= 1
                continue
            
            # Try to reassign each edge to a lower colour
            success = True
            new_colour = list(colour)
            for i in to_reassign:
                used = set()
                for j in conflict[i]:
                    if new_colour[j] != -1:
                        used.add(new_colour[j])
                # Try colours 0 to max_c-1
                c = 0
                found = False
                while c < max_c:
                    if c not in used:
                        new_colour[i] = c
                        found = True
                        break
                    c += 1
                if not found:
                    success = False
                    break
            if success:
                colour = new_colour
                colour_count -= 1
            else:
                break
        
        return colour, colour_count
    
    def greedy_with_order(conflict, m, order):
        """Greedy colouring with a given order."""
        colour = [-1] * m
        colour_count = 0
        for i in order:
            used = set()
            for j in conflict[i]:
                if colour[j] != -1:
                    used.add(colour[j])
            c = 0
            while c in used:
                c += 1
            colour[i] = c
            if c >= colour_count:
                colour_count = c + 1
        return colour, colour_count
    
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline:
            # Fallback: return empty (will be invalid, but avoid timeout)
            out.append([])
            continue
        
        edges = instance(*spec)
        m = len(edges)
        if m == 0:
            out.append([])
            continue
        
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        conflict = build_conflict(edges, adj)
        
        # DSATUR greedy
        colour, cc = dsatur_greedy(conflict, m)
        
        # Local optimisation
        colour, cc = local_optimise(conflict, m, colour, cc)
        
        # Try multiple random orderings
        rng = Random(42)
        best_colour = colour
        best_cc = cc
        
        for trial in range(50):
            if time.time() > deadline:
                break
            order = list(range(m))
            rng.shuffle(order)
            # Sort by degree descending as a heuristic
            order.sort(key=lambda i: -len(conflict[i]))
            
            c2, cc2 = greedy_with_order(conflict, m, order)
            c2, cc2 = local_optimise(conflict, m, c2, cc2)
            
            if cc2 < best_cc:
                best_cc = cc2
                best_colour = c2
        
        out.append([best_colour[i] for i in range(m)])
    
    return out
# EVOLVE-BLOCK-END
