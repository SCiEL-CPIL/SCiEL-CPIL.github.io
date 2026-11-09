"""Strong edge colouring of three fixed bounded-degree graphs."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill

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

    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def build_conflict_graph(edges, adj, n):
        """Build conflict graph where nodes are edges, edges connect conflicting pairs."""
        m = len(edges)
        # For each edge i, find all edges j that conflict with it
        # Conflict: share a vertex OR are joined by an edge in the original graph
        edge_idx = {e: i for i, e in enumerate(edges)}
        conflict = [set() for _ in range(m)]
        
        for i, (u, v) in enumerate(edges):
            # Vertices in the closed neighborhood of this edge
            near = {u, v} | adj[u] | adj[v]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u2 in near or v2 in near:
                    conflict[i].add(j)
                    conflict[j].add(i)
        return conflict

    def greedy_max_independent_set(conflict, candidates):
        """Greedy heuristic to find a large independent set from candidates."""
        if not candidates:
            return set()
        
        # Sort by degree in conflict graph (among candidates) ascending
        cand_set = set(candidates)
        sorted_cands = sorted(cand_set, key=lambda x: len(conflict[x] & cand_set))
        
        result = set()
        remaining = set(cand_set)
        
        for v in sorted_cands:
            if v not in remaining:
                continue
            result.add(v)
            remaining.discard(v)
            # Remove all neighbors of v from remaining
            for w in conflict[v]:
                remaining.discard(w)
        
        return result

    def branch_and_bound_max_is(conflict, candidates, best_size=0, deadline=None):
        """Branch and bound to find maximum independent set."""
        if deadline is not None and time.time() >= deadline:
            return set()
        
        if not candidates:
            return set()
        
        # If all candidates are mutually non-adjacent, take them all
        cand_set = set(candidates)
        if len(cand_set) <= 1:
            return cand_set.copy()
        
        # Quick upper bound: if |candidates| <= best_size, prune
        if len(cand_set) <= best_size:
            return set()
        
        # Try greedy first to get a good initial solution
        greedy_sol = greedy_max_independent_set(conflict, cand_set)
        if len(greedy_sol) > best_size:
            best_size = len(greedy_sol)
        
        # If greedy found all, return it
        if len(greedy_sol) == len(cand_set):
            return greedy_sol
        
        # Pick vertex with max degree in the induced subgraph on candidates
        # This gives better pruning
        v = max(cand_set, key=lambda x: len(conflict[x] & cand_set))
        
        # Branch 1: include v
        # Candidates for this branch: candidates minus v and its neighbors
        include_cands = cand_set - {v} - (conflict[v] & cand_set)
        sol_include = branch_and_bound_max_is(conflict, include_cands, best_size, deadline)
        if len(sol_include) + 1 > best_size:
            best_size = len(sol_include) + 1
            sol_include.add(v)
            # If we've found a perfect solution, return immediately
            if best_size == len(cand_set):
                return sol_include
        else:
            sol_include = set()
        
        # Branch 2: exclude v
        exclude_cands = cand_set - {v}
        sol_exclude = branch_and_bound_max_is(conflict, exclude_cands, best_size, deadline)
        
        if len(sol_exclude) > len(sol_include):
            return sol_exclude
        return sol_include

    def extract_color_classes(edges, adj, conflict, max_colors, deadline):
        """
        Extract color classes using maximum independent set approach.
        Returns a list of color classes (each is a set of edge indices).
        """
        remaining = set(range(len(edges)))
        color_classes = []
        
        while remaining and len(color_classes) < max_colors:
            if time.time() >= deadline:
                break
            
            # Find maximum independent set from remaining
            # Use B&B with a reasonable time limit per extraction
            time_limit = min(0.5, max(0.1, (deadline - time.time()) / (max_colors - len(color_classes) + 1)))
            
            # Try B&B first
            is_set = branch_and_bound_max_is(conflict, remaining, 0, time.time() + time_limit)
            
            # If B&B returned empty or very small, fall back to greedy
            if not is_set or len(is_set) < 2:
                is_set = greedy_max_independent_set(conflict, remaining)
            
            if not is_set:
                # Shouldn't happen, but safety
                break
            
            color_classes.append(is_set)
            remaining -= is_set
        
        # Assign remaining edges to new color classes using greedy
        while remaining:
            if time.time() >= deadline:
                break
            is_set = greedy_max_independent_set(conflict, remaining)
            if not is_set:
                break
            color_classes.append(is_set)
            remaining -= is_set
        
        return color_classes

    def assign_colors(color_classes, m):
        """Convert color classes to a color array."""
        colors = [0] * m
        for c, cls in enumerate(color_classes):
            for e in cls:
                colors[e] = c
        return colors

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        adj = build_adj(edges, n)
        m = len(edges)
        
        if m == 0:
            return []
        
        # Build conflict graph
        conflict = build_conflict_graph(edges, adj, n)
        
        # Upper bound on number of colors needed
        # Each color class is an induced matching, so each edge in a class
        # uses 2 vertices, and no two edges in a class can be adjacent
        # Maximum size of an induced matching in a graph with max degree d
        # is at most n/2, but more tightly bounded
        # Simple upper bound: m (each edge gets its own color)
        max_colors = m
        
        # Better upper bound: each color class can contain at most floor(m / (2d)) edges?
        # Actually, let's use a heuristic upper bound based on the structure
        # Each vertex has degree at most d, so in a color class, each vertex
        # can be incident to at most 1 edge. So max class size <= n/2
        # But also, edges in a class can't be joined by an edge, which further restricts
        # Let's just use m as upper bound and let the algorithm find the actual number
        
        # Time allocation
        instance_deadline = min(deadline, time.time() + 40)
        
        # Try to extract color classes
        color_classes = extract_color_classes(edges, adj, conflict, max_colors, instance_deadline)
        
        if not color_classes:
            # Fallback: greedy coloring
            colors = [0] * m
            for i in range(m):
                u, v = edges[i]
                near = {u, v} | adj[u] | adj[v]
                used = set()
                for j in range(i):
                    u2, v2 = edges[j]
                    if u2 in near or v2 in near:
                        used.add(colors[j])
                c = 0
                while c in used:
                    c += 1
                colors[i] = c
            return colors
        
        colors = assign_colors(color_classes, m)
        return colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() >= deadline:
            # Emergency fallback
            n, d, seed = spec
            edges = instance(n, d, seed)
            adj = build_adj(edges, n)
            m = len(edges)
            colors = [0] * m
            for i in range(m):
                u, v = edges[i]
                near = {u, v} | adj[u] | adj[v]
                used = set()
                for j in range(i):
                    u2, v2 = edges[j]
                    if u2 in near or v2 in near:
                        used.add(colors[j])
                c = 0
                while c in used:
                    c += 1
                colors[i] = c
            out.append(colors)
            continue
        coloring = solve_instance(*spec)
        out.append(coloring)
    
    return out

# EVOLVE-BLOCK-END
