import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def find_clique_from_seed(seed_vertex, adj, remaining_edges, n, time_limit):
        """Grow a clique starting from seed_vertex using greedy + local search."""
        if time.time() > time_limit:
            return [seed_vertex]
        
        # Start with seed vertex
        clique = [seed_vertex]
        clique_set = {seed_vertex}
        
        # Candidate vertices: neighbors of all clique members that have edges to all clique members
        candidates = set(adj.get(seed_vertex, set()))
        
        while True:
            if time.time() > time_limit:
                break
            
            # Find best candidate to add: one that connects to all current clique members
            best_candidate = None
            best_score = -1
            
            for v in candidates:
                # Check if v connects to all clique members via remaining_edges
                valid = True
                for u in clique:
                    if u < v:
                        if (u, v) not in remaining_edges:
                            valid = False
                            break
                    else:
                        if (v, u) not in remaining_edges:
                            valid = False
                            break
                if not valid:
                    continue
                
                # Score: degree within remaining graph
                deg = 0
                for w in adj.get(v, set()):
                    if w < v:
                        if (w, v) in remaining_edges:
                            deg += 1
                    else:
                        if (v, w) in remaining_edges:
                            deg += 1
                if deg > best_score:
                    best_score = deg
                    best_candidate = v
            
            if best_candidate is None:
                break
            
            clique.append(best_candidate)
            clique_set.add(best_candidate)
            
            # Update candidates: must connect to all clique members
            new_candidates = set()
            for v in candidates:
                valid = True
                for u in clique:
                    if u < v:
                        if (u, v) not in remaining_edges:
                            valid = False
                            break
                    else:
                        if (v, u) not in remaining_edges:
                            valid = False
                            break
                if valid:
                    new_candidates.add(v)
            candidates = new_candidates
        
        return clique

    def find_max_clique_multi_start(active_vertices, adj, remaining_edges, n, time_limit):
        """Try multiple starting strategies to find a large clique."""
        best_clique = []
        
        # Strategy 1: Start from max-degree vertex
        max_deg_vertex = None
        max_deg = -1
        for v in active_vertices:
            deg = 0
            for w in adj.get(v, set()):
                if w < v:
                    if (w, v) in remaining_edges:
                        deg += 1
                else:
                    if (v, w) in remaining_edges:
                        deg += 1
            if deg > max_deg:
                max_deg = deg
                max_deg_vertex = v
        
        if max_deg_vertex is not None:
            clique = find_clique_from_seed(max_deg_vertex, adj, remaining_edges, n, time_limit)
            if len(clique) > len(best_clique):
                best_clique = clique
        
        if time.time() > time_limit:
            return best_clique
        
        # Strategy 2: Start from second-highest degree vertex
        degrees = {}
        for v in active_vertices:
            deg = 0
            for w in adj.get(v, set()):
                if w < v:
                    if (w, v) in remaining_edges:
                        deg += 1
                else:
                    if (v, w) in remaining_edges:
                        deg += 1
            degrees[v] = deg
        
        sorted_vertices = sorted(active_vertices, key=lambda v: degrees.get(v, 0), reverse=True)
        
        # Try top few high-degree vertices
        for v in sorted_vertices[:3]:
            if time.time() > time_limit:
                break
            clique = find_clique_from_seed(v, adj, remaining_edges, n, time_limit)
            if len(clique) > len(best_clique):
                best_clique = clique
        
        # Strategy 3: Randomized starts
        if time.time() < time_limit - 0.5:
            rng = random.Random(42)
            for _ in range(5):
                if time.time() > time_limit:
                    break
                v = rng.choice(list(active_vertices))
                clique = find_clique_from_seed(v, adj, remaining_edges, n, time_limit)
                if len(clique) > len(best_clique):
                    best_clique = clique
        
        return best_clique

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency for the full graph
        full_adj = {u: set() for u in range(n)}
        for u, v in edges:
            full_adj[u].add(v)
            full_adj[v].add(u)
        
        remaining_edges = set(edges)
        partition = []
        
        while remaining_edges and time.time() < deadline:
            # Get active vertices
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
            
            if not active_vertices:
                break
            
            # Build subgraph adjacency from remaining edges
            sub_adj = {u: set() for u in active_vertices}
            for u, v in remaining_edges:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
            
            # Find a large clique using multi-start approach
            time_limit = min(deadline - 1, time.time() + 3)
            clique = find_max_clique_multi_start(active_vertices, sub_adj, remaining_edges, n, time_limit)
            
            if not clique or len(clique) < 2:
                # Fallback: pick an edge
                u, v = next(iter(remaining_edges))
                clique = [u, v]
            
            # Verify it's actually a clique in remaining_edges
            valid = True
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    if a < b:
                        if (a, b) not in remaining_edges:
                            valid = False
                            break
                    else:
                        if (b, a) not in remaining_edges:
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                # Fallback to edge
                u, v = next(iter(remaining_edges))
                clique = [u, v]
            
            partition.append(clique)
            
            # Remove all edges within this clique
            to_remove = set()
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    if a < b:
                        to_remove.add((a, b))
                    else:
                        to_remove.add((b, a))
            remaining_edges -= to_remove
        
        # If we ran out of time, add remaining edges as single edges
        if remaining_edges and time.time() >= deadline:
            for u, v in remaining_edges:
                partition.append([u, v])
        
        return partition

    result1 = solve_instance(32, 0.6, 601, 60)
    result2 = solve_instance(64, 0.55, 602, 65)
    
    return [result1, result2]
