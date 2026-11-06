import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)

    def enumerate_pairings(neighbor_list):
        """Enumerate all matchings on a set of items (pairings of neighbors).
        Returns list of tuples, each tuple is a frozenset of paired edge-indices.
        """
        m = len(neighbor_list)
        if m == 0:
            return [frozenset()]
        
        results = []
        
        def backtrack(remaining, current):
            if not remaining:
                results.append(frozenset(current))
                return
            first = remaining[0]
            rest = remaining[1:]
            
            # Option 1: leave first unpaired
            backtrack(rest, current)
            
            # Option 2: pair first with each other element
            for i in range(len(rest)):
                pair = (first, rest[i])
                new_rest = rest[:i] + rest[i+1:]
                backtrack(new_rest, current | {pair})
        
        backtrack(list(range(m)), frozenset())
        return results

    def solve_one(edges, time_budget):
        if not edges:
            return []
        
        n = max(max(u, v) for u, v in edges) + 1
        
        # Build adjacency: for each vertex, list of (neighbor, edge_index)
        adj = defaultdict(list)
        edge_to_vertices = {}
        for idx, (u, v) in enumerate(edges):
            adj[u].append((v, idx))
            adj[v].append((u, idx))
            edge_to_vertices[idx] = (u, v)
        
        # For each vertex, enumerate all possible pairings of its incident edges
        vertex_pairings = {}
        for v in range(n):
            if v not in adj or not adj[v]:
                vertex_pairings[v] = [frozenset()]
                continue
            neighbor_edges = [eid for _, eid in adj[v]]
            # Enumerate all matchings on these edge indices
            pairings = enumerate_pairings(neighbor_edges)
            vertex_pairings[v] = pairings
        
        # Initialize: random pairing choice for each vertex
        choice = {}
        for v in range(n):
            if v in vertex_pairings and vertex_pairings[v]:
                choice[v] = random.choice(vertex_pairings[v])
            else:
                choice[v] = frozenset()
        
        def count_paths_for_choice(choice_map):
            """Given a choice of pairings at each vertex, count the number of paths.
            An edge is 'paired' if it appears in a pair at BOTH endpoints.
            Number of paths = (number of unpaired edge-ends) / 2 + (number of cycles)
            Actually: each path has 2 endpoints. Total unpaired ends / 2 = number of paths (if no cycles)
            But cycles have 0 endpoints. So we need to detect cycles.
            
            Better approach: build the chain structure and count connected components that are paths.
            """
            # Determine for each edge whether it's paired at each endpoint
            edge_paired_at = {}  # edge_idx -> (paired_at_u, paired_at_v)
            for eid, (u, v) in edge_to_vertices.items():
                paired_u = False
                paired_v = False
                for pair in choice_map.get(u, frozenset()):
                    if eid in pair:
                        paired_u = True
                        break
                for pair in choice_map.get(v, frozenset()):
                    if eid in pair:
                        paired_v = True
                        break
                edge_paired_at[eid] = (paired_u, paired_v)
            
            # An edge is a "chain link" if paired at both ends
            # Build the chain graph: vertices are edges, connect two edges if they're paired at a shared vertex
            # Then count connected components; each component is either a path or a cycle
            # Number of paths = number of components that are paths (not cycles)
            
            # Build adjacency between edges
            edge_adj = defaultdict(set)
            for v in range(n):
                if v not in adj:
                    continue
                # Group edges by which pair they belong to at vertex v
                pairs_at_v = choice_map.get(v, frozenset())
                for pair in pairs_at_v:
                    e1, e2 = pair
                    # Both e1 and e2 are incident to v
                    edge_adj[e1].add(e2)
                    edge_adj[e2].add(e1)
            
            # Find connected components
            visited = set()
            num_paths = 0
            for eid in range(len(edges)):
                if eid in visited:
                    continue
                # BFS
                component = []
                stack = [eid]
                visited.add(eid)
                while stack:
                    e = stack.pop()
                    component.append(e)
                    for neighbor_e in edge_adj.get(e, set()):
                        if neighbor_e not in visited:
                            visited.add(neighbor_e)
                            stack.append(neighbor_e)
                
                # Check if this component is a cycle or a path
                # A path has exactly 2 edges with degree 1 in the component (the endpoints)
                # A cycle has 0 such edges
                deg_in_comp = defaultdict(int)
                for e in component:
                    for neighbor_e in edge_adj.get(e, set()):
                        if neighbor_e in component:
                            deg_in_comp[e] += 1
                            deg_in_comp[neighbor_e] += 1
                
                endpoints = sum(1 for e in component if deg_in_comp[e] == 1)
                if endpoints == 2:
                    num_paths += 1
                elif endpoints == 0:
                    # It's a cycle - count as 1 path (a cycle can be viewed as a path with endpoints identified,
                    # but for our purposes, a cycle is a valid "path" decomposition element if we allow it.
                    # However, the problem says "simple paths", so a cycle is not a path.
                    # We need to break the cycle into a path by removing one edge... but that changes the decomposition.
                    # For now, count it as 1 (we'll handle cycle-breaking later)
                    num_paths += 1
                else:
                    # Shouldn't happen if pairings are valid
                    num_paths += 1
            
            return num_paths
        
        def is_valid_choice(choice_map):
            """Check that the choice is consistent: an edge is paired at u iff it's paired at v."""
            for eid, (u, v) in edge_to_vertices.items():
                paired_u = any(eid in pair for pair in choice_map.get(u, frozenset()))
                paired_v = any(eid in pair for pair in choice_map.get(v, frozenset()))
                if paired_u != paired_v:
                    return False
            return True
        
        # Local search: try to improve by changing pairing at a single vertex
        best_choice = dict(choice)
        best_score = count_paths_for_choice(best_choice)
        
        # First, fix consistency issues
        # Greedily fix: for each edge, if paired at one end but not the other, unpair at the paired end
        for _ in range(10):  # Iterate a few times
            changed = False
            for eid, (u, v) in edge_to_vertices.items():
                paired_u = any(eid in pair for pair in choice.get(u, frozenset()))
                paired_v = any(eid in pair for pair in choice.get(v, frozenset()))
                if paired_u and not paired_v:
                    # Remove eid from all pairs at u
                    new_pairs = frozenset(pair for pair in choice[u] if eid not in pair)
                    choice[u] = new_pairs
                    changed = True
                elif paired_v and not paired_u:
                    new_pairs = frozenset(pair for pair in choice[v] if eid not in pair)
                    choice[v] = new_pairs
                    changed = True
            if not changed:
                break
        
        # Now do local search
        vertices_with_options = [v for v in range(n) if len(vertex_pairings.get(v, [])) > 1]
        
        for iteration in range(5000):
            if time.time() > start_time + time_budget:
                break
            
            # Pick a random vertex to modify
            if not vertices_with_options:
                break
            v = random.choice(vertices_with_options)
            
            # Try a random different pairing
            current_idx = vertex_pairings[v].index(choice[v]) if choice[v] in vertex_pairings[v] else 0
            new_idx = (current_idx + random.randint(1, len(vertex_pairings[v]) - 1)) % len(vertex_pairings[v])
            old_choice = choice[v]
            choice[v] = vertex_pairings[v][new_idx]
            
            # Fix consistency
            for eid, (u, w) in edge_to_vertices.items():
                if u == v:
                    other = w
                elif w == v:
                    other = u
                else:
                    continue
                paired_v = any(eid in pair for pair in choice[v])
                paired_other = any(eid in pair for pair in choice.get(other, frozenset()))
                if paired_v and not paired_other:
                    choice[v] = frozenset(pair for pair in choice[v] if eid not in pair)
                    paired_v = False
                elif paired_other and not paired_v:
                    choice[other] = frozenset(pair for pair in choice[other] if eid not in pair)
            
            new_score = count_paths_for_choice(choice)
            
            if new_score < best_score:
                best_score = new_score
                best_choice = dict(choice)
            else:
                # Revert
                choice[v] = old_choice
                # Re-fix consistency (should be back to original state)
                for eid, (u, w) in edge_to_vertices.items():
                    if u == v:
                        other = w
                    elif w == v:
                        other = u
                    else:
                        continue
                    paired_v = any(eid in pair for pair in choice[v])
                    paired_other = any(eid in pair for pair in choice.get(other, frozenset()))
                    if paired_v and not paired_other:
                        choice[v] = frozenset(pair for pair in choice[v] if eid not in pair)
                    elif paired_other and not paired_v:
                        choice[other] = frozenset(pair for pair in choice[other] if eid not in pair)
        
        # Reconstruct paths from best_choice
        choice = best_choice
        
        # Build edge adjacency (which edges are chained together)
        edge_adj = defaultdict(set)
        for v in range(n):
            if v not in adj:
                continue
            for pair in choice.get(v, frozenset()):
                e1, e2 = pair
                edge_adj[e1].add(e2)
                edge_adj[e2].add(e1)
        
        # Find all maximal chains (paths in the edge-adjacency graph)
        visited_edges = set()
        paths = []
        
        for eid in range(len(edges)):
            if eid in visited_edges:
                continue
            
            # Start a chain from this edge
            # Find the "start" of the chain: an edge with degree <= 1 in edge_adj
            # Walk the chain
            chain_edges = []
            current = eid
            
            # Find an endpoint
            while edge_adj[current] and all(nei in visited_edges for nei in edge_adj[current]):
                # This edge only connects to visited edges, so it's an endpoint
                break
            # Actually, let's just do a proper walk
            # Find start: pick an edge in the component with degree 1
            component = []
            stack = [eid]
            comp_set = set()
            while stack:
                e = stack.pop()
                if e in comp_set:
                    continue
                comp_set.add(e)
                component.append(e)
                for nei in edge_adj.get(e, set()):
                    if nei not in comp_set:
                        stack.append(nei)
            
            for e in component:
                visited_edges.add(e)
            
            # Find endpoints in this component
            endpoints = []
            for e in component:
                deg = sum(1 for nei in edge_adj.get(e, set()) if nei in comp_set)
                if deg <= 1:
                    endpoints.append(e)
            
            if not endpoints:
                # It's a cycle - break it
                endpoints = [component[0]]
            
            # Walk from one endpoint
            if len(endpoints) >= 1:
                start_e = endpoints[0]
                chain = [start_e]
                prev = None
                current = start_e
                
                while True:
                    next_e = None
                    for nei in edge_adj.get(current, set()):
                        if nei in comp_set and nei != prev and nei not in chain:
                            next_e = nei
                            break
                    if next_e is None:
                        break
                    chain.append(next_e)
                    prev = current
                    current = next_e
                
                # Convert edge chain to vertex path
                vertex_path = []
                for i, e in enumerate(chain):
                    u, v = edge_to_vertices[e]
                    if i == 0:
                        # Determine direction based on next edge
                        if i + 1 < len(chain):
                            next_u, next_v = edge_to_vertices[chain[i+1]]
                            if next_u == v or next_v == v:
                                vertex_path.append(u)
                            else:
                                vertex_path.append(v)
                        else:
                            vertex_path.append(u)
                    else:
                        prev_u, prev_v = edge_to_vertices[chain[i-1]]
                        if prev_u == u or prev_v == u:
                            vertex_path.append(v)
                        else:
                            vertex_path.append(u)
                
                if len(vertex_path) >= 2:
                    paths.append(vertex_path)
        
        return paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for i, spec in enumerate(specs):
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            # Fallback: single edges
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        # Allocate time proportionally
        time_budget = remaining_time * 0.6  # Leave some buffer
        
        edges = instance(*spec)
        paths = solve_one(edges, time_budget)
        
        # Verify and fallback if needed
        if not paths:
            paths = [list(e) for e in edges]
        
        results.append(paths)
    
    return results
