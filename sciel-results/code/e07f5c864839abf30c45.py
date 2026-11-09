import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_eulerian_trails(adj, start_vertices):
        """
        Find all Eulerian trails in the graph.
        adj: dict of set of neighbors (undirected, but we track used edges)
        start_vertices: list of vertices where trails can start (odd-degree vertices)
        Returns list of trails (each is a list of vertices).
        """
        # We need to handle the graph which may have multiple components.
        # First, let's identify components.
        visited_nodes = set()
        components = []
        
        # Build a set of all nodes that have edges
        all_nodes = set()
        for u in adj:
            if adj[u]:
                all_nodes.add(u)
                for v in adj[u]:
                    all_nodes.add(v)
        
        for node in all_nodes:
            if node not in visited_nodes:
                # BFS to find component
                comp = set()
                queue = deque([node])
                visited_nodes.add(node)
                while queue:
                    u = queue.popleft()
                    comp.add(u)
                    for v in adj[u]:
                        if v not in visited_nodes:
                            visited_nodes.add(v)
                            queue.append(v)
                components.append(comp)
        
        trails = []
        
        # For each component, find Eulerian trails
        # We need to be careful: the graph is undirected.
        # Hierholzer's algorithm works for Eulerian circuits.
        # For Eulerian trails (in a component with exactly 2 odd vertices),
        # we can add a virtual edge between the two odd vertices to make it Eulerian,
        # find the circuit, and remove the virtual edge.
        # Or, we can just use Hierholzer's starting from one of the odd vertices.
        # Actually, Hierholzer's works for trails too, as long as we start at the correct vertex.
        # If the component is Eulerian (0 odd vertices), start anywhere.
        # If the component has 2 odd vertices, start at one of them.
        
        for comp in components:
            if not any(adj[u] for u in comp):
                continue
            
            # Find odd vertices in this component
            odd_in_comp = [u for u in comp if len(adj[u]) % 2 != 0]
            
            if len(odd_in_comp) == 0:
                # Eulerian circuit
                start = list(comp)[0]
                trail = find_eulerian_circuit_in_comp(adj, comp, start)
                trails.append(trail)
            elif len(odd_in_comp) == 2:
                # Eulerian trail
                start = odd_in_comp[0]
                trail = find_eulerian_circuit_in_comp(adj, comp, start)
                trails.append(trail)
            else:
                # This shouldn't happen if we pre-processed correctly
                # But if it does, we might have a problem.
                # For safety, just return empty or handle somehow.
                pass
        
        return trails

    def find_eulerian_circuit_in_comp(adj, comp, start):
        """
        Hierholzer's algorithm restricted to a component.
        """
        # We need to track used edges.
        # Since adj is a dict of sets, and we can't easily modify it without affecting other calls,
        # let's make a copy or use a set of used edges.
        # For efficiency, let's use a set of used edges (frozenset of (u,v)).
        
        used_edges = set()
        circuit = []
        stack = [start]
        
        while stack:
            u = stack[-1]
            # Find an unused edge from u to a vertex in comp
            next_v = None
            for v in adj[u]:
                if v not in comp:
                    continue
                e = frozenset((u, v))
                if e not in used_edges:
                    next_v = v
                    break
            
            if next_v is None:
                circuit.append(stack.pop())
            else:
                e = frozenset((u, next_v))
                used_edges.add(e)
                stack.append(next_v)
        
        circuit.reverse()
        
        # If it's a circuit, the last vertex should be the same as the first.
        # If it's a trail, the last vertex is different.
        # The problem asks for paths. A circuit v0...vk v0 can be represented as path v0...vk.
        # A trail v0...vk is already a path.
        # So we just return the circuit as is.
        # But wait, if it's a circuit, the last element is the same as the first.
        # The verifier likely expects simple paths (no repeated vertices).
        # A circuit v0...vk v0 has v0 repeated.
        # We should remove the last vertex if it's the same as the first.
        
        if len(circuit) > 1 and circuit[0] == circuit[-1]:
            circuit.pop()
        
        return circuit

    def solve_graph(edges, seed):
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Calculate degrees
        degrees = {v: len(adj[v]) for v in adj}
        odd_vertices = [v for v in degrees if degrees[v] % 2 != 0]
        
        if not odd_vertices:
            # Eulerian
            trails = find_eulerian_trails(adj, [])
            return trails
        
        # Odd vertices exist
        # We need to pair up odd vertices and add paths between them.
        # To minimize the number of paths, we want to add as few paths as possible.
        # Each path added connects two odd vertices and reduces the number of odd vertices by 2.
        # We need (len(odd_vertices) // 2) paths.
        # To minimize the total number of paths, we want these added paths to be as "short" as possible?
        # No, the total number of paths is the number of Eulerian trails in the final graph.
        # The final graph will have 0 odd vertices (if we add enough paths).
        # Wait, if we add a path between two odd vertices, their degrees become even.
        # The internal vertices of the path have their degrees increased by 2 (even), so they remain even/odd as before.
        # So if we add k paths, we reduce the number of odd vertices by 2k.
        # We need to reduce it to 0. So k = len(odd_vertices) // 2.
        # The final graph will be Eulerian, so it will have exactly 1 Eulerian circuit per component.
        # Wait, if the graph is connected, it will have 1 circuit.
        # If the graph is not connected, it will have 1 circuit per component.
        # But the original graph is connected (by instance_graph construction).
        # So the final graph will be connected and Eulerian.
        # So it will have exactly 1 Eulerian circuit.
        # This means the number of paths is 1!
        # This is clearly wrong. The number of paths in the decomposition is the number of trails we output.
        # If the graph is Eulerian, we output 1 trail.
        # If the graph has 2k odd vertices, we add k paths, making it Eulerian, and then output 1 trail.
        # This would mean the number of paths is always 1, which is not true.
        
        # Let's re-read the problem.
        # "Paths must be simple, use graph edges, be edge-disjoint and cover all edges."
        # "Minimize the total number of paths"
        # Gallai's conjecture: at most (n-1)/2 paths.
        # For n=20, at most 9.5 -> 10? No, (20-1)/2 = 9.5, so 10.
        # Wait, Gallai's conjecture is about decomposing a graph into paths.
        # The number of paths is not necessarily 1.
        # My confusion: if I add paths to make it Eulerian, I am *changing* the graph.
        # The decomposition must use the *original* edges.
        # The "adding paths" is a conceptual step to find the decomposition.
        # The actual paths in the decomposition are the original edges partitioned into paths.
        
        # Correct approach:
        # 1. Pair up odd vertices.
        # 2. Find paths in the original graph connecting each pair.
        # 3. These paths are part of the decomposition.
        # 4. The remaining edges (original edges minus the edges used in these paths) form an Eulerian graph.
        # 5. Decompose the remaining edges into Eulerian circuits/trails.
        # 6. The total decomposition is the union of the paths from step 2 and the trails from step 5.
        
        # So the number of paths is (number of pairs) + (number of Eulerian components in the remaining graph).
        # To minimize the total, we want to minimize the number of pairs (which is fixed at k) and the number of components.
        # Since the original graph is connected, and we remove paths, the remaining graph might become disconnected.
        # To keep it connected, we should choose the paths carefully.
        # However, for a first pass, let's just find *any* paths connecting the pairs.
        # We can use shortest paths (BFS) to connect the pairs.
        
        # Pair up odd vertices
        # To minimize the number of components in the remaining graph, we should try to keep the graph connected.
        # A simple heuristic: pair up odd vertices that are close to each other.
        # Or, pair them arbitrarily.
        
        # Let's try arbitrary pairing first.
        rng = random.Random(seed)
        rng.shuffle(odd_vertices)
        
        pairs = []
        for i in range(0, len(odd_vertices), 2):
            pairs.append((odd_vertices[i], odd_vertices[i+1]))
        
        # Find paths for each pair
        used_edges = set()
        paths = []
        
        for u, v in pairs:
            # Find shortest path from u to v in the graph, avoiding used_edges
            # BFS
            parent = {u: None}
            queue = deque([u])
            found = False
            while queue and not found:
                curr = queue.popleft()
                if curr == v:
                    found = True
                    break
                for neighbor in adj[curr]:
                    e = frozenset((curr, neighbor))
                    if e in used_edges:
                        continue
                    if neighbor not in parent:
                        parent[neighbor] = curr
                        queue.append(neighbor)
            
            if not found:
                # Should not happen if graph is connected and we haven't used all edges
                # But just in case
                continue
            
            # Reconstruct path
            path = [v]
            while path[-1] != u:
                path.append(parent[path[-1]])
            path.reverse()
            
            # Mark edges as used
            for i in range(len(path) - 1):
                e = frozenset((path[i], path[i+1]))
                used_edges.add(e)
            
            paths.append(path)
        
        # Now, decompose the remaining edges
        remaining_edges = edge_set - {tuple(sorted(e)) for e in used_edges}
        
        if not remaining_edges:
            return paths
        
        rem_adj = defaultdict(set)
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Find Eulerian trails in the remaining graph
        rem_trails = find_eulerian_trails(rem_adj, [])
        paths.extend(rem_trails)
        
        return paths

    results = []
    for i, spec in enumerate([(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]):
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        edges = instance(*spec)
        paths = solve_graph(edges, spec[2])
        results.append(paths)
        
    return results

