import time
import sys
import random
import numpy as np
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_shortest_path(start, end, remaining_edges, adj, max_len=None):
        """BFS to find shortest path using only remaining edges."""
        if start == end:
            return [start]
        
        prev = {start: None}
        q = deque([start])
        found = False
        
        while q and not found:
            u = q.popleft()
            for v in adj[u]:
                e = tuple(sorted((u, v)))
                if e in remaining_edges and v not in prev:
                    prev[v] = u
                    if v == end:
                        found = True
                        break
                    q.append(v)
        
        if not found:
            return None
        
        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        
        if max_len and len(path) > max_len:
            return None
            
        return path

    def repair_trail(trail, edges, adj):
        """
        Takes a trail (walk) which may repeat vertices.
        Splits it into simple paths that are edge-disjoint and cover the trail's edges.
        Returns a list of simple paths.
        """
        if not trail:
            return []
        
        # Map each edge in the trail to its position
        # We need to decompose the sequence of vertices into simple paths.
        # A simple path cannot repeat vertices.
        # We can greedily extract simple paths from the trail.
        
        remaining_trail_edges = set()
        # Reconstruct edges from trail
        for i in range(len(trail) - 1):
            u, v = trail[i], trail[i+1]
            remaining_trail_edges.add(tuple(sorted((u, v))))
            
        # We will greedily pick edges from the trail to form simple paths.
        # Since the trail is a sequence, we can just walk it.
        # However, if we just take the whole trail, it might not be simple.
        # We need to split it.
        
        # Strategy: 
        # 1. Identify the set of edges in the trail.
        # 2. While edges remain:
        #    a. Find a vertex in the remaining edges.
        #    b. Perform a DFS/BFS on the subgraph induced by remaining edges 
        #       to find a maximal simple path.
        #    c. Remove those edges.
        
        # This is essentially decomposing a multigraph (which is a trail, so max degree 2 at non-endpoints, 
        # but wait, a trail can visit a vertex multiple times, so the degree in the "used edges" sense 
        # can be > 2? No, a trail is a walk where edges are distinct. 
        # The subgraph formed by the edges of a trail is a collection of paths and cycles? 
        # No, a trail is a single connected component if it doesn't lift the pen? 
        # Actually, a trail is a walk with no repeated edges. The graph formed by its edges 
        # has max degree 2? No. 
        # Example: 1-2-3-2-4. Edges: (1,2), (2,3), (3,2), (2,4). 
        # Vertex 2 has degree 3 in this edge set? 
        # (1,2) and (2,3) and (2,4). Yes. 
        # So the subgraph is not necessarily a collection of disjoint paths/cycles. 
        # It's a graph where we need to find a path cover.
        
        # But wait, the edges in the trail form a connected subgraph (the trail itself).
        # We can just do a greedy path extraction on this subgraph.
        
        subgraph_edges = remaining_trail_edges
        paths = []
        
        while subgraph_edges:
            # Pick a start vertex. Prefer vertices with degree 1 in the subgraph if any.
            sub_adj = defaultdict(set)
            for u, v in subgraph_edges:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
                
            start = None
            # Find a vertex with degree 1 (leaf) to start a path
            for v in sub_adj:
                if len(sub_adj[v]) == 1:
                    start = v
                    break
            
            if start is None:
                # All degrees >= 2, pick any
                start = next(iter(sub_adj))
                
            # Greedy walk to find a simple path
            path = [start]
            current = start
            used_in_path = {start}
            
            while True:
                next_v = None
                # Find a neighbor in subgraph_edges not in used_in_path
                for nb in sub_adj[current]:
                    e = tuple(sorted((current, nb)))
                    if e in subgraph_edges and nb not in used_in_path:
                        next_v = nb
                        break
                        
                if next_v is None:
                    break
                    
                e = tuple(sorted((current, next_v)))
                subgraph_edges.discard(e)
                path.append(next_v)
                used_in_path.add(next_v)
                current = next_v
                
            if len(path) > 1:
                paths.append(path)
            else:
                # Should not happen if subgraph_edges not empty and start has degree > 0
                # But if start had degree 0? No, we picked from sub_adj which has edges.
                # If we got stuck immediately, it means no valid move.
                # This implies the subgraph might have been disconnected or we made a bad choice?
                # Since it's a trail, it's connected. 
                # If we can't move, it means all neighbors are in used_in_path.
                # This can happen if the "simple path" constraint forces a stop.
                # We just continue the loop.
                pass
                
        return paths

    def decompose_graph(edges):
        if not edges:
            return []

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # 1. Calculate degrees
        degrees = {v: 0 for v in adj}
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
            
        odd_vertices = [v for v, d in degrees.items() if d % 2 == 1]
        
        # 2. If no odd vertices, we have an Eulerian circuit (one trail).
        #    If odd vertices, we need to pair them.
        
        # We will use a randomized approach to find a good pairing.
        # The number of trails is ceil(|odd| / 2).
        # We want to find paths between paired odd vertices that are "short" or "simple" 
        # to minimize the need for splitting later?
        # Actually, the total number of edges is fixed. 
        # The number of trails is fixed by the pairing count.
        # The issue is that the trails might not be simple.
        
        # Strategy:
        # 1. Generate a pairing of odd vertices.
        # 2. Find shortest paths between each pair in the original graph.
        # 3. Combine these paths? No, that doesn't cover all edges.
        
        # Correct Strategy for Minimum Path Cover (Trails):
        # 1. Add "dummy" edges to pair up odd vertices. 
        #    This makes the graph Eulerian.
        # 2. Find an Eulerian circuit in the augmented graph.
        # 3. The Eulerian circuit corresponds to a set of trails in the original graph,
        #    where the "dummy" edges indicate where one trail ends and another begins.
        # 4. These trails cover all edges.
        # 5. Repair each trail to be a simple path.
        
        # Implementation:
        # We need to choose the pairing of odd vertices.
        # We want to minimize the "cost" of the pairing, where cost is related to 
        # the difficulty of repairing the resulting trails.
        # A proxy for difficulty is the length of the path between the pair in the original graph?
        # Or simply, we can try multiple random pairings.
        
        # Let's try a few random pairings.
        best_paths = None
        best_score = float('inf')
        
        num_restarts = 10
        if len(odd_vertices) < 4:
            num_restarts = 50 # Fewer odd vertices, more room to search
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Randomly shuffle odd vertices and pair them
            odds = odd_vertices[:]
            random.shuffle(odds)
            pairs = [(odds[i], odds[i+1]) for i in range(0, len(odds), 2)]
            
            # Build augmented graph
            # We need to find an Eulerian circuit in G + dummy_edges.
            # Instead of explicitly building the augmented graph and finding Eulerian circuit,
            # we can simulate the trail decomposition.
            
            # Actually, finding the Eulerian circuit in the augmented graph gives us the trails.
            # Let's build the augmented adjacency.
            aug_adj = defaultdict(list) # list to allow multi-edges (dummy ones)
            # Copy original edges
            for u, v in edges:
                aug_adj[u].append(v)
                aug_adj[v].append(u)
                
            # Add dummy edges
            for u, v in pairs:
                aug_adj[u].append(v)
                aug_adj[v].append(u)
                
            # Find Eulerian circuit
            # Hierholzer's algorithm
            circuit = []
            stack = [odd_vertices[0] if odd_vertices else next(iter(adj))]
            # Start from a vertex with non-zero degree in augmented graph
            start_v = stack[0]
            
            # We need to track used edges. 
            # Since we have dummy edges, we need to distinguish them.
            # Let's use a set of used edges. 
            # Original edges are (u, v) where u < v.
            # Dummy edges are also (u, v). 
            # This is a problem if a dummy edge coincides with a real edge.
            # To handle this, we can label edges.
            
            # Let's re-implement Hierholzer with edge IDs.
            # Or simpler: 
            # Since n is small, we can just do a DFS.
            
            # Let's use a multiset of edges.
            # For simplicity, let's just find the trails by walking.
            
            # Alternative approach:
            # The trails are the segments of the Eulerian circuit that don't use dummy edges.
            # We can find the Eulerian circuit of the augmented graph.
            
            # Let's build a list of edges with IDs.
            # Real edges: ID = index
            # Dummy edges: ID = index + len(real_edges)
            
            real_edges_list = list(edges)
            num_real = len(real_edges_list)
            
            # Map edge tuple to list of IDs
            edge_to_ids = defaultdict(list)
            for i, e in enumerate(real_edges_list):
                edge_to_ids[e].append(i)
                
            dummy_edges_list = pairs
            for i, e in enumerate(dummy_edges_list):
                edge_to_ids[e].append(num_real + i)
                
            # Build adjacency with edge IDs
            aug_adj_ids = defaultdict(list)
            for e, ids in edge_to_ids.items():
                u, v = e
                for eid in ids:
                    aug_adj_ids[u].append((v, eid))
                    aug_adj_ids[v].append((u, eid))
                    
            used_ids = set()
            circuit_vertices = []
            
            # Hierholzer
            stack_v = [start_v]
            stack_e = [] # not needed if we pop from adj
            
            # Standard Hierholzer implementation
            def hierholzer(start):
                stack = [start]
                circuit = []
                while stack:
                    v = stack[-1]
                    if aug_adj_ids[v]:
                        # Pick an unused edge
                        # We need to remove it from aug_adj_ids
                        # Find an unused edge
                        while aug_adj_ids[v]:
                            nv, eid = aug_adj_ids[v].pop()
                            if eid not in used_ids:
                                used_ids.add(eid)
                                # Remove the reverse edge too? 
                                # Since we are modifying the list, we just pop.
                                # But we need to remove the corresponding edge from the other side.
                                # This is tricky with lists.
                                # Let's use a different representation.
                                break
                        if not aug_adj_ids[v]:
                            break
                        # We popped (nv, eid)
                        # We need to remove (v, eid) from aug_adj_ids[nv]
                        # This is O(degree). Acceptable for small n.
                        # Find and remove
                        for i, (u, e_id) in enumerate(aug_adj_ids[nv]):
                            if e_id == eid:
                                aug_adj_ids[nv].pop(i)
                                break
                        stack.append(nv)
                    else:
                        circuit.append(stack.pop())
                return circuit
            
            # The above Hierholzer is a bit messy with the removal.
            # Let's use a simpler "greedy trail" approach on the augmented graph?
            # No, we need the exact Eulerian circuit to define the trails.
            
            # Let's stick to a simpler method:
            # We know the number of trails is K.
            # We can just find K edge-disjoint paths that cover all edges?
            # That's the original problem!
            
            # Let's go back to the "Add dummy edges -> Eulerian Circuit -> Split" method.
            # It is theoretically sound.
            
            # To make Hierholzer easier, let's use a dict of sets for remaining edges.
            rem_edges = defaultdict(set)
            for e, ids in edge_to_ids.items():
                for eid in ids:
                    u, v = e
                    rem_edges[u].add((v, eid))
                    rem_edges[v].add((u, eid))
                    
            circuit = []
            stack = [start_v]
            
            while stack:
                v = stack[-1]
                if rem_edges[v]:
                    # Pick one
                    nv, eid = next(iter(rem_edges[v]))
                    # Remove
                    rem_edges[v].remove((nv, eid))
                    rem_edges[nv].remove((v, eid))
                    stack.append(nv)
                else:
                    circuit.append(stack.pop())
                    
            circuit.reverse()
            
            # Now split the circuit into trails based on dummy edges.
            # A trail is a sequence of vertices where the edges between them are real.
            # When we traverse a dummy edge, we "break" the trail.
            
            trails = []
            current_trail = [circuit[0]]
            
            for i in range(len(circuit) - 1):
                u = circuit[i]
                v = circuit[i+1]
                e = tuple(sorted((u, v)))
                
                # Check if this edge is a dummy edge
                is_dummy = False
                # We can check if the edge ID used was a dummy one.
                # But we didn't record which ID was used for each step in the circuit.
                # We can infer it: if the edge is a dummy edge, it's not in the original graph.
                # Wait, a dummy edge might coincide with a real edge.
                # If it coincides, we used the real edge first (or dummy first?).
                # In our `rem_edges` construction, we added both.
                # When we popped, we didn't know which one it was.
                # This is a flaw.
                
                # To fix: We must distinguish edges.
                # If (u,v) is a real edge AND a dummy edge, we have two instances.
                # In the Eulerian circuit, we use one instance.
                # If we use the real instance, it's part of a trail.
                # If we use the dummy instance, it's a break.
                
                # How to know which instance was used?
                # We can't from the vertex sequence alone if they overlap.
                
                # Solution: When constructing `rem_edges`, store the edge ID.
                # When popping, we get the ID.
                # We need to record the ID for each step in the circuit.
                
                # Let's restart this part with ID tracking.
            
            # Restart Augmented Graph Logic
            rem_edges = defaultdict(set)
            for e, ids in edge_to_ids.items():
                for eid in ids:
                    u, v = e
                    rem_edges[u].add((v, eid))
                    rem_edges[v].add((u, eid))
                    
            circuit = []
            circuit_edges = [] # List of (u, v, eid)
            stack = [start_v]
            
            while stack:
                v = stack[-1]
                if rem_edges[v]:
                    nv, eid = next(iter(rem_edges[v]))
                    rem_edges[v].remove((nv, eid))
                    rem_edges[nv].remove((v, eid))
                    stack.append(nv)
                    if len(stack) > 1:
                        prev_v = stack[-2]
                        circuit_edges.append((prev_v, nv, eid))
                else:
                    circuit.append(stack.pop())
                    
            # Now circuit_edges contains the sequence of edges used.
            # We can split this sequence into trails.
            
            trails = []
            current_trail_edges = []
            
            for u, v, eid in circuit_edges:
                if eid < num_real:
                    # Real edge
                    current_trail_edges.append((u, v))
                else:
                    # Dummy edge
                    # End current trail
                    if current_trail_edges:
                        # Convert edge list to vertex list
                        trail_verts = [current_trail_edges[0][0]]
                        for eu, ev in current_trail_edges:
                            trail_verts.append(ev)
                        trails.append(trail_verts)
                        current_trail_edges = []
                        
            if current_trail_edges:
                trail_verts = [current_trail_edges[0][0]]
                for eu, ev in current_trail_edges:
                    trail_verts.append(ev)
                trails.append(trail_verts)
                
            # Now repair each trail
            all_paths = []
            for trail in trails:
                if not trail:
                    continue
                # The trail edges are the edges in the original graph used in this segment.
                # We need to repair the vertex sequence into simple paths.
                # The edges used are exactly the ones in the trail segment.
                # We can pass the vertex sequence and the set of edges to repair_trail.
                # But repair_trail currently takes a trail (vertex seq) and finds edges from it.
                # This is correct.
                
                # However, repair_trail might produce paths that don't cover all edges of the trail?
                # No, it extracts maximal simple paths from the subgraph of the trail.
                # Since the trail is a walk, the subgraph is connected.
                # The greedy extraction should cover all edges.
                
                paths = repair_trail(trail, edges, adj)
                all_paths.extend(paths)
                
            # Calculate score: number of paths
            score = len(all_paths)
            
            if score < best_score:
                best_score = score
                best_paths = all_paths
                
            # Early stop if we hit the lower bound
            lower_bound = (len(odd_vertices) + 1) // 2
            if best_score <= lower_bound:
                break
                
        if best_paths is None:
            # Fallback
            return [[u, v] for u, v in edges]
            
        return best_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline - 5:
            break
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
    
    # If we ran out of time, fill remaining with trivial edge paths
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Total paths: {sum(len(r) for r in res)}")
