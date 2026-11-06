import time
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    from random import Random
    import heapq

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges):
        if not edges:
            return []

        # 1. Build graph structure
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # 2. Find Maximum Spanning Tree (MaxST)
        # Using Prim's algorithm with a max-heap.
        # Since all weights are 1, any MST works, but we want to maximize the chance
        # of covering many edges in the tree structure itself if possible, 
        # though for uniform weights it's just an MST.
        # Actually, for path decomposition, we want the tree to be "long" and "thin" 
        # to facilitate leaf pairing. A standard MST is fine.
        
        visited = set()
        mst_edges = []
        if adj:
            # Start from arbitrary node
            start_node = next(iter(adj))
            visited.add(start_node)
            heap = []
            
            # Initialize heap with neighbors of start
            for nb in adj[start_node]:
                heapq.heappush(heap, (-1, start_node, nb)) # -1 for max heap behavior on weight 1
            
            while heap:
                w, u, v = heapq.heappop(heap)
                if v in visited:
                    continue
                visited.add(v)
                mst_edges.append((min(u, v), max(u, v)))
                
                # Add neighbors of v
                for nb in adj[v]:
                    if nb not in visited:
                        heapq.heappush(heap, (-1, v, nb))
            
            # If graph is disconnected (shouldn't be per task description, but safe)
            # Find unvisited components
            all_nodes = set()
            for u, v in edges:
                all_nodes.add(u)
                all_nodes.add(v)
            
            for node in all_nodes:
                if node not in visited:
                    visited.add(node)
                    heap = []
                    for nb in adj[node]:
                        if nb not in visited:
                            heapq.heappush(heap, (-1, node, nb))
                    while heap:
                        w, u, v = heapq.heappop(heap)
                        if v in visited:
                            continue
                        visited.add(v)
                        mst_edges.append((min(u, v), max(u, v)))
                        for nb in adj[v]:
                            if nb not in visited:
                                heapq.heappush(heap, (-1, v, nb))

        mst_set = set(mst_edges)
        non_tree_edges = [e for e in edges if e not in mst_set]

        # 3. Decompose MST into paths
        # Strategy: Repeatedly find a leaf, walk to the next leaf (or dead end), 
        # record the path, remove it.
        # To be efficient, we can maintain a degree map and a list of leaves.
        
        mst_paths = []
        if mst_edges:
            # Build adjacency for MST only
            mst_adj = defaultdict(list)
            for u, v in mst_edges:
                mst_adj[u].append(v)
                mst_adj[v].append(u)
            
            deg = {v: len(neigh) for v, neigh in mst_adj.items()}
            # Initialize leaves (degree 1)
            leaves = [v for v, d in deg.items() if d == 1]
            
            # If no leaves (single node or cycle - but MST is a tree, so if >1 node, has leaves)
            # Handle isolated nodes in MST? No, MST connects components.
            
            remaining_mst_edges = set(mst_edges)
            
            while remaining_mst_edges:
                # Pick a leaf
                if not leaves:
                    # This can happen if we've reduced to a single node with no edges? 
                    # Or if we have a cycle? MST has no cycles.
                    # If remaining edges exist, there must be a leaf.
                    # Re-scan for leaves if list is empty (shouldn't happen often)
                    for v in list(deg.keys()):
                        if deg[v] == 1:
                            leaves.append(v)
                            break
                    if not leaves:
                        # Fallback: pick any node with degree > 0
                        for v in deg:
                            if deg[v] > 0:
                                leaves.append(v)
                                break
                        if not leaves:
                            break

                start = leaves.pop(0)
                
                # Walk from start
                path = [start]
                current = start
                used_in_path = {start}
                
                # We need to ensure we don't revisit vertices (simple path)
                # In a tree, any walk without repeating edges is simple if we don't backtrack.
                # But we are removing edges, so the remaining graph is a forest.
                
                while True:
                    # Find next node
                    next_v = None
                    for nb in mst_adj[current]:
                        edge = tuple(sorted((current, nb)))
                        if edge in remaining_mst_edges and nb not in used_in_path:
                            # Prefer neighbors with higher degree? Or just any?
                            # Greedy: pick any. To maximize path length, 
                            # maybe avoid degree 1 nodes if possible?
                            # Actually, just walking to the end of the branch is good.
                            next_v = nb
                            break
                    
                    if next_v is None:
                        break
                    
                    # Move
                    edge = tuple(sorted((current, next_v)))
                    remaining_mst_edges.discard(edge)
                    
                    # Update degrees
                    deg[current] -= 1
                    deg[next_v] -= 1
                    
                    # If current becomes a leaf (and not start? start is already processed)
                    # We don't need to add current to leaves unless it's an intermediate node 
                    # that becomes a leaf after removal? No, we are removing the edge we just used.
                    # The node `current` loses a neighbor. If it had degree 2, it becomes 1.
                    # But `current` is already in the path. We don't want to start a new path from it 
                    # unless it's the end of this path.
                    
                    # If next_v becomes a leaf, add to leaves list?
                    # Yes, if deg[next_v] becomes 1, it's a leaf in the remaining forest.
                    if deg[next_v] == 1:
                        leaves.append(next_v)
                    
                    # If current becomes a leaf?
                    # If deg[current] becomes 1, it's a leaf. But it's the previous node.
                    # If the path continues, it's not a leaf for the *next* path start 
                    # unless the path ends here.
                    # Actually, if deg[current] == 1 after removal, and current != start,
                    # it means the path "dangling" from current is now a separate component 
                    # or current is the new end. But we are moving forward.
                    # The edge (current, next_v) is removed.
                    # If deg[current] was 2, it is now 1. Current is a leaf in the remaining graph.
                    # Should we add it to leaves? 
                    # If we add it, we might start a new path from current later.
                    # But current is already in `path`. 
                    # Wait, the path is `start -> ... -> current -> next_v`.
                    # If we stop at next_v, current is an internal node.
                    # If we continue from next_v, current is internal.
                    # The "leaf" status in the *remaining* graph is what matters for future starts.
                    # If deg[current] becomes 1, it is a leaf in the remaining graph.
                    # So yes, add it to leaves.
                    if deg[current] == 1 and current != start:
                        leaves.append(current)
                        
                    path.append(next_v)
                    used_in_path.add(next_v)
                    current = next_v
                
                if len(path) > 1:
                    mst_paths.append(path)

        # 4. Handle non-tree edges
        # The non-tree edges form a set of chords. 
        # We can try to insert them into existing paths or form new ones.
        # Simple approach: Treat non-tree edges as a separate graph and decompose it.
        # Since n is small, we can just use the same leaf-pairing on the non-tree edges?
        # No, non-tree edges might not form a tree.
        # Let's use a simple greedy for non-tree edges:
        # While non_tree_edges remains:
        #   Pick an edge (u,v)
        #   Try to extend from u and v using only non_tree_edges? 
        #   Or just form a path of length 1?
        #   Given n=30, non-tree edges are few. 
        #   Let's just append them as single-edge paths for now, or try to merge.
        
        non_tree_paths = []
        remaining_non_tree = set(non_tree_edges)
        
        while remaining_non_tree:
            # Pick an edge
            e = next(iter(remaining_non_tree))
            u, v = e
            remaining_non_tree.discard(e)
            
            # Try to extend
            path = [u, v]
            used = {u, v}
            
            # Extend from u
            while True:
                extended = False
                for nb in adj[u]:
                    edge = tuple(sorted((u, nb)))
                    if edge in remaining_non_tree and nb not in used:
                        path.insert(0, nb)
                        used.add(nb)
                        remaining_non_tree.discard(edge)
                        u = nb
                        extended = True
                        break
                if not extended:
                    break
            
            # Extend from v
            while True:
                extended = False
                for nb in adj[v]:
                    edge = tuple(sorted((v, nb)))
                    if edge in remaining_non_tree and nb not in used:
                        path.append(nb)
                        used.add(nb)
                        remaining_non_tree.discard(edge)
                        v = nb
                        extended = True
                        break
                if not extended:
                    break
            
            non_tree_paths.append(path)

        all_paths = mst_paths + non_tree_paths
        
        # 5. Optional: Merge short paths if possible?
        # For now, return as is.
        
        return all_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
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
