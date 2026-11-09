import time
import sys

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Identify core nodes (degree >= 3)
    core_nodes = [i for i in range(n) if len(adj[i]) >= 3]
    
    # If no core nodes, the graph is a collection of paths and cycles
    if not core_nodes:
        # Find connected components
        visited = [False] * n
        best_path = []
        for i in range(n):
            if not visited[i]:
                # BFS/DFS to get component
                comp = []
                stack = [i]
                visited[i] = True
                while stack:
                    node = stack.pop()
                    comp.append(node)
                    for nei in adj[node]:
                        if not visited[nei]:
                            visited[nei] = True
                            stack.append(nei)
                # If component is a path or cycle, longest path is size of component (or size-1 for cycle? No, simple path in cycle is N-1)
                # Actually, if it's a tree (path), longest path is diameter.
                # If it's a cycle, longest path is N-1.
                # For a tree, we can find the diameter.
                # But since deg <= 2, it's a path or cycle.
                # If it's a path, the longest simple path is the whole path.
                # If it's a cycle, the longest simple path is N-1.
                # How to distinguish?
                # In a path, there are 2 nodes with deg 1.
                # In a cycle, all nodes have deg 2.
                degs = [len(adj[node]) for node in comp]
                if all(d == 2 for d in degs):
                    # Cycle
                    path_len = len(comp) - 1
                    # Construct path: just take all but one edge
                    # We need to return the actual vertices.
                    # Let's find the cycle order.
                    # Start at comp[0], next is a neighbor, etc.
                    path = [comp[0]]
                    visited_cycle = {comp[0]}
                    curr = comp[0]
                    while len(path) < path_len:
                        for nei in adj[curr]:
                            if nei not in visited_cycle:
                                path.append(nei)
                                visited_cycle.add(nei)
                                curr = nei
                                break
                    if len(path) > len(best_path):
                        best_path = path
                else:
                    # Path
                    # Find endpoint (deg 1)
                    start = None
                    for node in comp:
                        if len(adj[node]) == 1:
                            start = node
                            break
                    if start is None:
                        # Single node?
                        start = comp[0]
                    path = [start]
                    visited_path = {start}
                    curr = start
                    while True:
                        next_node = None
                        for nei in adj[curr]:
                            if nei not in visited_path:
                                next_node = nei
                                break
                        if next_node is None:
                            break
                        path.append(next_node)
                        visited_path.add(next_node)
                        curr = next_node
                    if len(path) > len(best_path):
                        best_path = path
        return best_path

    # Compress graph
    # Map core nodes to indices in skeleton
    core_index = {node: i for i, node in enumerate(core_nodes)}
    num_core = len(core_nodes)
    
    # Skeleton adjacency: list of (neighbor_core_index, weight)
    # Weight is number of vertices added when traversing from current core to neighbor core, excluding current.
    # Or rather, the length of the chain in terms of vertices.
    # Let's define weight as the number of vertices in the simple path from u to v, excluding u.
    # So if u-v is direct edge, weight is 1 (vertex v).
    # If u-x-v, weight is 2 (vertices x, v).
    
    skeleton_adj = [[] for _ in range(num_core)]
    # To handle parallel edges, we'll store max weight for each pair?
    # Or just store all, and during DFS, we can only pick one edge between a pair?
    # Since we are looking for a simple path in the skeleton, we visit each core node at most once.
    # So we won't traverse u->v and then v->u later.
    # But we might have two different chains between u and v.
    # We can only use one of them. The one with the larger weight is always better.
    # So we can just keep the max weight for each pair (u,v).
    
    max_weight = {}
    
    for u in core_nodes:
        u_idx = core_index[u]
        for v in adj[u]:
            if v in core_index:
                # Direct edge to core
                v_idx = core_index[v]
                # Weight is 1 (just v)
                key = (min(u_idx, v_idx), max(u_idx, v_idx))
                if key not in max_weight or max_weight[key] < 1:
                    max_weight[key] = 1
            else:
                # v is not core (deg 1 or 2)
                # Follow chain
                path = [v]
                curr = v
                prev = u
                end_node = None
                while True:
                    nexts = [x for x in adj[curr] if x != prev]
                    if not nexts:
                        # Hit a leaf
                        end_node = 'leaf'
                        break
                    if len(nexts) > 1:
                        # Hit a core node (deg >= 3)
                        # Since curr is not core, deg is 2. So there is exactly 1 next.
                        # Wait, if deg(curr) == 2, and one is prev, the other is next.
                        # If next is core, we stop.
                        next_node = nexts[0]
                        if next_node in core_index:
                            path.append(next_node)
                            end_node = next_node
                            break
                        else:
                            # This shouldn't happen if we defined core correctly?
                            # If deg(next_node) < 3, it's not core.
                            # But we are following a chain of non-core nodes.
                            # So next_node is not core.
                            path.append(next_node)
                            prev = curr
                            curr = next_node
                            continue
                    else:
                        # deg(curr) == 1? No, we started from u (core), so v has at least 1 neighbor (u).
                        # If v is leaf, we would have caught it?
                        # If v is deg 1, adj[v] = {u}. nexts is empty. Caught above.
                        # If v is deg 2, nexts has 1 element.
                        next_node = nexts[0]
                        path.append(next_node)
                        prev = curr
                        curr = next_node
                        continue
                
                if end_node == 'leaf':
                    # This is a tail attached to u
                    # Length of tail (vertices excluding u) is len(path)
                    # We store this as a tail length for u
                    # We need to keep the top 2 tail lengths for each core node
                    pass # Handle tails separately
                
                else:
                    # end_node is a core node
                    v_idx = core_index[end_node]
                    # Weight is len(path)
                    # path contains v, ..., end_node
                    # Number of vertices added is len(path)
                    key = (min(u_idx, v_idx), max(u_idx, v_idx))
                    if key not in max_weight or max_weight[key] < len(path):
                        max_weight[key] = len(path)

    # Build tail lengths for each core node
    # tails[i] = list of lengths of tails attached to core_nodes[i]
    tails = [[] for _ in range(num_core)]
    
    for u in core_nodes:
        u_idx = core_index[u]
        for v in adj[u]:
            if v in core_index:
                continue
            # Follow chain
            path = [v]
            curr = v
            prev = u
            while True:
                nexts = [x for x in adj[curr] if x != prev]
                if not nexts:
                    break
                next_node = nexts[0]
                if next_node in core_index:
                    # Should not happen for tails
                    break
                path.append(next_node)
                prev = curr
                curr = next_node
            tails[u_idx].append(len(path))
            
    # Sort tails in descending order, keep top 2
    for i in range(num_core):
        tails[i].sort(reverse=True)
        if len(tails[i]) > 2:
            tails[i] = tails[i][:2]
            
    # Build skeleton adjacency list from max_weight
    for (u_idx, v_idx), w in max_weight.items():
        skeleton_adj[u_idx].append((v_idx, w))
        skeleton_adj[v_idx].append((u_idx, w))
        
    # DFS on skeleton to find longest path
    best_len = 0
    best_path_core = []
    
    # We need to reconstruct the original path.
    # This is tricky. We need to know which chain was used.
    # Let's store the actual path vertices for each skeleton edge?
    # That would be expensive.
    # Alternative: Just compute the max length first.
    # Then, if we want to return the path, we can do a second pass or store the path.
    # Given the small size, we can store the path.
    
    # Let's modify max_weight to store the actual path vertices (excluding start core)
    # path_map[(u_idx, v_idx)] = [v, ..., end_core]
    
    path_map = {}
    # Re-compute path_map to get the actual vertices
    for u in core_nodes:
        u_idx = core_index[u]
        for v in adj[u]:
            if v in core_index:
                v_idx = core_index[v]
                key = (min(u_idx, v_idx), max(u_idx, v_idx))
                # Check if we need to update
                if key not in path_map or len(path_map[key][1]) < 1: # Direct edge has path [v]
                    path_map[key] = (u_idx, [v]) # Store start core idx and path
            else:
                path = [v]
                curr = v
                prev = u
                end_node = None
                while True:
                    nexts = [x for x in adj[curr] if x != prev]
                    if not nexts:
                        break
                    next_node = nexts[0]
                    if next_node in core_index:
                        path.append(next_node)
                        end_node = next_node
                        break
                    path.append(next_node)
                    prev = curr
                    curr = next_node
                if end_node:
                    v_idx = core_index[end_node]
                    key = (min(u_idx, v_idx), max(u_idx, v_idx))
                    if key not in path_map or len(path_map[key][1]) < len(path):
                        path_map[key] = (u_idx, path)
                    else:
                        # Check if other direction is better?
                        # path_map stores the path from the min index?
                        # We need to be careful with direction.
                        pass

    # This is getting complicated. Let's just do the DFS to find the max length first.
    # Then, we can backtrack to find the path.
    
    def dfs_skeleton(u_idx, visited, current_len, current_path):
        nonlocal best_len, best_path_core
        if current_len > best_len:
            best_len = current_len
            best_path_core = current_path[:]
            
        for v_idx, w in skeleton_adj[u_idx]:
            if v_idx not in visited:
                # Calculate new length
                # w is the number of vertices added.
                new_len = current_len + w
                dfs_skeleton(v_idx, visited | {v_idx}, new_len, current_path + [v_idx])
                
    # We need to account for tails.
    # The DFS above only considers paths starting and ending at core nodes.
    # We need to add tail lengths at the start and end.
    
    # Let's modify the DFS to track the start and end core nodes, and then add tails.
    # Or, simpler: 
    # For each simple path in the skeleton u1...uk:
    # Base len = 1 + sum(weights)
    # Max start tail = tails[u1][0] if tails[u1] else 0
    # Max end tail = tails[uk][0] if tails[uk] else 0
    # If k==1:
    #   Max len = 1 + (tails[u1][0] if tails[u1] else 0) + (tails[u1][1] if len(tails[u1])>1 else 0)
    # Else:
    #   Max len = Base len + Max start tail + Max end tail
    
    # We can just iterate over all simple paths in the skeleton.
    # Since num_core is small, this is fine.
    
    all_paths = []
    def gen_paths(u_idx, visited, path):
        all_paths.append(path[:])
        for v_idx, _ in skeleton_adj[u_idx]:
            if v_idx not in visited:
                gen_paths(v_idx, visited | {v_idx}, path + [v_idx])
                
    for i in range(num_core):
        gen_paths(i, {i}, [i])
        
    max_total_len = 0
    best_skel_path = []
    
    for path in all_paths:
        if not path: continue
        u_start = path[0]
        u_end = path[-1]
        
        base_len = 1 # for u_start
        for i in range(len(path)-1):
            u1 = path[i]
            u2 = path[i+1]
            key = (min(u1, u2), max(u1, u2))
            w = max_weight.get(key, 0)
            base_len += w
            
        if len(path) == 1:
            t1 = tails[u_start][0] if tails[u_start] else 0
            t2 = tails[u_start][1] if len(tails[u_start]) > 1 else 0
            total_len = base_len + t1 + t2
        else:
            t_start = tails[u_start][0] if tails[u_start] else 0
            t_end = tails[u_end][0] if tails[u_end] else 0
            total_len = base_len + t_start + t_end
            
        if total_len > max_total_len:
            max_total_len = total_len
            best_skel_path = path
            
    if not best_skel_path:
        return [0] # Fallback
        
    # Reconstruct original path
    # We need to expand the skeleton path back to original vertices.
    
    # Get tails for start and end
    start_core = best_skel_path[0]
    end_core = best_skel_path[-1]
    
    start_tail_len = 0
    end_tail_len = 0
    if len(best_skel_path) == 1:
        if tails[start_core]:
            start_tail_len = tails[start_core][0]
        if len(tails[start_core]) > 1:
            end_tail_len = tails[start_core][1]
    else:
        if tails[start_core]:
            start_tail_len = tails[start_core][0]
        if tails[end_core]:
            end_tail_len = tails[end_core][0]
            
    # We need the actual vertices of the tails.
    # This requires storing the tail paths.
    # Let's re-run the tail collection to store the paths.
    
    tail_paths = {i: [] for i in range(num_core)}
    for u in core_nodes:
        u_idx = core_index[u]
        for v in adj[u]:
            if v in core_index:
                continue
            path = [v]
            curr = v
            prev = u
            while True:
                nexts = [x for x in adj[curr] if x != prev]
                if not nexts:
                    break
                next_node = nexts[0]
                if next_node in core_index:
                    break
                path.append(next_node)
                prev = curr
                curr = next_node
            tail_paths[u_idx].append(path)
            
    # Sort tail_paths by length descending
    for i in range(num_core):
        tail_paths[i].sort(key=lambda x: len(x), reverse=True)
        
    # Build final path
    final_path = []
    
    # Add start tail
    if start_tail_len > 0:
        # Find the tail with length start_tail_len
        # There might be multiple, pick the first one
        for tp in tail_paths[start_core]:
            if len(tp) == start_tail_len:
                final_path.extend(tp)
                break
                
    # Add core path
    # We need to expand the core path using path_map
    # path_map needs to be built correctly.
    
    # Let's rebuild path_map properly.
    # path_map[(u_idx, v_idx)] = list of vertices from u to v, excluding u.
    
    path_map = {}
    for u in core_nodes:
        u_idx = core_index[u]
        for v in adj[u]:
            if v in core_index:
                v_idx = core_index[v]
                key = (min(u_idx, v_idx), max(u_idx, v_idx))
                # Direct edge
                if key not in path_map:
                    path_map[key] = {u_idx: [v], v_idx: [u]}
                else:
                    # Already exists, do nothing (direct edge is unique)
                    pass
            else:
                path = [v]
                curr = v
                prev = u
                end_node = None
                while True:
                    nexts = [x for x in adj[curr] if x != prev]
                    if not nexts:
                        break
                    next_node = nexts[0]
                    if next_node in core_index:
                        path.append(next_node)
                        end_node = next_node
                        break
                    path.append(next_node)
                    prev = curr
                    curr = next_node
                if end_node:
                    v_idx = core_index[end_node]
                    key = (min(u_idx, v_idx), max(u_idx, v_idx))
                    # Store path for both directions
                    if key not in path_map:
                        path_map[key] = {}
                    # From u to v: path is [v, ..., end_node]
                    # From v to u: path is [u, ..., v] reversed? No, just reverse the list and swap u,v?
                    # The list `path` is vertices from u to end_node (excluding u).
                    # So for u -> v, we use `path`.
                    # For v -> u, we need the vertices from v to u (excluding v).
                    # That would be `path` reversed? No.
                    # path = [x1, x2, ..., xk, v].
                    # Reverse: [v, xk, ..., x1].
                    # But we need to exclude the start node.
                    # So for v -> u, the vertices are [xk, ..., x1].
                    # Which is `path[:-1]` reversed.
                    
                    if u_idx not in path_map[key]:
                        path_map[key][u_idx] = path
                    else:
                        # Keep the longer one? They should be the same length.
                        pass
                        
                    rev_path = list(reversed(path[:-1]))
                    if v_idx not in path_map[key]:
                        path_map[key][v_idx] = rev_path
                    else:
                        pass

    # Expand core path
    if len(best_skel_path) > 0:
        final_path.append(core_nodes[best_skel_path[0]])
        
    for i in range(len(best_skel_path) - 1):
        u_idx = best_skel_path[i]
        v_idx = best_skel_path[i+1]
        key = (min(u_idx, v_idx), max(u_idx, v_idx))
        if key in path_map and u_idx in path_map[key]:
            final_path.extend(path_map[key][u_idx])
        else:
            # Error in path_map construction
            pass
            
    # Add end tail
    if end_tail_len > 0:
        for tp in tail_paths[end_core]:
            if len(tp) == end_tail_len:
                final_path.extend(tp)
                break
                
    # Verify path
    # Check if all vertices are distinct
    if len(set(final_path)) != len(final_path):
        # Fallback to simple greedy
        a=[set() for _ in range(P['n'])]
        for u,v in P['edges']:a[u].add(v);a[v].add(u)
        best=[]
        for start in range(P['n']):
            path=[start];seen={start}
            while a[path[-1]]-seen:
                v=min(a[path[-1]]-seen,key=lambda w:(len(a[w]-seen),w))
                path.append(v);seen.add(v)
            if len(path)>len(best):best=path
        return best
        
    # Check if edges exist
    for i in range(len(final_path)-1):
        if final_path[i+1] not in adj[final_path[i]]:
            # Fallback
            a=[set() for _ in range(P['n'])]
            for u,v in P['edges']:a[u].add(v);a[v].add(u)
            best=[]
            for start in range(P['n']):
                path=[start];seen={start}
                while a[path[-1]]-seen:
                    v=min(a[path[-1]]-seen,key=lambda w:(len(a[w]-seen),w))
                    path.append(v);seen.add(v)
                if len(path)>len(best):best=path
            return best
            
    return final_path
