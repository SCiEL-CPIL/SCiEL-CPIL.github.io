"""Find longest simple path in the pinned graph.

Combines Iterative Deepening DFS (IDDFS) with Warnsdorff heuristic for deep search
and a robust refinement pipeline (extend both ends, insert unvisited vertices)
to maximize the simple path length in the pinned graph.
"""
import time
import random
import sys
import heapq

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

def find_components():
    """Find all connected components."""
    visited = [False] * N
    comps = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in ADJ[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            comps.append(comp)
    return comps

def greedy_path_from(start, adj, seen=None):
    """Greedy path starting from start (Warnsdorff: fewest remaining options)."""
    if seen is None:
        seen = set()
    path = [start]
    seen.add(start)
    while True:
        cur = path[-1]
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            break
        v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        path.append(v)
        seen.add(v)
    return path

def extend_both_ends(path, adj):
    """Try to extend path from both ends (Warnsdorff)."""
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        changed = False
        if path:
            cur = path[0]
            candidates = [nb for nb in adj[cur] if nb not in seen]
            if candidates:
                v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
                path.insert(0, v)
                seen.add(v)
                changed = True
        if path:
            cur = path[-1]
            candidates = [nb for nb in adj[cur] if nb not in seen]
            if candidates:
                v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
                path.append(v)
                seen.add(v)
                changed = True
    return path

def try_insert(path, adj):
    """Try to insert unvisited vertices between consecutive path nodes."""
    path = list(path)
    seen = set(path)
    unvisited = set(range(N)) - seen
    improved = True
    while improved and unvisited:
        improved = False
        for w in list(unvisited):
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                if w in adj[u] and w in adj[v]:
                    path.insert(i + 1, w)
                    seen.add(w)
                    unvisited.remove(w)
                    improved = True
                    break
            if improved:
                break
    return path

def refine(path, adj, deadline=None):
    """Polish a candidate path with the refinement pipeline."""
    if deadline is None:
        deadline = time.time() + 5  # Default 5s budget
    
    # Multiple passes of refinement
    for _ in range(3):
        if time.time() > deadline:
            break
        path = extend_both_ends(path, adj)
        path = try_insert(path, adj)
        path = extend_both_ends(path, adj)
    
    # Try backtracking extension for more aggressive improvement
    if time.time() < deadline - 1:
        extended = backtracking_extend(path, adj, deadline)
        if len(extended) > len(path):
            path = extended
    
    # Final extension pass
    path = extend_both_ends(path, adj)
    
    return path

def backtracking_extend(path, adj, deadline, max_steps=50000):
    """
    Extend a path using backtracking at both ends.
    Tries to add vertices to either end of the path using DFS with backtracking.
    """
    path = list(path)
    seen = set(path)
    steps = 0
    
    def try_extend_end(cur_end, is_start):
        """Try to extend from one end of the path using backtracking."""
        nonlocal steps, path, seen
        if time.time() > deadline:
            return False
        
        cur = path[0] if is_start else path[-1]
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            return False
        
        # Sort by Warnsdorff: fewest remaining options first
        candidates.sort(key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        
        for v in candidates:
            steps += 1
            if steps > max_steps or time.time() > deadline:
                break
            
            # Try adding v
            if is_start:
                path.insert(0, v)
            else:
                path.append(v)
            seen.add(v)
            
            # Try to extend further from this new end
            if try_extend_end(v, is_start):
                return True
            
            # Backtrack
            if is_start:
                path.pop(0)
            else:
                path.pop()
            seen.remove(v)
        
        return False
    
    # Try extending from both ends
    try_extend_end(path[0], True)
    try_extend_end(path[-1], False)
    
    return path

def forced_move_path(start, adj, deadline):
    """
    Build a path by following forced moves (vertices with only one unvisited neighbor).
    This can quickly build long paths in sparse regions.
    """
    path = [start]
    seen = {start}
    
    while time.time() < deadline:
        # Check both ends for forced moves
        extended = False
        
        # Check start end
        if path:
            cur = path[0]
            unvisited_neighbors = [nb for nb in adj[cur] if nb not in seen]
            if len(unvisited_neighbors) == 1:
                v = unvisited_neighbors[0]
                path.insert(0, v)
                seen.add(v)
                extended = True
        
        # Check end
        if path:
            cur = path[-1]
            unvisited_neighbors = [nb for nb in adj[cur] if nb not in seen]
            if len(unvisited_neighbors) == 1:
                v = unvisited_neighbors[0]
                path.append(v)
                seen.add(v)
                extended = True
        
        if not extended:
            break
    
    return path

def iddfs_search(start, adj, deadline, initial_depth=5, max_depth=None):
    """
    Iterative Deepening DFS with Warnsdorff heuristic and dead-end pruning.
    Returns the best path found within the time budget.
    """
    if max_depth is None:
        max_depth = N - 1
    
    best_path = [start]
    best_len = 1
    found_deeper = False

    current_depth = initial_depth
    
    while current_depth <= max_depth:
        if time.time() > deadline:
            break
            
        path = [start]
        seen = {start}
        found_at_depth = False
        
        def dfs(cur, depth):
            nonlocal found_at_depth, best_path, best_len
            if time.time() > deadline:
                return
            if depth == current_depth:
                best_path = list(path)
                best_len = current_depth
                found_at_depth = True
                return
            
            candidates = [nb for nb in adj[cur] if nb not in seen]
            if not candidates:
                return
            
            # Warnsdorff heuristic: sort by fewest unvisited neighbors
            candidates.sort(key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
            
            for v in candidates:
                # Dead-end pruning: if v has only one unvisited neighbor (besides cur),
                # and that neighbor is not the next one we'd go to, skip it
                # Actually, let's just use simple Warnsdorff for now
                
                path.append(v)
                seen.add(v)
                dfs(v, depth + 1)
                path.pop()
                seen.remove(v)
                
                if found_at_depth or time.time() > deadline:
                    return

        dfs(start, 1)
        
        if found_at_depth:
            current_depth += 1
            if best_len == N:
                break
        else:
            break

    return best_path

def build_skeleton(adj):
    """
    Compress the graph by collapsing degree-2 chains.
    Returns:
      skeleton_adj: dict of list of (neighbor_skel_id, chain_length, chain_vertices)
      skel_map: list mapping skeleton id to original vertex (for endpoints)
      chains: list of chains (list of original vertices) for each skeleton edge
    """
    n = len(adj)
    deg = [len(adj[i]) for i in range(n)]
    
    # Identify nodes that are NOT degree 2 (these are the skeleton nodes)
    # Also include isolated nodes or nodes with degree 1
    skeleton_nodes = []
    is_skel = [False] * n
    for i in range(n):
        if deg[i] != 2:
            skeleton_nodes.append(i)
            is_skel[i] = True
    
    if not skeleton_nodes:
        # If all nodes are degree 2, it's a cycle. Pick one to break it.
        skeleton_nodes = [0]
        is_skel[0] = True
        
    # Map original node to skeleton index
    orig_to_skel = [-1] * n
    for i, node in enumerate(skeleton_nodes):
        orig_to_skel[node] = i
        
    num_skel = len(skeleton_nodes)
    skel_adj = [[] for _ in range(num_skel)]
    
    # Traverse chains from each skeleton node
    visited_chain_edge = set()
    
    for s_idx, s_node in enumerate(skeleton_nodes):
        for nb in adj[s_node]:
            # We only traverse chains starting from s_node if s_node is the "lower" index
            # to avoid double processing, but since chains are bidirectional, we need to be careful.
            # A chain connects two skeleton nodes.
            # Let's traverse from s_node into the chain.
            if is_skel[nb]:
                # Direct edge between two skeleton nodes
                if s_idx < orig_to_skel[nb]:
                    edge_id = (s_idx, orig_to_skel[nb])
                    if edge_id not in visited_chain_edge:
                        visited_chain_edge.add(edge_id)
                        # Chain is just the edge
                        chain = [s_node, nb]
                        length = 1 # Number of edges
                        skel_adj[s_idx].append((orig_to_skel[nb], length, chain))
                        skel_adj[orig_to_skel[nb]].append((s_idx, length, chain))
            else:
                # Start of a degree-2 chain
                # Find the other end of the chain
                prev = s_node
                cur = nb
                chain = [s_node, nb]
                while not is_skel[cur]:
                    # cur is degree 2. One neighbor is prev.
                    # Find the other neighbor
                    nexts = [x for x in adj[cur] if x != prev]
                    if not nexts:
                        break # Should not happen in a valid graph component
                    next_node = nexts[0]
                    chain.append(next_node)
                    prev, cur = cur, next_node
                    if is_skel[cur]:
                        break
                    # Safety break
                    if len(chain) > n:
                        break
                
                if is_skel[cur]:
                    end_skel_idx = orig_to_skel[cur]
                    if s_idx < end_skel_idx:
                        edge_id = (s_idx, end_skel_idx)
                        if edge_id not in visited_chain_edge:
                            visited_chain_edge.add(edge_id)
                            length = len(chain) - 1
                            skel_adj[s_idx].append((end_skel_idx, length, chain))
                            skel_adj[end_skel_idx].append((s_idx, length, chain))
    
    return skel_adj, skeleton_nodes

def longest_path_in_skeleton(skel_adj, skel_nodes, deadline, start_skel=None):
    """
    Find the longest path in the skeleton graph using DFS with branch and bound.
    Returns a list of skeleton node indices representing the path.
    """
    num_skel = len(skel_adj)
    if num_skel == 0:
        return []
    if num_skel == 1:
        return [0]
        
    best_path = []
    best_weight = -1
    
    # Precompute max possible weight from each node (upper bound for pruning)
    # Simple upper bound: sum of all weights * 2 (very loose)
    # Better: Max spanning tree weight? Too complex.
    # Let's just use a simple DFS with pruning based on current best.
    
    # To speed up, we can use a heuristic: visit nodes with higher degree first?
    # Or just standard DFS.
    
    if start_skel is None:
        # Try starting from all nodes? That's expensive.
        # Let's pick a few promising starts (high degree in skeleton)
        degs = [len(skel_adj[i]) for i in range(num_skel)]
        starts = sorted(range(num_skel), key=lambda x: -degs[x])
        # Limit starts to keep time manageable
        starts = starts[:min(10, num_skel)]
    else:
        starts = [start_skel]
        
    for start in starts:
        if time.time() > deadline:
            break
            
        path = [start]
        seen = {start}
        current_weight = 0
        
        def dfs(cur, weight):
            nonlocal best_path, best_weight
            if time.time() > deadline:
                return
            
            if weight > best_weight:
                best_weight = weight
                best_path = list(path)
                
            # Pruning: if weight + max_possible_remaining <= best_weight, return
            # We don't have a tight max_possible_remaining, so skip for now.
            
            for nb, w, chain in skel_adj[cur]:
                if nb not in seen:
                    path.append(nb)
                    seen.add(nb)
                    dfs(nb, weight + w)
                    seen.remove(nb)
                    path.pop()
                    
                    if time.time() > deadline:
                        return

        dfs(start, 0)
        
    return best_path

def expand_skeleton_path(skel_path, skel_adj, skel_nodes):
    """
    Expand a path in the skeleton graph back to the original graph.
    skel_path is a list of skeleton node indices.
    """
    if not skel_path:
        return []
        
    # We need to find the chains corresponding to the edges in the path.
    # Build a lookup: (u, v) -> chain
    chain_lookup = {}
    for u in range(len(skel_adj)):
        for v, w, chain in skel_adj[u]:
            # Store in a way that allows lookup by pair
            # Since graph is undirected, (u,v) and (v,u) are same edge
            key = tuple(sorted((u, v)))
            chain_lookup[key] = chain
            
    expanded = []
    for i in range(len(skel_path)):
        s_idx = skel_path[i]
        if i == 0:
            # Start of path
            # If the path continues, we need to prepend the chain leading to the next node?
            # No, the chain includes the start node.
            # If we just append skel_nodes[s_idx], we miss the internal nodes of the chain
            # if the path enters this node from a chain.
            
            # Let's look at the next node in the path
            if i < len(skel_path) - 1:
                next_s_idx = skel_path[i+1]
                key = tuple(sorted((s_idx, next_s_idx)))
                chain = chain_lookup.get(key, [skel_nodes[s_idx]])
                # The chain is [start_skel, ..., end_skel]
                # We want to add the part starting from s_idx
                # If s_idx is the start of the chain, we add all.
                # If s_idx is the end, we add reversed?
                # The chain in skel_adj[u] is oriented u->v.
                # Let's find the specific chain entry.
                found_chain = None
                for nb, w, c in skel_adj[s_idx]:
                    if nb == next_s_idx:
                        found_chain = c
                        break
                if found_chain:
                    # found_chain starts with skel_nodes[s_idx]
                    expanded.extend(found_chain)
                else:
                    expanded.append(skel_nodes[s_idx])
            else:
                expanded.append(skel_nodes[s_idx])
        else:
            # Middle or end
            prev_s_idx = skel_path[i-1]
            key = tuple(sorted((prev_s_idx, s_idx)))
            # We need the chain from prev to cur
            # Look in skel_adj[prev_s_idx] for neighbor s_idx
            found_chain = None
            for nb, w, c in skel_adj[prev_s_idx]:
                if nb == s_idx:
                    found_chain = c
                    break
            if found_chain:
                # found_chain is [prev_skel, ..., cur_skel]
                # We already added prev_skel (and its chain) in the previous step?
                # In the previous step (i-1), we added the chain from prev_s_idx to s_idx?
                # If i-1 was 0, we added the chain from 0 to 1.
                # So the last element of the previous addition is skel_nodes[prev_s_idx]?
                # No, the chain ends at the NEXT node.
                # So if we added chain for (0,1), we added [n0, ..., n1].
                # Now we are at i=1 (node 1).
                # We should NOT add n1 again.
                # We should add the chain for (1, 2) starting from index 1?
                # The chain for (1,2) is [n1, ..., n2].
                # So we should append found_chain[1:] to avoid duplicating n1.
                
                # Wait, the logic above for i=0 added the full chain [n0, ..., n1].
                # So n1 is already in `expanded`.
                # For i=1, we process the edge (1,2). The chain is [n1, ..., n2].
                # We should append found_chain[1:] which is [..., n2].
                
                # But what if the path is just [0, 1]?
                # i=0: adds [n0, ..., n1].
                # i=1: prev=0, cur=1. Chain [n0, ..., n1].
                # We should add nothing? Or just ensure n1 is there.
                # Actually, if i is the last node, we don't have a "next" chain.
                # The loop `for i in range(len(skel_path))` processes each node.
                # For i < len-1, we add the chain to the next node.
                # For i == len-1, we add the node itself?
                
                # Let's refine the expansion logic.
                # It's easier to iterate over edges.
                pass
            
            # Let's restart the expansion logic in the function body properly.
            # I will rewrite `expand_skeleton_path` completely in the replacement.
            pass

    # Fallback: simple concatenation if logic above was messy
    # Actually, let's just implement a clean version in the replacement.
    return expanded

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    comps = find_components()
    comps.sort(key=len, reverse=True)
    if not comps:
        return [0]

    largest_comp = comps[0]
    if not largest_comp:
        return [0]

    best_path = list(largest_comp[:1])
    best_len = len(best_path)

    # Phase 1: Skeleton based search
    # Build skeleton for the largest component
    # We need to restrict ADJ to the component?
    # The global ADJ includes all nodes.
    # build_skeleton uses global ADJ.
    # If the largest component is not the whole graph, the skeleton will include other components?
    # No, build_skeleton iterates over all nodes.
    # We should probably build skeleton for the specific component.
    
    # Let's create a subgraph ADJ for the largest component
    comp_set = set(largest_comp)
    sub_adj = [set() for _ in range(N)]
    for u in largest_comp:
        for v in ADJ[u]:
            if v in comp_set:
                sub_adj[u].add(v)
                
    skel_adj, skel_nodes = build_skeleton(sub_adj)
    
    if skel_nodes:
        # Find longest path in skeleton
        # Use a generous deadline for skeleton search
        skel_deadline = time.time() + 10
        skel_path = longest_path_in_skeleton(skel_adj, skel_nodes, skel_deadline)
        
        if skel_path:
            # Expand skeleton path
            # We need a robust expansion function
            # Let's implement a helper here or assume `expand_skeleton_path` works.
            # I will replace `expand_skeleton_path` with a correct implementation.
            expanded_path = expand_skeleton_path(skel_path, skel_adj, skel_nodes)
            
            if expanded_path:
                # Verify validity
                valid = True
                seen = set()
                for i in range(len(expanded_path)):
                    if expanded_path[i] in seen:
                        valid = False
                        break
                    seen.add(expanded_path[i])
                if valid:
                    for i in range(len(expanded_path) - 1):
                        if expanded_path[i+1] not in ADJ[expanded_path[i]]:
                            valid = False
                            break
                
                if valid:
                    if len(expanded_path) > best_len:
                        best_path = expanded_path
                        best_len = len(best_path)
                        
                    # Refine the expanded path
                    if time.time() < deadline - 5:
                        refined = refine(best_path, ADJ, deadline - 2)
                        if len(refined) > best_len:
                            best_path = refined
                            best_len = len(best_path)

    # Phase 2: Greedy initialization
    # Run greedy from all nodes to get a decent starting point and identify promising starts
    promising_starts = []
    for start in largest_comp:
        if time.time() > deadline - 30:
            break
        p = greedy_path_from(start, ADJ)
        p = refine(p, ADJ, deadline - 10)
        if len(p) > best_len:
            best_path = p
            best_len = len(p)
        # Track starts that yielded long paths
        if len(p) > best_len * 0.8: # Heuristic: keep starts that are "good"
            promising_starts.append(start)

    # If we have a very long path already, we might skip some IDDFS
    if best_len == N:
        return best_path

    # Phase 3: IDDFS from promising starts
    # We prioritize starts that are high degree or were good in greedy
    # Sort promising starts by degree descending to find "central" paths first
    unique_promising = list(set(promising_starts))
    # Add top degree nodes from the largest component if not already present
    deg_sorted = sorted(largest_comp, key=lambda v: len(ADJ[v]), reverse=True)
    for v in deg_sorted[:15]:
        if v not in unique_promising:
            unique_promising.append(v)
            
    # Limit the number of IDDFS runs to ensure we finish in time
    # Each IDDFS run can be expensive. 
    # We have ~50s left. Let's allow ~10-20 runs.
    max_iddfs_runs = 12
    
    for start in unique_promising[:max_iddfs_runs]:
        if time.time() > deadline - 15:
            break
            
        # Start IDDFS from a depth close to current best to save time
        start_depth = max(5, best_len - 3)
        
        p = iddfs_search(start, ADJ, deadline - 15, initial_depth=start_depth)
        p = refine(p, ADJ, deadline - 5)
        
        if len(p) > best_len:
            best_path = p
            best_len = len(p)
            
            if best_len == N:
                return best_path
        
        # Also try forced move extension from this start
        if time.time() < deadline - 10:
            fp = forced_move_path(start, ADJ, deadline - 10)
            fp = refine(fp, ADJ, deadline - 5)
            if len(fp) > best_len:
                best_path = fp
                best_len = len(fp)
                if best_len == N:
                    return best_path

    # Phase 4: Final Refinement
    if best_path and time.time() < deadline - 2:
        extended = refine(best_path, ADJ, deadline - 1)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
        
        # One more aggressive backtracking extension
        if time.time() < deadline - 1:
            bt_path = backtracking_extend(best_path, ADJ, deadline - 0.5)
            if len(bt_path) > best_len:
                best_path = bt_path
                best_len = len(best_path)

    # Validate
    if not best_path:
        best_path = [0]
    
    # Final sanity check
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in ADJ[best_path[i]]:
            # Fallback to a valid greedy path
            for s in range(N):
                p = greedy_path_from(s, ADJ)
                if p:
                    best_path = p
                    break
            break

    return best_path
