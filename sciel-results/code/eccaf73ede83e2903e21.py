import time
import random
from collections import defaultdict

def instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def initial_decomposition(n, edges):
    """
    Creates an initial valid path decomposition using a greedy approach.
    This serves as the starting point for the matching-based improvement.
    """
    if not edges:
        return []
    
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    paths = []
    
    while remaining:
        # Pick an edge. To be more effective, we might pick edges that lead to longer paths,
        # but for initialization, any edge works. Let's pick one with high degree sum to encourage longer paths.
        best_edge = None
        best_score = -1
        for e in remaining:
            u, v = e
            score = len(adj[u]) + len(adj[v])
            if score > best_score:
                best_score = score
                best_edge = e
        
        u, v = best_edge
        remaining.remove((u, v))
        adj[u].discard(v)
        adj[v].discard(u)
        
        path = [u, v]
        path_set = {u, v}
        
        # Extend from v
        curr = v
        while True:
            next_v = None
            # Prefer neighbors not in path
            candidates = [w for w in adj[curr] if w not in path_set]
            if candidates:
                # Pick one that allows further extension? Or just first.
                # To make paths longer, maybe pick one with most unvisited neighbors.
                next_v = max(candidates, key=lambda w: len(adj[w]))
            else:
                break
            
            e2 = tuple(sorted((curr, next_v)))
            if e2 in remaining:
                remaining.remove(e2)
                adj[curr].discard(next_v)
                adj[next_v].discard(curr)
                path.append(next_v)
                path_set.add(next_v)
                curr = next_v
            else:
                break

        # Extend from u backwards
        curr = u
        while True:
            next_v = None
            candidates = [w for w in adj[curr] if w not in path_set]
            if candidates:
                next_v = max(candidates, key=lambda w: len(adj[w]))
            else:
                break
            
            e2 = tuple(sorted((curr, next_v)))
            if e2 in remaining:
                remaining.remove(e2)
                adj[curr].discard(next_v)
                adj[next_v].discard(curr)
                path.insert(0, next_v)
                path_set.add(next_v)
                curr = next_v
            else:
                break

        paths.append(path)
        
    return paths

def max_bipartite_matching(adj, n_left, n_right):
    """
    Finds maximum cardinality matching in a bipartite graph.
    adj: dict mapping left_node -> list of right_nodes
    n_left, n_right: number of nodes on each side
    Returns: list of (left, right) pairs
    """
    match_right = [-1] * n_right
    match_left = [-1] * n_left
    
    def dfs(u, visited):
        for v in adj.get(u, []):
            if visited[v]:
                continue
            visited[v] = True
            if match_right[v] == -1 or dfs(match_right[v], visited):
                match_right[v] = u
                match_left[u] = v
                return True
        return False
    
    for u in range(n_left):
        visited = [False] * n_right
        dfs(u, visited)
        
    matches = []
    for u in range(n_left):
        if match_left[u] != -1:
            matches.append((u, match_left[u]))
    return matches

def improve_paths(paths, graph_edges, n, time_limit):
    """
    Iteratively merges paths using maximum matching on endpoints.
    """
    current_paths = [list(p) for p in paths]
    edge_set = set(graph_edges)
    
    # Precompute neighbors for quick lookup
    adj = defaultdict(set)
    for u, v in graph_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    start_time = time.time()
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        num_paths = len(current_paths)
        if num_paths == 0:
            break
            
        # Build auxiliary graph
        # Left side: path i, right endpoint (path[i][-1])
        # Right side: path j, left endpoint (path[j][0])
        # Edge exists if:
        # 1. i != j
        # 2. current end of i is adjacent to current start of j
        # 3. Merging them (removing the connection) does not create a cycle in the vertex set of the merged path.
        #    i.e., start(j) is not in path(i) and end(i) is not in path(j).
        
        aux_adj = defaultdict(list)
        
        for i in range(num_paths):
            end_i = current_paths[i][-1]
            for j in range(num_paths):
                if i == j:
                    continue
                start_j = current_paths[j][0]
                
                # Check adjacency
                if start_j not in adj[end_i]:
                    continue
                
                # Check simplicity violation
                # If start_j is in path i, merging would create a cycle (or revisit).
                # If end_i is in path j, merging would create a cycle (or revisit).
                # Actually, if start_j is in path_i, then the merged path visits start_j twice?
                # Path i: ... -> end_i. Path j: start_j -> ...
                # Merged: ... -> end_i -> start_j -> ...
                # If start_j is in path_i, it appears earlier. So the vertex is visited twice. Invalid.
                # If end_i is in path_j, it appears later. So the vertex is visited twice. Invalid.
                
                if start_j in current_paths[i] or end_i in current_paths[j]:
                    continue
                
                # If valid, add edge i -> j
                aux_adj[i].append(j)
                
        if not aux_adj:
            break
            
        # Find max matching
        matches = max_bipartite_matching(aux_adj, num_paths, num_paths)
        
        if not matches:
            break
            
        # Apply merges
        # We need to be careful: if we merge i->j, we consume endpoints of i and j.
        # A matching ensures each path endpoint is used at most once.
        # So we can apply all matches in the set simultaneously.
        
        used_paths = set()
        new_paths = []
        
        # Create a mapping of which path is the "tail" (i) and which is the "head" (j)
        # match (i, j) means tail of i connects to head of j.
        # New path is path_i + path_j[1:]
        
        merge_map = {}
        for i, j in matches:
            merge_map[i] = j # i is tail, j is head
            used_paths.add(i)
            used_paths.add(j)
            
        # We need to construct the new paths.
        # Since it's a matching, no path is involved in more than one merge as a tail and more than one as a head?
        # Wait, max matching in bipartite graph:
        # Left node i is matched to at most one right node j.
        # Right node j is matched to at most one left node i.
        # So each path index k is involved in at most one merge as a tail (if k in left) and at most one as a head (if k in right).
        # It could be a tail in one merge and a head in another?
        # If k is a tail (k->l) and a head (m->k), then path m -> path k -> path l.
        # This forms a chain of merges.
        
        # Let's trace the chains.
        # tail_map: i -> j (i connects to j)
        # head_map: j -> i (j is connected from i)
        
        tail_map = {}
        head_map = {}
        for i, j in matches:
            tail_map[i] = j
            head_map[j] = i
            
        visited = set()
        
        for k in range(num_paths):
            if k in visited:
                continue
                
            # Find the start of the chain: a path that is a head but not a tail?
            # Or just start at k and follow tail_map.
            # If k is a head (k in head_map), it is part of a merge where it is the second part.
            # We want to find the "root" of the chain: a path that is not a head (no one connects to it).
            
            curr = k
            while curr in head_map:
                curr = head_map[curr]
            
            # Now curr is the start of the chain
            # Traverse the chain
            chain = []
            node = curr
            while True:
                chain.append(node)
                if node in tail_map:
                    node = tail_map[node]
                else:
                    break
            
            # Merge the paths in the chain
            if len(chain) == 1:
                # No merge for this path
                new_paths.append(current_paths[chain[0]])
            else:
                # Concatenate
                merged = []
                for idx, p_idx in enumerate(chain):
                    if idx == 0:
                        merged.extend(current_paths[p_idx])
                    else:
                        # Skip the first vertex of subsequent paths as it's already the last of the previous
                        merged.extend(current_paths[p_idx][1:])
                new_paths.append(merged)
                
            for p_idx in chain:
                visited.add(p_idx)
                
        current_paths = new_paths
        
    return current_paths

def construct():
    deadline = time.time() + 140
    results = []
    
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    # Allocate time per graph
    # We have ~140s. 3 graphs. ~45s each.
    # Initial decomposition is fast. Improvement is the bottleneck.
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        
        # 1. Initial decomposition
        paths = initial_decomposition(n, edges)
        
        # 2. Improve with matching
        # Time budget for improvement
        # Leave 5s for other overheads
        time_left = deadline - time.time()
        # Allocate proportional to graph size? Or equal?
        # Let's just use a fraction of remaining time.
        # To be safe, ensure we finish before deadline.
        # We will run improvement until time is up, but we must return valid results.
        # The improvement function has an internal time limit.
        
        # We want to spend most of the time on the largest graph?
        # Or just split evenly?
        # Let's split evenly among remaining graphs.
        graphs_remaining = specs.index((n,p,seed)) + 1
        # Actually, we are in a loop.
        # Let's just give each graph 1/3 of the total budget minus a small buffer.
        # Total 140s. 3 graphs. 45s each.
        
        # Better: dynamic allocation.
        # But for simplicity, let's just run improvement with a time limit based on remaining time.
        # If we run out of time, we return the current best.
        
        # Let's reserve 10s for the last graph's verification/return overhead?
        # No, verification is O(E).
        
        # Let's just pass a time limit to improve_paths.
        # We'll use 40 seconds per graph.
        # 3 * 40 = 120s. Leaves 20s buffer.
        
        # However, if the first graph is fast, we can give more to the second.
        # Let's just use a fixed 40s per graph for now.
        
        t_start_improve = time.time()
        paths = improve_paths(paths, edges, n, 40.0)
        
        # 3. Verification
        # Ensure validity
        valid = True
        used_edges = set()
        for path in paths:
            if len(path) < 2:
                valid = False
                break
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                e = tuple(sorted((u, v)))
                if e not in set(edges):
                    valid = False
                    break
                if e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            if len(set(path)) != len(path):
                valid = False
                break
            if not valid:
                break
        
        if not valid or len(used_edges) != len(edges):
            # Fallback to initial decomposition (which is valid by construction)
            paths = initial_decomposition(n, edges)
            
        results.append(paths)
        
        # Check time
        if time.time() > deadline - 5:
            # If we are running out of time, just break and return what we have?
            # But we need 3 results.
            # If we haven't processed all, we must process them quickly.
            pass

    return results

if __name__ == "__main__":
    construct()
