import time
import random
from collections import deque

def _generate_trees(n=20, seed=611):
    rng = random.Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def _tree_adjacency(tree_edges, k):
    adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _find_embedding(tree_edges, k, used_edges, deadline):
    if not tree_edges:
        return list(range(k)) if k == 1 else None
    
    adj = _tree_adjacency(tree_edges, k)
    used_set = set()
    for u, v in used_edges:
        used_set.add((u, v) if u < v else (v, u))
    
    # Find a good root: vertex with highest degree in the tree
    root = max(range(k), key=lambda v: len(adj[v]))
    
    # BFS order from root
    order = [root]
    parent = [-1] * k
    parent[root] = root
    visited = [False] * k
    visited[root] = True
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                order.append(v)
                queue.append(v)
    
    # Backtracking search
    assignment = [-1] * k
    assignment[root] = -1  # will be set during search
    
    def backtrack(idx, available):
        if time.time() > deadline:
            return None
        if idx == k:
            return assignment.copy()
        
        tree_v = order[idx]
        if idx == 0:
            # Root can be any vertex
            candidates = list(available)
            random.shuffle(candidates)
            for host_v in candidates:
                assignment[tree_v] = host_v
                new_available = available - {host_v}
                result = backtrack(idx + 1, new_available)
                if result is not None:
                    return result
            return None
        else:
            # Non-root: must be adjacent to parent's host vertex
            parent_tree_v = parent[tree_v]
            parent_host = assignment[parent_tree_v]
            # Candidates must be adjacent to parent_host and in available
            candidates = [v for v in available if v in adj_to[parent_host]]
            # Sort by degree in remaining graph (most constrained first)
            candidates.sort(key=lambda v: len([x for x in available if x != v and x in adj_to[v]]))
            random.shuffle(candidates)  # slight randomization for diversity
            
            for host_v in candidates:
                # Check edge is not used
                edge_key = (min(parent_host, host_v), max(parent_host, host_v))
                if edge_key in used_set:
                    continue
                
                assignment[tree_v] = host_v
                new_available = available - {host_v}
                result = backtrack(idx + 1, new_available)
                if result is not None:
                    return result
            return None
    
    # Build adjacency sets for K20
    adj_to = [set() for _ in range(20)]
    for i in range(20):
        for j in range(20):
            if i != j:
                adj_to[i].add(j)
    
    available = set(range(20))
    result = backtrack(0, available)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s budget
    
    trees = _generate_trees(20, 611)
    
    # Sort trees by size descending (most constrained first)
    # But we need to return them in original order
    order = list(range(20))
    order.sort(key=lambda i: len(trees[i]), reverse=True)
    
    embeddings = [None] * 20
    used_edges = set()
    
    # Try multiple random seeds for diversity
    best_score = 0
    best_embeddings = [None] * 20
    
    for seed in range(100):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        cur_embeddings = [None] * 20
        cur_used = set()
        
        success = True
        for idx in order:
            k = idx + 1
            tree_edges = trees[idx]
            
            if k == 1:
                # Single vertex, any unused vertex works
                used_vertices = set()
                for emb in cur_embeddings:
                    if emb is not None:
                        used_vertices.update(emb)
                available = [v for v in range(20) if v not in used_vertices]
                if not available:
                    success = False
                    break
                cur_embeddings[idx] = [available[0]]
                continue
            
            if k == 2:
                # Just an edge
                # Find an unused edge
                found = False
                for u in range(20):
                    for v in range(u+1, 20):
                        if (u, v) not in cur_used:
                            cur_embeddings[idx] = [u, v]
                            cur_used.add((u, v))
                            found = True
                            break
                    if found:
                        break
                if not found:
                    success = False
                    break
                continue
            
            # Backtracking search
            emb = _find_embedding(tree_edges, k, cur_used, deadline)
            if emb is None:
                success = False
                break
            
            cur_embeddings[idx] = emb
            for u, v in tree_edges:
                host_u = emb[u]
                host_v = emb[v]
                edge_key = (min(host_u, host_v), max(host_u, host_v))
                if edge_key in cur_used:
                    success = False
                    break
                cur_used.add(edge_key)
            if not success:
                break
        
        if success:
            score = sum(len(trees[i]) for i in range(20))
            if score > best_score:
                best_score = score
                best_embeddings = cur_embeddings.copy()
    
    # If no full solution found, try partial packing
    if best_score == 0:
        # Try to pack as many as possible
        for seed in range(50):
            if time.time() > deadline:
                break
            
            random.seed(seed + 1000)
            cur_embeddings = [None] * 20
            cur_used = set()
            
            for idx in order:
                k = idx + 1
                tree_edges = trees[idx]
                
                if k == 1:
                    used_vertices = set()
                    for emb in cur_embeddings:
                        if emb is not None:
                            used_vertices.update(emb)
                    available = [v for v in range(20) if v not in used_vertices]
                    if not available:
                        continue
                    cur_embeddings[idx] = [available[0]]
                    continue
                
                if k == 2:
                    found = False
                    for u in range(20):
                        for v in range(u+1, 20):
                            if (u, v) not in cur_used:
                                cur_embeddings[idx] = [u, v]
                                cur_used.add((u, v))
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        continue
                    continue
                
                emb = _find_embedding(tree_edges, k, cur_used, deadline)
                if emb is not None:
                    cur_embeddings[idx] = emb
                    for u, v in tree_edges:
                        host_u = emb[u]
                        host_v = emb[v]
                        edge_key = (min(host_u, host_v), max(host_u, host_v))
                        if edge_key not in cur_used:
                            cur_used.add(edge_key)
            
            score = sum(len(trees[i]) for i in range(20) if cur_embeddings[i] is not None)
            if score > best_score:
                best_score = score
                best_embeddings = cur_embeddings.copy()
    
    # Validate and convert to proper types
    result = []
    for i in range(20):
        if best_embeddings[i] is None:
            result.append(None)
        else:
            # Convert to list of Python ints
            result.append([int(v) for v in best_embeddings[i]])
    
    return result
