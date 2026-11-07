import time
import sys
from random import Random
from collections import deque

# Increase recursion limit for deep backtracking
sys.setrecursionlimit(10000)

def _instance_trees(n=20, seed=611):
    rng = Random(seed)
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

class EdgePacker:
    def __init__(self):
        self.n = 20
        self.trees = _instance_trees(20, 611)
        # Precompute adjacency bitmasks for K20
        # adj_mask[h] is a bitmask of neighbors of h in K20
        self.adj_mask = [0] * self.n
        for u in range(self.n):
            for v in range(self.n):
                if u != v:
                    self.adj_mask[u] |= (1 << v)
        
        # used_edges_mask is a list of 20 ints, where used_edges_mask[u] has bit v set if (u,v) is used
        self.used_edges_mask = [0] * self.n
        self.used_vertices_mask = 0

    def reset(self):
        self.used_edges_mask = [0] * self.n
        self.used_vertices_mask = 0

    def is_edge_free(self, u, v):
        return (self.used_edges_mask[u] & (1 << v)) == 0

    def mark_edge_used(self, u, v):
        self.used_edges_mask[u] |= (1 << v)
        self.used_edges_mask[v] |= (1 << u)

    def unmark_edge_used(self, u, v):
        self.used_edges_mask[u] &= ~(1 << v)
        self.used_edges_mask[v] &= ~(1 << u)

    def is_vertex_free(self, v):
        return (self.used_vertices_mask & (1 << v)) == 0

    def mark_vertex_used(self, v):
        self.used_vertices_mask |= (1 << v)

    def unmark_vertex_used(self, v):
        self.used_vertices_mask &= ~(1 << v)

    def get_available_neighbors(self, host_u, exclude_mask=0):
        """
        Returns a list of host vertices v such that:
        1. v is not host_u
        2. edge (host_u, v) is free
        3. v is not in exclude_mask (typically used vertices)
        """
        # neighbors of host_u in K20 that are not used
        # avail = adj_mask[host_u] & ~used_edges_mask[host_u] & ~used_vertices_mask
        avail = self.adj_mask[host_u] & ~self.used_edges_mask[host_u] & ~self.used_vertices_mask
        # exclude any vertices already assigned in this tree (handled by used_vertices_mask during tree embedding)
        # Actually, used_vertices_mask is global. We need to track local assignments too.
        # For simplicity in the fast path, we rely on the global state.
        # But within a single tree embedding, we need to ensure distinct vertices.
        # The standard backtracking handles this by checking a local 'local_used' mask.
        
        # This function is for the global state. The local check is done in the DFS.
        return avail

    def embed_tree(self, tree_edges, k, rng, deadline, local_depth_limit=5000):
        """
        Embeds a tree with k vertices into the available graph.
        Returns a list of host vertices or None.
        Uses randomized DFS with backtracking.
        """
        if k == 1:
            # Find any free vertex
            free_verts = [v for v in range(self.n) if self.is_vertex_free(v)]
            if not free_verts:
                return None
            v = rng.choice(free_verts)
            return [v]
        
        if k == 2:
            # Find any free edge
            for u in range(self.n):
                if not self.is_vertex_free(u):
                    continue
                for v in range(u + 1, self.n):
                    if not self.is_vertex_free(v):
                        continue
                    if self.is_edge_free(u, v):
                        return [u, v]
            return None

        # Build tree adjacency
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Compute degrees
        deg = [len(adj[v]) for v in range(k)]
        
        # Root selection: highest degree first to constrain early
        roots = sorted(range(k), key=lambda x: -deg[x])
        rng.shuffle(roots)
        
        # We will try a few random roots
        for root in roots[:3]: # Limit to 3 best roots to save time
            if time.time() > deadline:
                return None
            
            # BFS order
            visited = [False] * k
            order = []
            parent = [-1] * k
            queue = deque([root])
            visited[root] = True
            while queue:
                cur = queue.popleft()
                order.append(cur)
                # Shuffle neighbors to add randomness to the order
                nbrs = adj[cur][:]
                rng.shuffle(nbrs)
                for nb in nbrs:
                    if not visited[nb]:
                        visited[nb] = True
                        parent[nb] = cur
                        queue.append(nb)
            
            # Local state for this embedding attempt
            assign = [-1] * k
            local_used_verts = 0 # bitmask of vertices used in this tree
            
            # Pre-compute valid host candidates for each tree vertex given the parent?
            # No, dynamic.
            
            def dfs(idx):
                nonlocal local_used_verts
                if time.time() > deadline:
                    return False
                
                if idx == len(order):
                    return True
                
                tv = order[idx]
                
                if tv == root:
                    # Candidate hosts: any free global vertex
                    candidates = []
                    for h in range(self.n):
                        if not self.is_vertex_free(h):
                            continue
                        # Prefer hosts with high remaining degree in the *global* unused graph?
                        # Or just random
                        candidates.append(h)
                    
                    if not candidates:
                        return False
                    
                    rng.shuffle(candidates)
                    
                    for h in candidates:
                        assign[tv] = h
                        local_used_verts |= (1 << h)
                        self.mark_vertex_used(h) # Temporarily mark globally? 
                        # Actually, it's better to keep global state clean and just check against local_used_verts + global_used
                        # But the "is_vertex_free" checks global. We need to ensure h is not in local_used_verts.
                        # Since we just picked h from free global vertices, it's not in global_used.
                        # We must ensure it's not in local_used_verts.
                        # The loop above doesn't check local_used_verts. Let's fix that.
                        self.unmark_vertex_used(h) # Undo global mark if we did it, we didn't.
                        
                        # Correct logic:
                        # 1. h must be free globally.
                        # 2. h must not be in local_used_verts.
                        # Since we iterate over free global vertices, condition 1 is met.
                        # We need to check condition 2.
                        
                        # Let's re-structure:
                        pass
                    
                    # Re-implementation of root candidate loop with local check
                    for h in range(self.n):
                        if self.is_vertex_free(h) and not (local_used_verts & (1 << h)):
                            assign[tv] = h
                            local_used_verts |= (1 << h)
                            
                            if dfs(idx + 1):
                                return True
                            
                            local_used_verts &= ~(1 << h)
                            assign[tv] = -1
                    return False
                else:
                    p = parent[tv]
                    ph = assign[p]
                    if ph == -1:
                        return False
                    
                    # Candidate hosts:
                    # 1. Free globally
                    # 2. Not in local_used_verts
                    # 3. Edge (ph, h) is free globally
                    candidates = []
                    for h in range(self.n):
                        if not self.is_vertex_free(h):
                            continue
                        if local_used_verts & (1 << h):
                            continue
                        if not self.is_edge_free(ph, h):
                            continue
                        candidates.append(h)
                    
                    if not candidates:
                        return False
                    
                    # Heuristic: prefer h with more free edges to other potential future nodes?
                    # Or just shuffle
                    rng.shuffle(candidates)
                    
                    for h in candidates:
                        assign[tv] = h
                        local_used_verts |= (1 << h)
                        
                        if dfs(idx + 1):
                            return True
                        
                        local_used_verts &= ~(1 << h)
                        assign[tv] = -1
                    return False

            if dfs(0):
                # Commit the embedding to global state
                for i in range(k):
                    h = assign[i]
                    self.mark_vertex_used(h)
                for u, v in tree_edges:
                    hu, hv = assign[u], assign[v]
                    self.mark_edge_used(hu, hv)
                return assign
            else:
                # Rollback local state (already handled by backtracking returning False without committing)
                # But we might have partially marked global state if we committed? No, we only commit on success.
                # However, the `mark_vertex_used` inside the loop was NOT done. We only used local_used_verts.
                # So global state is clean.
                pass

        return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    trees = _instance_trees(20, 611)
    
    # Order: Largest to smallest
    order = list(range(19, -1, -1)) # k=20 down to k=1
    
    best_result = [None] * 20
    best_value = 0
    
    # Multiple trials with different random seeds
    for trial in range(5000):
        if time.time() > deadline:
            break
        
        rng = Random(trial * 104729 + 12345)
        packer = EdgePacker()
        result = [None] * 20
        total_edges = 0
        success = True
        
        for k in order:
            if time.time() > deadline:
                success = False
                break
            
            tree_edges = trees[k - 1]
            
            if not tree_edges:
                # T1
                free_verts = [v for v in range(20) if packer.is_vertex_free(v)]
                if not free_verts:
                    success = False
                    break
                v = rng.choice(free_verts)
                packer.mark_vertex_used(v)
                result[k - 1] = [v]
                continue
            
            embedding = packer.embed_tree(tree_edges, k, rng, deadline)
            
            if embedding is None:
                success = False
                break
            
            result[k - 1] = embedding
            total_edges += len(tree_edges)
            
            # Early termination if we have packed all edges
            if total_edges == 190:
                break
        
        if success and total_edges > best_value:
            best_value = total_edges
            best_result = result[:]
            if best_value == 190:
                break
    
    # Ensure correct types
    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    
    return final
