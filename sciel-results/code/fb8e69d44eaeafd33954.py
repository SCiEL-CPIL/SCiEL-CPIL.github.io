import time
import random
from random import Random

def instance_trees(n=20, seed=611):
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

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    trees = instance_trees()
    N_HOSTS = 20
    
    # Pre-compute edges for each tree
    tree_edges = []
    for i in range(20):
        tree_edges.append(trees[i])
        
    # Build adjacency list for each tree
    tree_adj = []
    for i in range(20):
        k = i + 1
        adj = [[] for _ in range(k)]
        for u, v in tree_edges[i]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
        
    # Find leaves for each tree
    tree_leaves = []
    for i in range(20):
        k = i + 1
        if k == 1:
            tree_leaves.append([0])
        else:
            leaves = [v for v in range(k) if len(tree_adj[i][v]) == 1]
            tree_leaves.append(leaves)

    def backtrack_embed(tree_idx, used_edges, mapping):
        """
        Embed tree_idx into K20 using backtracking.
        used_edges: set of (min(u,v), max(u,v)) already used by other trees.
        mapping: dict tree_vertex -> host_vertex
        Returns mapping dict if successful, None otherwise.
        """
        k = tree_idx + 1
        if k == 1:
            # T1: just pick any host
            for h in range(N_HOSTS):
                return {0: h}
            return None
        
        # Start with a leaf
        start_leaf = tree_leaves[tree_idx][0]
        parent_of = {}
        
        # We will build the embedding incrementally.
        # Start with the start_leaf mapped to some host.
        # Then expand.
        
        # To make it efficient, we can try to fix the root (start_leaf) to a specific host?
        # No, we need to search.
        # However, since K20 is symmetric, we can fix start_leaf to 0 WLOG?
        # Yes! K20 is vertex-transitive. We can fix one vertex to 0.
        # This reduces the search space by factor 20.
        
        mapping = {start_leaf: 0}
        
        # Now we need to embed the rest of the tree.
        # We can use a stack or queue of "pending" vertices to embed.
        # A vertex is pending if it is in the tree but not yet mapped, and has at least one mapped neighbor.
        
        # Let's use a simple recursive backtracking.
        # State: mapping dict.
        # We pick a vertex v in the tree that is unmapped but has a mapped neighbor.
        # We try all possible host vertices for v, consistent with used_edges and existing mapping.
        
        def dfs():
            # Find a candidate vertex to map
            candidate = None
            for v in range(k):
                if v not in mapping:
                    # Check if it has a mapped neighbor
                    has_mapped_neighbor = False
                    for u in tree_adj[tree_idx][v]:
                        if u in mapping:
                            has_mapped_neighbor = True
                            break
                    if has_mapped_neighbor:
                        candidate = v
                        break
            
            if candidate is None:
                # All vertices mapped
                return True
            
            v = candidate
            # Find a mapped neighbor
            mapped_neighbors = [u for u in tree_adj[tree_idx][v] if u in mapping]
            
            # Try all possible host vertices for v
            for h in range(N_HOSTS):
                # Check if h is already used in mapping
                if h in mapping.values():
                    continue
                
                # Check if edges to mapped neighbors are valid
                valid = True
                for u in mapped_neighbors:
                    a = mapping[u]
                    b = h
                    e = (min(a, b), max(a, b))
                    if e in used_edges:
                        valid = False
                        break
                    # Also, we must ensure that this edge is not used by another edge in the SAME tree?
                    # No, the tree is a tree, so no cycles. And mapping is injective, so no self-loops.
                    # But we must ensure that we don't create a conflict with edges ALREADY ASSIGNED in this tree?
                    # Wait, we are building the mapping. The edges of this tree are determined by the mapping.
                    # We only need to check against `used_edges` (from other trees).
                    # But what if two edges in THIS tree map to the same host edge?
                    # That would mean a cycle in the tree, which is impossible.
                    # Or it means the mapping is not injective? No, we check h not in mapping.values().
                    # So edges within the tree are distinct host edges.
                
                if not valid:
                    continue
                
                # Check if the new host edges are disjoint from each other?
                # Yes, as argued.
                
                # Apply
                mapping[v] = h
                new_edges = set()
                for u in mapped_neighbors:
                    a = mapping[u]
                    b = h
                    new_edges.add((min(a, b), max(a, b)))
                
                # Recurse
                if dfs():
                    return True
                
                # Undo
                del mapping[v]
            
            return False
        
        if dfs():
            return mapping
        return None

    # Strategy:
    # 1. Sort trees by size (descending) to embed the most constrained ones first?
    #    Or ascending?
    #    Usually, larger trees are harder to embed if the graph gets crowded.
    #    But K20 is dense.
    #    Let's try embedding in order T20, T19, ..., T1.
    #    Or T1..T20?
    #    The inspiration says "Backtracking with BFS ordering from a leaf systematically finds valid embeddings".
    #    Let's try descending order of edges.
    
    order = list(range(19, -1, -1)) # T20, T19, ..., T1
    
    used_edges = set()
    final_mappings = [None] * 20
    
    # We might need to restart if we get stuck?
    # Backtracking is exact. If it returns None, no embedding exists for that tree given the current used_edges.
    # But maybe a different order or different host assignment for previous trees would work.
    # Since we fixed T20's root to 0, we lost some generality? No, symmetry handles it.
    # But what if T20 cannot be embedded with root 0?
    # Actually, any embedding of T20 can be rotated so that its "root" (the leaf we chose) is at 0.
    # So fixing root to 0 is WLOG.
    
    # However, if backtracking fails, it means no embedding exists for T20?
    # No, it means no embedding exists for T20 with root 0 AND disjoint from used_edges (which is empty).
    # So T20 is always embeddable in K20.
    
    # The problem arises when embedding T19. It must be disjoint from T20's edges.
    # If backtracking fails for T19, it means T19 cannot be embedded disjoint from T20's specific embedding.
    # But maybe a DIFFERENT embedding of T20 would allow T19.
    
    # So we need to search over embeddings of the previous trees?
    # That's exponential.
    
    # Alternative: Randomized Restart.
    # Try multiple random embeddings for the large trees?
    # Or use the SA approach but with a strong local search?
    
    # Let's stick to the deterministic backtracking but add a fallback:
    # If we fail, try a different order?
    # Or try randomizing the "root" host? (We fixed it to 0, but maybe we should try all 20?)
    # No, symmetry says 0 is WLOG.
    
    # What if we randomize the ORDER of vertices to try in the backtracking?
    # The current code tries h in 0..19.
    # We can shuffle this order.
    
    # Let's implement a randomized backtracking with restarts.
    
    success = False
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        # Shuffle the order of host vertices to try in backtracking
        # We can't easily modify the global N_HOSTS loop.
        # Let's modify backtrack_embed to accept a shuffled list of hosts.
        
        pass # Placeholder

    # Let's rewrite backtrack_embed to be more flexible.
    
    def backtrack_embed_random(tree_idx, used_edges, host_order):
        k = tree_idx + 1
        if k == 1:
            return {0: host_order[0]}
        
        start_leaf = tree_leaves[tree_idx][0]
        mapping = {start_leaf: host_order[0]} # Fix to first in order (WLOG 0 if sorted)
        
        def dfs():
            candidate = None
            for v in range(k):
                if v not in mapping:
                    has_mapped_neighbor = False
                    for u in tree_adj[tree_idx][v]:
                        if u in mapping:
                            has_mapped_neighbor = True
                            break
                    if has_mapped_neighbor:
                        candidate = v
                        break
            
            if candidate is None:
                return True
            
            v = candidate
            mapped_neighbors = [u for u in tree_adj[tree_idx][v] if u in mapping]
            
            for h in host_order:
                if h in mapping.values():
                    continue
                
                valid = True
                for u in mapped_neighbors:
                    a = mapping[u]
                    b = h
                    e = (min(a, b), max(a, b))
                    if e in used_edges:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                mapping[v] = h
                if dfs():
                    return True
                del mapping[v]
            
            return False
        
        if dfs():
            return mapping
        return None

    # Try to pack all trees
    # Order: T20, T19, ..., T1
    # Use random host order for each tree?
    # Actually, fixing the root to 0 is WLOG for the FIRST tree.
    # For subsequent trees, we cannot fix the root to 0 because 0 is already taken by the first tree?
    # No, host vertices can be reused across different trees?
    # YES. "Embedded trees must be edge-disjoint". Vertex disjointness is NOT required.
    # So multiple trees can share host vertices.
    # This makes the problem much easier!
    
    # So we can fix the root of EVERY tree to 0?
    # No, if T20 uses edge (0,1), T19 cannot use edge (0,1).
    # But T19 CAN use vertex 0, as long as it doesn't use the edge (0,1) if T20 does.
    # So fixing root to 0 for all trees is still WLOG?
    # Yes, because the graph is vertex-transitive.
    # If there is a solution where T20's root is at 5 and T19's root is at 3,
    # we can rotate the whole graph so T20's root is at 0.
    # But then T19's root moves to some other vertex.
    # So we can't fix ALL roots to 0 simultaneously in a single coordinate system?
    # Wait. The "rotation" is a global permutation of host vertices.
    # If we fix T20's root to 0, we have used up the symmetry.
    # For T19, we cannot assume its root is 0.
    # So for T19, we must search over all 20 possible host vertices for its root.
    
    # Revised strategy:
    # 1. Embed T20 with root fixed to 0.
    # 2. Embed T19 with root tried at all 20 hosts.
    # 3. Embed T18 with root tried at all 20 hosts.
    # ...
    
    # This increases the branching factor for the root, but the rest of the tree is determined by backtracking.
    
    def embed_tree(tree_idx, used_edges):
        k = tree_idx + 1
        if k == 1:
            # T1: any host
            return {0: 0}
            
        start_leaf = tree_leaves[tree_idx][0]
        
        for root_host in range(N_HOSTS):
            mapping = {start_leaf: root_host}
            
            def dfs():
                candidate = None
                for v in range(k):
                    if v not in mapping:
                        has_mapped_neighbor = False
                        for u in tree_adj[tree_idx][v]:
                            if u in mapping:
                                has_mapped_neighbor = True
                                break
                        if has_mapped_neighbor:
                            candidate = v
                            break
                
                if candidate is None:
                    return True
                
                v = candidate
                mapped_neighbors = [u for u in tree_adj[tree_idx][v] if u in mapping]
                
                for h in range(N_HOSTS):
                    if h in mapping.values():
                        continue
                    
                    valid = True
                    for u in mapped_neighbors:
                        a = mapping[u]
                        b = h
                        e = (min(a, b), max(a, b))
                        if e in used_edges:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    mapping[v] = h
                    if dfs():
                        return True
                    del mapping[v]
                
                return False
            
            if dfs():
                return mapping
        
        return None

    # Try to embed all trees
    # Order: T20, T19, ..., T1
    used_edges = set()
    final_mappings = [None] * 20
    
    for idx in range(19, -1, -1):
        if time.time() > deadline:
            break
        m = embed_tree(idx, used_edges)
        if m is None:
            # Failed to embed this tree
            # Try to drop it? Or restart?
            # Let's try to drop it and continue with the rest.
            # But we want to maximize edges.
            # So we should try to embed the largest ones first.
            # If T20 fails, we can't do anything. (But T20 never fails).
            # If T19 fails, maybe we can drop T19 and embed T18..T1?
            # Let's just break and handle the rest in a fallback.
            break
        else:
            final_mappings[idx] = m
            # Add edges to used_edges
            for u, v in tree_edges[idx]:
                a = m[u]
                b = m[v]
                used_edges.add((min(a, b), max(a, b)))

    # Check if we embedded all
    if all(final_mappings[i] is not None for i in range(20)):
        result = [list(final_mappings[i].values()) for i in range(20)]
        # Wait, the result format is:
        # Entry k-1 is a list of k distinct int host vertices.
        # Position i is the image of vertex i of the fixed tree T_k.
        # So result[i] should be [mapping[0], mapping[1], ..., mapping[k-1]]
        
        result = []
        for i in range(20):
            k = i + 1
            m = final_mappings[i]
            result.append([m[j] for j in range(k)])
        
        return result

    # Fallback: If we didn't embed all, try to embed the remaining ones?
    # Or use the SA fallback from the original code?
    # Let's try to embed the remaining trees in ascending order?
    # No, let's just try to fill in the missing ones with a simple greedy or SA.
    
    # For now, let's try to embed the missing trees one by one, skipping if they fail.
    for idx in range(20):
        if final_mappings[idx] is None:
            if time.time() > deadline:
                break
            m = embed_tree(idx, used_edges)
            if m is not None:
                final_mappings[idx] = m
                for u, v in tree_edges[idx]:
                    a = m[u]
                    b = m[v]
                    used_edges.add((min(a, b), max(a, b)))

    # Construct result
    result = [None] * 20
    for i in range(20):
        if final_mappings[i] is not None:
            k = i + 1
            m = final_mappings[i]
            result[i] = [int(m[j]) for j in range(k)]
        else:
            result[i] = None
            
    return result

# Test locally
if __name__ == "__main__":
    res = construct()
    # Check validity
    used = set()
    total_edges = 0
    trees = instance_trees()
    for i in range(20):
        if res[i] is None:
            continue
        mapping = res[i]
        # Check distinctness
        if len(set(mapping)) != len(mapping):
            print(f"Tree {i} not distinct")
        for u, v in trees[i]:
            a = mapping[u]
            b = mapping[v]
            e = (min(a, b), max(a, b))
            if e in used:
                print(f"Conflict at tree {i}, edge {e}")
            used.add(e)
            total_edges += 1
    print(f"Total edges: {total_edges}")
    print(f"Score: {total_edges / 611}")
