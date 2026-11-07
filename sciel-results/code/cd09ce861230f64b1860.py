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
    deadline = start_time + 115.0  # Safety margin
    
    trees = instance_trees()
    n_trees = len(trees)
    n_vertices = 20
    
    # Pre-compute tree edges
    tree_edges = []
    for t in trees:
        tree_edges.append(t)
        
    # State:
    # mapping[i] is a list of length len(trees[i]) mapping tree vertex i to host vertex 0..19
    # used_edges is a set of frozensets or sorted tuples of (u, v) for u < v
    
    best_result = [None] * n_trees
    best_value = 0
    
    # Heuristic: Embed smaller trees first? Or larger?
    # Smaller trees have fewer constraints. Let's try embedding in order 0..19.
    # Actually, T1 is empty, T2 is 1 edge.
    
    # To allow randomization in search, we can shuffle the order of trees or the vertex mappings.
    # However, the output must be in the original order.
    
    # We will use a recursive backtracking solver.
    # To make it fast, we can try to fix some mappings if possible, or use random restarts with limited depth.
    # Given 120s, we can afford a significant search.
    
    # A better strategy:
    # 1. Try to find a packing that covers all edges (190 edges).
    # 2. If not, maximize edges.
    
    # Since K20 has 190 edges, and sum of edges in trees is 190, a perfect packing uses all edges.
    # This is a decomposition of K20 into these specific trees.
    
    # Algorithm:
    # Backtrack on trees. For each tree, try to assign its vertices to host vertices such that
    # no edge is reused.
    
    # To speed up, we can use a "most constrained variable" heuristic if we were doing full CSP,
    # but here the structure is fixed. We just need to find a bijection for each tree's vertices
    # (subset of 0..19) such that edges don't collide.
    
    # Note: The trees are not necessarily spanning. T_k has k vertices.
    # So we are choosing a subset of size k from 0..19 for tree k.
    
    # Let's implement a randomized backtracking search.
    
    def solve(idx, used_edges, current_mappings, current_value):
        nonlocal best_result, best_value
        
        if time.time() > deadline:
            return
        
        if idx == n_trees:
            if current_value > best_value:
                best_value = current_value
                best_result = [m[:] if m else None for m in current_mappings]
            return
            
        # Pruning: if even with perfect packing of remaining trees we can't beat best_value, return
        # Max remaining edges = sum(len(t) for t in tree_edges[idx:])
        max_remaining = sum(len(t) for t in tree_edges[idx:])
        if current_value + max_remaining <= best_value:
            return
            
        tree = tree_edges[idx]
        k = len(tree) + 1
        
        # Generate possible embeddings for this tree
        # We need to map tree vertices 0..k-1 to distinct host vertices 0..19
        # such that for each edge (u, v) in tree, (host[u], host[v]) is not in used_edges
        
        # To reduce search space, we can fix the first vertex of the tree to a specific host vertex?
        # No, symmetry breaking is hard here without knowing the structure.
        
        # However, we can iterate through permutations of host vertices for the tree's vertices.
        # k can be up to 20. 20! is too big.
        
        # We need a smarter approach.
        # Let's use a recursive assignment for the tree vertices.
        
        # Order tree vertices by degree (highest first) to constrain early?
        # Or just 0..k-1.
        
        # Let's try a DFS for the tree embedding.
        # State: tree_vertex -> host_vertex mapping (partial)
        # We can pre-calculate the degrees of tree vertices to order them.
        
        # Actually, a simpler heuristic:
        # Try random mappings for the tree, and check validity.
        # If k is small, this is fast. If k is large, it's slow.
        
        # Let's combine:
        # For small k (<= 6), try all permutations of available vertices? No, available vertices depend on previous choices.
        
        # Better:
        # For each tree, pick a random set of k host vertices from 0..19.
        # Then try to find a bijection from tree vertices to these host vertices that is edge-disjoint.
        
        # This reduces the problem to finding a valid labeling of a fixed set of host vertices.
        
        # Let's implement a helper to check if a tree can be embedded into a specific set of host vertices
        # given the used_edges.
        
        def can_embed_tree(tree_idx, host_set):
            # Returns a mapping (list of host vertices for tree vertices) or None
            # Uses backtracking on tree vertices
            
            t_edges = tree_edges[tree_idx]
            k = len(t_edges) + 1
            host_list = list(host_set)
            
            # Order tree vertices by degree (descending) to constrain early
            deg = [0] * k
            for u, v in t_edges:
                deg[u] += 1
                deg[v] += 1
            order = sorted(range(k), key=lambda x: -deg[x])
            
            mapping = [-1] * k
            assigned = [False] * len(host_list)
            
            def dfs_tree(pos):
                if time.time() > deadline:
                    return None
                if pos == len(order):
                    # Check all edges
                    for u, v in t_edges:
                        hu = mapping[u]
                        hv = mapping[v]
                        edge = (min(hu, hv), max(hu, hv))
                        if edge in used_edges:
                            return None
                    return mapping[:]
                
                tv = order[pos]
                # Try each available host vertex
                # Shuffle to randomize
                indices = list(range(len(host_list)))
                random.shuffle(indices)
                
                for i in indices:
                    if assigned[i]:
                        continue
                    hv = host_list[i]
                    mapping[tv] = hv
                    assigned[i] = True
                    
                    # Check conflicts with already assigned neighbors
                    valid = True
                    for u, v in t_edges:
                        if u == tv or v == tv:
                            other = v if u == tv else u
                            if mapping[other] != -1:
                                ho = mapping[other]
                                edge = (min(hv, ho), max(hv, ho))
                                if edge in used_edges:
                                    valid = False
                                    break
                                # Also check if this edge is already used by another part of THIS tree?
                                # No, tree has no cycles, so no internal conflict if we just check used_edges.
                                # Wait, we must ensure that the edge (hv, ho) is not used by *other* trees (in used_edges)
                                # AND that we don't reuse an edge within the same tree (impossible for a tree).
                                # But we must also ensure that we don't assign two tree vertices to the same host vertex.
                                # That is handled by 'assigned'.
                    
                    if valid:
                        res = dfs_tree(pos + 1)
                        if res is not None:
                            return res
                    
                    mapping[tv] = -1
                    assigned[i] = False
                return None
            
            return dfs_tree(0)

        # Strategy for main solver:
        # Iterate through random subsets of host vertices of size k for the current tree.
        # This is expensive if we do it naively.
        
        # Alternative:
        # Just pick a random permutation of 0..19 and try to map the tree vertices to the first k available?
        # No, the tree vertices are specific.
        
        # Let's try a simpler random restart approach:
        # 1. Pick a random permutation of 0..19.
        # 2. For tree 1, take the first k1 vertices from the permutation that don't conflict?
        #    This is tricky because "conflict" depends on the specific mapping.
        
        # Let's go back to the DFS on trees, but optimize the candidate generation.
        # For tree idx, we need to choose k distinct host vertices.
        # We can generate random combinations of k host vertices.
        # Number of combinations C(20, k) can be large for k=10 (184756).
        # But we can just try a few random ones.
        
        # To be efficient:
        # We will perform multiple random restarts.
        # In each restart, we attempt to pack all trees.
        # For each tree, we try a limited number of random host sets and mappings.
        
        # Actually, the previous `can_embed_tree` is good if we give it a host set.
        # How to choose host sets?
        # We can maintain a pool of available vertices? No, vertices can be reused for different trees, just not edges.
        # So the "host set" for tree i is just any subset of 0..19 of size k_i.
        
        # Let's modify the main loop to try random host sets for the current tree.
        
        def backtrack(idx, used_edges, current_mappings, current_value):
            nonlocal best_result, best_value
            
            if time.time() > deadline:
                return
            
            if idx == n_trees:
                if current_value > best_value:
                    best_value = current_value
                    best_result = [m[:] if m else None for m in current_mappings]
                return
                
            # Pruning
            max_remaining = sum(len(t) for t in tree_edges[idx:])
            if current_value + max_remaining <= best_value:
                return
            
            tree = tree_edges[idx]
            k = len(tree) + 1
            
            # Try to find a valid embedding for this tree
            # We will try a few random host sets
            # If k is small, we can be more exhaustive.
            
            # To increase diversity, we shuffle the order of trees in some restarts?
            # No, let's keep order fixed but randomize the search inside.
            
            # Generate random permutations of 0..19
            # For each permutation, try to assign the first k available? No.
            # Just pick k random distinct vertices.
            
            attempts = 0
            max_attempts = 50  # Limit per tree to avoid getting stuck
            
            while attempts < max_attempts:
                attempts += 1
                if time.time() > deadline:
                    return
                
                # Pick random k vertices
                host_set = random.sample(range(n_vertices), k)
                
                mapping = can_embed_tree(idx, host_set)
                
                if mapping is not None:
                    # Check edges again to be safe (can_embed_tree should have done it)
                    new_used = set(used_edges)
                    for u, v in tree:
                        hu = mapping[u]
                        hv = mapping[v]
                        edge = (min(hu, hv), max(hu, hv))
                        new_used.add(edge)
                    
                    # Recurse
                    current_mappings.append(mapping)
                    backtrack(idx + 1, new_used, current_mappings, current_value + len(tree))
                    current_mappings.pop()
                    
                    # If we found a solution, we might want to continue to find better ones?
                    # The problem asks for max edges. If we found 190, we can stop.
                    if best_value == 190:
                        return
                    # Otherwise, we continue to try other mappings for this tree?
                    # Yes, to find a better packing.
                    # But this can be very slow.
                    # Let's limit the number of successful mappings we explore per tree.
                    # Or just return after the first success if we are close to optimal?
                    # No, let's try a few.
                    
                    # Break after first success to keep it simple and fast?
                    # If we break, we might miss a better global packing.
                    # But with random restarts, we might find it later.
                    # Let's try to explore a bit more if time permits.
                    # For now, let's just return after the first valid embedding found in this restart path.
                    # This makes it a "first fit" with randomization.
                    return
            
            # If no mapping found in max_attempts, we fail this branch
            return

        # Run multiple restarts
        num_restarts = 1000
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Randomize the search
            random.seed(r) # Or just let it run
            
            # We need to clear the state for each restart?
            # Yes, used_edges starts empty.
            backtrack(0, set(), [], 0)
            
            if best_value == 190:
                break
        
        # Ensure types are correct
        final_result = []
        for i in range(n_trees):
            if best_result[i] is None:
                final_result.append(None)
            else:
                # Convert to list of ints
                final_result.append([int(x) for x in best_result[i]])
        
        return final_result

if __name__ == "__main__":
    res = construct()
    # Print value for debugging if needed
    # total = 0
    # for r in res:
    #    if r: total += len(r) - 1
    # print(total)
    pass
