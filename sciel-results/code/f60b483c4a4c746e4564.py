"""
Strategy:
1. Generate the specific trees T1..T20 using the provided instance_trees logic.
2. Sort the trees by number of edges (ascending). T1 has 0 edges, T2 has 1, etc.
   Smaller trees are easier to pack and leave more "room" (in terms of unused edges) 
   for the larger, more constrained trees.
3. Use a backtracking search to embed the trees one by one into K20.
   - State: `used_edges` (set of tuples (u, v) with u < v).
   - For each tree, try to find an embedding (mapping of tree vertices 0..k-1 to host vertices 0..19)
     such that all mapped edges are distinct and not in `used_edges`.
   - Heuristic: Try to place the tree's "center" or high-degree vertices first? 
     Actually, simple backtracking on vertex assignments with pruning is likely sufficient for small trees.
     However, K20 is dense. The constraint is just edge-disjointness.
     Total edges in K20 = 190.
     Total edges in T1..T20 = sum(k-1 for k=1..20) = 0+1+...+19 = 190.
     So we MUST pack ALL trees perfectly to achieve the max score of 190.
     If we can't pack all, we pack as many as possible (highest raw value).
     Since the total edge count equals the capacity, a perfect packing is a decomposition of K20 into these trees.
     This is a known problem: Does K20 decompose into these specific trees?
     The trees are generated randomly. It's highly likely a perfect packing exists or is very close.
     
     Backtracking strategy:
     - Order trees by size (ascending).
     - For each tree, iterate through possible embeddings.
     - To speed up: Use a heuristic ordering for vertex assignment within a tree.
       Assign vertices of the tree to host vertices.
       Prune if any edge collides with used_edges.
     
     Time limit: 120s. 
     If backtracking is too slow, we might need to be smarter.
     However, for n=20, the trees are small (max 20 vertices).
     The search space for embedding a single tree of size k into 20 vertices is P(20, k) which is large, 
     but pruning by edge availability is strong.
     
     Alternative: 
     Since the total edges exactly match, this is an exact cover problem or graph decomposition.
     We can use a simple recursive backtracking.
     If it times out, we return the best partial result.
"""

import time
import sys
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: 
            trees.append([] if k == 1 else [(0, 1)]); 
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: 
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); 
            degree[leaf] -= 1; 
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); 
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    trees = instance_trees(20, 611)
    
    # Create a list of (num_edges, index, tree_edges)
    # We want to pack smaller trees first? Or larger?
    # Usually, packing larger, more rigid structures first is better for constraint satisfaction.
    # Let's try sorting by size descending. T20 first, then T19, etc.
    # Why? T20 is a spanning tree. It uses 19 edges. 
    # If we place T20 first, it consumes a large chunk of the edge set.
    # The remaining graph must accommodate T19...T1.
    # Actually, T20 is the most constrained in terms of "spanning" the whole vertex set if we consider connectivity, 
    # but here we just need edge disjointness.
    # Let's stick to sorting by size descending.
    
    indexed_trees = [(len(t), i, t) for i, t in enumerate(trees)]
    indexed_trees.sort(key=lambda x: x[0], reverse=True)
    
    # used_edges: set of (u, v) with u < v
    used_edges = set()
    # result: list of 20 entries, initially None
    result = [None] * 20
    
    # Pre-compute all edges in K20
    all_edges = set()
    for u in range(20):
        for v in range(u + 1, 20):
            all_edges.add((u, v))
            
    def backtrack(idx):
        nonlocal used_edges, result, deadline
        
        if time.time() > deadline:
            return False
            
        if idx == len(indexed_trees):
            return True
            
        num_edges, orig_idx, tree_edges = indexed_trees[idx]
        
        # If tree has 0 edges (T1), any unused vertex works? 
        # No, T1 is a single vertex. The entry is a list of 1 vertex.
        # It doesn't consume edges. But it consumes a "slot" for a vertex?
        # The problem says: "Entry k-1 is None or a list of k distinct int host vertices".
        # "Embedded trees must be edge-disjoint".
        # It does NOT say vertices must be disjoint.
        # So T1 can be mapped to ANY vertex. Let's map it to 0.
        # Wait, if T1 is mapped to 0, does it conflict? No, only edges conflict.
        # So T1 is trivial.
        
        if num_edges == 0:
            # Assign first available vertex? Or just 0?
            # Since vertices can be reused, just pick 0.
            result[orig_idx] = [0]
            return backtrack(idx + 1)
            
        # For trees with edges, we need to find an embedding.
        # Heuristic: Order the vertices of the tree to assign them.
        # We'll use a simple backtracking for the mapping of this specific tree.
        
        # To make the search efficient, let's order the vertices of the current tree 
        # such that high-degree vertices are assigned first.
        # Build adjacency list for the current tree
        adj = [[] for _ in range(len(tree_edges) + 1)] # Vertices are 0..k-1
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Sort vertices by degree descending
        k = len(adj)
        order = sorted(range(k), key=lambda v: len(adj[v]), reverse=True)
        
        # We need to map tree_vertex -> host_vertex
        # mapping: dict or list of length k, initialized to -1
        mapping = [-1] * k
        used_host_vertices = set() # Not strictly necessary for edge disjointness, but distinctness of mapping is required.
        
        # We also need to check edge collisions.
        # To prune early, we can check edges as soon as both endpoints are mapped.
        
        def map_tree(pos):
            nonlocal used_edges, deadline
            
            if time.time() > deadline:
                return False
                
            if pos == len(order):
                # All vertices mapped. Check if all edges are valid (not in used_edges).
                # Actually, we should check as we go to prune earlier.
                # Let's assume we checked during mapping.
                # But we need to add them to used_edges before returning True?
                # Yes.
                
                # Add edges to used_edges
                for u, v in tree_edges:
                    hu, hv = mapping[u], mapping[v]
                    e = (min(hu, hv), max(hu, hv))
                    if e in used_edges:
                        return False # Should not happen if we pruned correctly
                    used_edges.add(e)
                
                result[orig_idx] = [int(mapping[i]) for i in range(k)]
                
                # Recurse
                success = backtrack(idx + 1)
                
                if not success:
                    # Undo
                    for u, v in tree_edges:
                        hu, hv = mapping[u], mapping[v]
                        e = (min(hu, hv), max(hu, hv))
                        used_edges.remove(e)
                    result[orig_idx] = None
                    
                return success
            
            tv = order[pos]
            
            # Try assigning each host vertex
            for hv in range(20):
                if hv in used_host_vertices:
                    continue
                
                # Check if any edge from tv to an already mapped neighbor collides
                valid = True
                for neighbor in adj[tv]:
                    if mapping[neighbor] != -1:
                        hn = mapping[neighbor]
                        e = (min(hv, hn), max(hv, hn))
                        if e in used_edges:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Assign
                mapping[tv] = hv
                used_host_vertices.add(hv)
                
                if map_tree(pos + 1):
                    return True
                
                # Undo
                mapping[tv] = -1
                used_host_vertices.remove(hv)
                
            return False
            
        if map_tree(0):
            return True
            
        return False

    # Start backtracking
    # Note: The order of trees matters.
    # If we fail, we might want to try different orders or randomize?
    # Given the tight constraint (exact cover), randomization might help escape local minima if we were doing greedy,
    # but for exact backtracking, order is fixed by our sort.
    # If the first attempt fails (returns False), it means no solution exists with this order?
    # No, backtracking explores ALL possibilities for the given order.
    # If it returns False, it means there is NO solution for this specific set of trees?
    # Or it timed out?
    # If it times out, `backtrack` returns False implicitly or via time check.
    
    success = backtrack(0)
    
    # If success, all 20 are filled.
    # If not, we return whatever is in `result`.
    # `result` contains valid embeddings for the trees that were successfully placed before failure/timeout.
    # We need to ensure that if a tree was placed, its edges are in `used_edges`.
    # The code above handles adding/removing edges correctly in the recursion.
    # If we exit due to timeout, the state of `used_edges` and `result` corresponds to the current path.
    
    # One edge case: If T1 (0 edges) is placed, it doesn't add to used_edges.
    # This is correct.
    
    # Convert to ensure types are correct
    final_result = []
    for entry in result:
        if entry is None:
            final_result.append(None)
        else:
            # Ensure list of ints
            final_result.append([int(x) for x in entry])
            
    return final_result

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    # Calculate score
    score = 0
    for entry in res:
        if entry is not None:
            k = len(entry)
            if k > 1:
                score += k - 1
    print(f"Score: {score}")
