import time
import math
import random

def construct():
    start_time = time.time()
    # Evaluator kills at 90s. We leave a 10s buffer for safety and cleanup.
    deadline = start_time + 80.0
    
    N = 256
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1
    
    # 1. Identify valid nodes (2*a is not a square)
    valid_nodes = []
    for a in range(1, N + 1):
        if 2 * a not in squares:
            valid_nodes.append(a)
            
    # Map original values to indices 0..M-1 for compact bitmasks
    M = len(valid_nodes)
    val_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # 2. Build conflict graph as bitmasks
    # adj_mask[i] has bit j set if valid_nodes[i] and valid_nodes[j] conflict
    adj_mask = [0] * M
    
    for i, a in enumerate(valid_nodes):
        mask = 0
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                if b in val_to_idx:
                    j = val_to_idx[b]
                    if j != i: # Diagonal already handled by valid_nodes filter
                        mask |= (1 << j)
        adj_mask[i] = mask
        
    # 3. Greedy Coloring for Upper Bound
    # Returns: 
    #   order: list of vertices in the order they were colored (reverse of processing)
    #   colors: list of color indices for each vertex in 'order'
    #   num_colors: total number of colors used
    def greedy_color(vertices):
        # vertices is a list of indices
        # We want to color the subgraph induced by vertices
        # Tomita's approach: 
        # 1. Select vertex with min degree (or similar heuristic)
        # 2. Assign smallest available color
        # 3. Remove vertex and its neighbors? No, standard greedy coloring for bound:
        #    Actually, for MIS B&B, we color the *candidate* set.
        #    The bound is the number of colors.
        #    The standard Tomita coloring for MIS:
        #    - Maintain a set of uncolored vertices.
        #    - Pick a vertex, assign a color, remove it and its neighbors from consideration for *that* color?
        #    No, that's for finding a clique.
        #    For MIS, we want an upper bound on the independent set size.
        #    Coloring the graph such that adjacent vertices have different colors.
        #    The size of the MIS <= number of colors.
        #    
        #    Efficient greedy coloring for bound:
        #    1. Sort vertices by some heuristic (e.g., degree in subgraph).
        #    2. Iterate, assigning the smallest color not used by neighbors.
        
        # To make it fast, we use a simple greedy approach on the subgraph.
        # Since M is small (~150-200), we can afford O(M^2) or better per call if called often.
        # But B&B calls this at every node. We need it fast.
        
        # Tomita's specific coloring for MIS (also known as "coloring for upper bound"):
        # It processes vertices one by one.
        # Let's use the "DSATUR" or simple greedy.
        # Simple greedy is often sufficient and fast.
        
        # We will return the vertices ordered by color class, and the color count.
        # This ordering is crucial for the B&B loop.
        
        # Heuristic: Pick vertex with minimum degree in the remaining subgraph?
        # Or just process in a fixed order?
        # Let's try: 
        # 1. Get list of vertices.
        # 2. While list not empty:
        #    a. Pick vertex v with min degree in current list (or max, depends).
        #       Actually, for coloring, picking min degree first often leads to fewer colors?
        #       Or picking the one with most constraints?
        #       Let's stick to a standard greedy:
        #       Iterate through vertices. For each, assign smallest color not used by neighbors.
        #       To improve bound, we should process vertices that are "dense" first?
        
        # Let's implement the Tomita-style coloring used in MAXClique/MIS solvers:
        # It maintains a set of uncolored vertices.
        # It picks a vertex, assigns a color, and removes it from the set.
        # It does NOT remove neighbors.
        # The bound is the max color index + 1.
        
        # To make the bound tighter, the order of picking vertices matters.
        # A common heuristic is to pick the vertex with the highest degree in the *remaining* subgraph?
        # Or lowest?
        # Let's try: pick vertex with max degree in the subgraph induced by uncolored vertices.
        
        uncolored = set(vertices)
        colors = {}
        color_count = 0
        order = []
        
        # Precompute degrees in the subgraph? Too slow to recompute from scratch.
        # We can approximate or just pick arbitrarily if M is small.
        # Given M ~ 150, even O(M^2) per color call is ~22k ops.
        # B&B tree could be large.
        
        # Let's use a simpler, faster greedy:
        # 1. Sort vertices by degree in the *original* graph (precomputed).
        # 2. Iterate, assign smallest available color.
        
        # Precomputed degrees
        degs = [bin(adj_mask[i]).count('1') for i in range(M)]
        
        # Sort vertices by degree descending (densest first)
        # This tends to use more colors, which is a looser bound?
        # Wait, we want a *tight* bound (small number of colors).
        # Using fewer colors is better for pruning.
        # Greedy coloring with "first fit" on a specific order.
        # Order by increasing degree often yields fewer colors?
        # Let's try increasing degree.
        
        verts_list = list(vertices)
        verts_list.sort(key=lambda x: degs[x])
        
        # Map color to set of vertices having that color?
        # We just need the count and the order.
        
        # To be efficient, we check neighbors in the *current* subgraph?
        # No, we check neighbors in the *original* graph that are also in `vertices`.
        
        for v in verts_list:
            # Find colors used by neighbors of v in `vertices`
            used_colors = set()
            mask = adj_mask[v]
            # Iterate neighbors
            # Optimization: If we stored neighbors as a list, iteration is faster than bit scanning?
            # With M~150, bit scanning is fine.
            
            # Get neighbors of v that are in `vertices`
            # We don't have a mask for `vertices` handy in this local scope efficiently without precomp.
            # Let's just check if neighbor is in `uncolored`? No, `uncolored` changes.
            # We need to check neighbors that have *already been colored*.
            
            # Actually, standard greedy coloring:
            # When coloring v, look at neighbors of v that are already colored.
            
            # This requires knowing which neighbors are colored.
            # Since we process in `verts_list` order, we can just check `colors` dict.
            
            for u in range(M):
                if (mask >> u) & 1:
                    if u in colors:
                        used_colors.add(colors[u])
            
            # Assign smallest color not in used_colors
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            if c + 1 > color_count:
                color_count = c + 1
            order.append(v)
            
        return order, colors, color_count

    # 4. Branch and Bound
    best_size = 0
    best_set = set()
    
    # Initial lower bound: Greedy heuristic
    # Let's run a quick greedy to get a starting best_size
    remaining = set(range(M))
    current_set = []
    while remaining:
        # Pick node with min degree in remaining
        min_deg = float('inf')
        min_node = None
        for v in remaining:
            deg = 0
            mask = adj_mask[v]
            # Count neighbors in remaining
            # Simple loop
            for u in remaining:
                if u != v and (mask >> u) & 1:
                    deg += 1
            if deg < min_deg:
                min_deg = deg
                min_node = v
        if min_node is None:
            break
        current_set.append(min_node)
        remaining.discard(min_node)
        # Remove neighbors
        mask = adj_mask[min_node]
        to_remove = []
        for u in remaining:
            if (mask >> u) & 1:
                to_remove.append(u)
        for u in to_remove:
            remaining.discard(u)
            
    best_size = len(current_set)
    best_set = set(current_set)
    
    # B&B Search
    # We use a stack for DFS
    # Each item in stack: (candidates_list, colors_dict, num_colors, current_set_list)
    # To save memory and time, we pass lists.
    
    # Initial state
    initial_order, initial_colors, initial_num_colors = greedy_color(list(range(M)))
    
    # The B&B loop structure from Tomita:
    # expand(P, X, k)
    # P: set of candidate vertices (ordered by color)
    # X: set of vertices already in the independent set
    # k: size of X
    
    def expand(P, X, k):
        nonlocal best_size, best_set
        nonlocal deadline, start_time
        
        if k + len(P) <= best_size:
            return
            
        # Color P
        order, colors, num_colors = greedy_color(P)
        
        if k + num_colors <= best_size:
            return
            
        # Iterate through order
        # P is effectively the set of vertices in 'order'
        # We need to process them in the order they appear in 'order'
        
        # To make it efficient, we can just iterate through 'order'
        # But we need to maintain the set of remaining candidates for the recursive call.
        
        # Let's use the standard recursive formulation:
        # for i in range(len(order)):
        #    v = order[i]
        #    if colors[v] + k > best_size: # Bound check
        #       new_P = [u for u in order[i+1:] if u not in neighbors(v)]
        #       new_X = X + [v]
        #       expand(new_P, new_X, k+1)
        
        # This list comprehension `new_P` is expensive.
        # We can optimize by passing indices or using bitmasks.
        # Given time constraints, let's try the list version first but optimize if slow.
        
        # Optimization: 
        # Instead of creating a new list `new_P` by filtering `order[i+1:]`,
        # we can note that `new_P` consists of vertices in `order[i+1:]` that are NOT neighbors of v.
        
        for i in range(len(order)):
            v = order[i]
            
            # Upper bound check using color of v
            # The max size we can get from P[i:] is num_colors - colors[v] + 1?
            # Actually, the bound is num_colors.
            # If k + 1 + (num_colors - 1) <= best_size, prune?
            # The standard check is: if k + 1 + (number of colors in P[i+1:]) <= best_size
            # Since colors are assigned 1..C, and we process in order,
            # the remaining vertices have colors <= num_colors.
            # A tighter bound: k + 1 + (num_colors - colors[v]) ? No.
            # Let's stick to the simple bound: if k + 1 + len(P[i+1:]) <= best_size
            # Or the coloring bound: if k + 1 + (num_colors) <= best_size is too loose.
            
            # Tomita's bound: 
            # If k + 1 + (num_colors - colors[v]) <= best_size: return?
            # Actually, the vertices after i have colors in [1, num_colors].
            # The max independent set in the subgraph induced by P[i+1:] is <= num_colors.
            # So if k + 1 + num_colors <= best_size, we could prune the whole thing earlier.
            # Inside the loop, we can check:
            # if k + 1 + (num_colors - colors[v]) <= best_size: 
            #    # This is not quite right. The bound for the suffix is not simply num_colors - color[v].
            
            # Let's use the standard: 
            # if k + 1 + len(P[i+1:]) <= best_size: continue
            # This is weak.
            
            # Let's rely on the coloring bound at the start of expand() to prune whole branches.
            # Inside the loop, we just recurse.
            
            # Check time
            if time.time() > deadline:
                return
                
            # Recurse
            # New P: vertices in order[i+1:] that are not neighbors of v
            v_mask = adj_mask[v]
            new_P = []
            for j in range(i + 1, len(order)):
                u = order[j]
                if not (v_mask >> u) & 1:
                    new_P.append(u)
            
            new_X = X + [v]
            if k + 1 > best_size:
                best_size = k + 1
                best_set = set(new_X)
                
            expand(new_P, new_X, k + 1)

    # Start B&B
    # To avoid recursion depth issues or to manage time better, we can use a stack.
    # But Python recursion limit is ~1000. M~150, so depth <= 150. It's safe.
    
    # However, the `new_P` list creation is O(M).
    # Total work could be high.
    
    # Let's run the B&B
    # Initial P is all valid nodes
    # We need to order them initially?
    # expand() calls greedy_color which orders them.
    
    # One issue: `expand` modifies `best_size`.
    # We need to be careful with `P` being a list of indices.
    
    # Let's wrap it in a try-except to catch any unexpected errors or timeouts
    try:
        # We pass a list of all valid nodes
        # Note: greedy_color inside expand will sort them.
        expand(list(range(M)), [], 0)
    except Exception as e:
        # If B&B crashes or times out internally (shouldn't, but just in case)
        pass

    # Convert best_set (indices) back to values
    result = [valid_nodes[i] for i in best_set]
    
    # Final verification (optional but good for safety)
    # Verify no sum is square
    squares_set = squares
    is_valid = True
    for i in range(len(result)):
        a = result[i]
        if 2*a in squares_set:
            is_valid = False
            break
        for j in range(i+1, len(result)):
            b = result[j]
            if a+b in squares_set:
                is_valid = False
                break
        if not is_valid:
            break
            
    if not is_valid:
        # Fallback to a simple valid set if something went wrong
        # Just return a single element that is valid
        for v in valid_nodes:
            return [v]
            
    return result

# Test run (commented out for submission)
# if __name__ == "__main__":
#     res = construct()
#     print(f"Size: {len(res)}")
#     print(res)
