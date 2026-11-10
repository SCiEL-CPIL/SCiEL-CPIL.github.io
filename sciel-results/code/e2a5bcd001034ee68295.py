import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, leaving 20s margin
    
    # 1. Construct the Clebsch Graph (Triangle-free, 16 vertices, 40 edges, chi=4)
    # Vertices are 0..15.
    # Clebsch graph is the folded 5-cube.
    # Two vertices u, v in {0..31} are connected if they differ in 1 bit.
    # Folded: u ~ v if v = u XOR 31.
    # We map 0..31 to 0..15 by taking min(u, u^31) as representative? 
    # Actually, standard Clebsch construction:
    # Vertices are 5-bit strings. Distance 1 in the folded cube.
    # Let's use the standard generation:
    
    edges_set = set()
    # Map 32 nodes to 16 classes
    # Class of x is min(x, x^31)
    def get_class(x):
        return min(x, x ^ 31)
    
    # We need to ensure we use 0..15 as labels.
    # The classes are the 16 pairs {x, x^31}.
    # Let's assign an index 0..15 to each class.
    class_to_id = {}
    id_to_class = []
    for i in range(32):
        c = get_class(i)
        if c not in class_to_id:
            class_to_id[c] = len(class_to_id)
            id_to_class.append(c)
            
    node_to_id = {}
    for i in range(32):
        node_to_id[i] = class_to_id[get_class(i)]
        
    for i in range(32):
        u_id = node_to_id[i]
        for k in range(5):
            j = i ^ (1 << k)
            v_id = node_to_id[j]
            if u_id < v_id:
                edges_set.add((u_id, v_id))
                
    initial_edges = [list(e) for e in edges_set]
    initial_count = len(initial_edges)
    
    # 2. Verification and Potential Search (Structural Change: Bitmask Hill-Climbing)
    # We represent the graph as a list of 16-bit integers (bitmasks).
    # adj[i] is a bitmask of neighbors of i.
    adj = [0] * 16
    for u, v in edges_set:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def is_triangle_free(adj):
        for u in range(16):
            # For each pair of neighbors v, w of u, check if v-w is an edge
            # Faster: for each neighbor v, check if adj[v] & adj[u] has any bits set other than u?
            # Actually, triangle (u,v,w) exists if adj[u] & adj[v] & adj[w] != 0? No.
            # Triangle exists if there are u,v,w such that adj[u] has v, adj[v] has w, adj[w] has u.
            # Check: for each u, for each v in adj[u], if (adj[u] & adj[v]) != 0, then there is a common neighbor.
            # If (adj[u] & adj[v]) is not 0, let w be a bit in it. Then u-v, u-w, v-w are edges?
            # If w is in adj[u] and w is in adj[v], then u-w and v-w are edges.
            # So if (adj[u] & adj[v]) != 0, we have a triangle u-v-w.
            # Wait, if w is a common neighbor, edges are (u,v), (u,w), (v,w).
            # So we just need to check if adj[u] & adj[v] is non-zero for any edge (u,v)?
            # No, if (u,v) is an edge, and they share a neighbor w, then (u,v), (u,w), (v,w) form a triangle.
            # So: For every edge (u,v), if (adj[u] & adj[v]) != 0, it's NOT triangle free.
            for v in range(16):
                if adj[u] & (1 << v): # if (u,v) is an edge
                    if adj[u] & adj[v]: # if they have a common neighbor
                        return False
        return True

    def is_not_3_colorable(adj):
        # Backtracking 3-coloring
        # colors[i] = 0, 1, 2 or -1
        colors = [-1] * 16
        
        def backtrack(i):
            if i == 16:
                return True
            for c in range(3):
                ok = True
                for j in range(i):
                    if adj[i] & (1 << j): # if i and j are connected
                        if colors[j] == c:
                            ok = False
                            break
                if ok:
                    colors[i] = c
                    if backtrack(i + 1):
                        return True
                    colors[i] = -1
            return False
            
        return not backtrack(0)

    # Check initial graph
    if not is_triangle_free(adj) or not is_not_3_colorable(adj):
        # Fallback to Groetzsch if Clebsch construction failed (should not happen)
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5 + w], [w, 5 + u]]
        return edges + [[5 + u, 10] for u in range(5)]

    best_edges = initial_edges
    best_adj = adj[:]
    best_count = initial_count
    
    # Try to improve using local search (Add edges, Swap edges)
    # Since Clebsch is maximal, adding edges is impossible.
    # We can try swapping: remove 1 edge, add 2?
    # This is a 2-opt move.
    # Given time constraints, and high likelihood of optimality, we do a very limited search.
    
    if time.time() < deadline - 5:
        # Try random edge removals and then greedy edge addition
        # This is a "restart" strategy
        attempts = 0
        while time.time() < deadline - 10 and attempts < 5:
            attempts += 1
            current_adj = best_adj[:]
            
            # Remove 1-2 random edges
            edges_list = [(u, v) for u in range(16) for v in range(u+1, 16) if current_adj[u] & (1 << v)]
            if not edges_list: break
            
            for _ in range(2):
                if not edges_list: break
                idx = random.randint(0, len(edges_list) - 1)
                u, v = edges_list.pop(idx)
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            
            # Greedy addition of edges
            # Sort all non-edges
            non_edges = [(u, v) for u in range(16) for v in range(u+1, 16) if not (current_adj[u] & (1 << v))]
            # Try adding them if it doesn't create a triangle
            # We don't check chi here during greedy add to save time, only at the end
            for u, v in non_edges:
                if time.time() > deadline - 10: break
                # Check if adding (u,v) creates a triangle
                # Common neighbors of u and v
                common = current_adj[u] & current_adj[v]
                if common == 0:
                    # Safe to add
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
            
            # Check validity of the new graph
            if is_triangle_free(current_adj) and is_not_3_colorable(current_adj):
                # Count edges
                count = 0
                for u in range(16):
                    count += bin(current_adj[u]).count('1')
                count //= 2
                
                if count > best_count:
                    best_count = count
                    best_adj = current_adj
                    # Convert back to edge list
                    new_edges = []
                    for u in range(16):
                        for v in range(u+1, 16):
                            if current_adj[u] & (1 << v):
                                new_edges.append([u, v])
                    best_edges = new_edges

    # Final conversion
    if best_adj is not adj: # If we updated
        final_edges = []
        for u in range(16):
            for v in range(u+1, 16):
                if best_adj[u] & (1 << v):
                    final_edges.append([u, v])
        return final_edges
    else:
        return best_edges
