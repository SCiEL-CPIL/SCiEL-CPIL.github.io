import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 105  # safety margin

    # Bitmask adjacency: adj[i] is a bitmask of neighbors of vertex i
    adj = [0] * n

    def count_edges():
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def has_edge(u, v):
        return (adj[u] >> v) & 1

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def is_add_valid(u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        if has_edge(u, v):
            return True
        nu = adj[u]
        nv = adj[v]
        # For each pair (a1, a2) in N(u), check if they share >= 2 common neighbors in N(v)
        # a1, a2 must be distinct from u, v
        nu_mask = nu & ~(1 << v)  # exclude v from nu consideration
        nv_mask = nv & ~(1 << u)  # exclude u from nv consideration
        
        # Iterate over all pairs in nu
        temp = nu_mask
        while temp:
            # Get lowest set bit
            lb = temp & (-temp)
            a1 = lb.bit_length() - 1
            temp ^= lb
            # For each a2 > a1 in nu
            temp2 = nu_mask & ((1 << n) - 1) & ~((1 << (a1 + 1)) - 1)
            while temp2:
                lb2 = temp2 & (-temp2)
                a2 = lb2.bit_length() - 1
                temp2 ^= lb2
                # Common neighbors of a1 and a2 that are also in nv
                common = adj[a1] & adj[a2] & nv_mask
                if bin(common).count('1') >= 2:
                    return False
        return True

    def find_violation():
        """Find a K3,3 in the current graph. Returns list of 9 edges or None."""
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    common = adj[i] & adj[j] & adj[k]
                    if bin(common).count('1') >= 3:
                        # Get 3 common neighbors
                        temp = common
                        b_list = []
                        while len(b_list) < 3:
                            lb = temp & (-temp)
                            b = lb.bit_length() - 1
                            b_list.append(b)
                            temp ^= lb
                        b1, b2, b3 = b_list
                        return [(i, b1), (i, b2), (i, b3),
                                (j, b1), (j, b2), (j, b3),
                                (k, b1), (k, b2), (k, b3)]
        return None

    def get_edges_from_adj(adj_list):
        edges = []
        for i in range(n):
            temp = adj_list[i]
            while temp:
                lb = temp & (-temp)
                j = lb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                temp ^= lb
        return edges

    def is_valid_graph(adj_list):
        """Verify no K3,3 exists."""
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    common = adj_list[i] & adj_list[j] & adj_list[k]
                    if bin(common).count('1') >= 3:
                        return False
        return True

    # Initialize with structured 5x4 grid-like graph
    # Vertices 0-4: group A, 5-9: group B, 10-14: group C, 15-19: group D
    # Connect groups in a pattern that avoids K3,3
    random.seed(42)
    
    # Start with a moderately dense random graph
    p_init = 0.45
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < p_init:
                add_edge(u, v)
    
    # Remove all K3,3 violations
    max_iter = 10000
    for _ in range(max_iter):
        if time.time() > deadline - 30:
            break
        viol = find_violation()
        if viol is None:
            break
        # Remove a random edge from the violation
        e = random.choice(viol)
        remove_edge(e[0], e[1])
    
    best_edges = count_edges()
    best_adj = adj[:]
    
    # Local search with 1-edge add and 2-edge swap
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        move_type = random.random()
        
        if move_type < 0.4:
            # Try adding a single edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or has_edge(u, v):
                no_improve += 1
                continue
            if is_add_valid(u, v):
                add_edge(u, v)
                new_count = count_edges()
                if new_count > best_edges:
                    best_edges = new_count
                    best_adj = adj[:]
                    no_improve = 0
                else:
                    remove_edge(u, v)
                    no_improve += 1
            else:
                no_improve += 1
                
        elif move_type < 0.8:
            # Try 2-edge swap: remove edge e1, add edge e2
            # Pick a random existing edge to potentially remove
            existing_edges = []
            for i in range(n):
                temp = adj[i]
                while temp:
                    lb = temp & (-temp)
                    j = lb.bit_length() - 1
                    if i < j:
                        existing_edges.append((i, j))
                    temp ^= lb
            
            if not existing_edges:
                no_improve += 1
                continue
                
            e1 = random.choice(existing_edges)
            u1, v1 = e1
            
            # Pick a random non-edge to potentially add
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2:
                no_improve += 1
                continue
            if has_edge(u2, v2):
                no_improve += 1
                continue
            # Make sure it's not the same edge
            if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                no_improve += 1
                continue
            
            # Remove e1 temporarily
            remove_edge(u1, v1)
            
            # Check if adding e2 is valid
            if is_add_valid(u2, v2):
                add_edge(u2, v2)
                # Check if graph is still valid (no K3,3)
                if is_valid_graph(adj):
                    new_count = count_edges()
                    if new_count > best_edges:
                        best_edges = new_count
                        best_adj = adj[:]
                        no_improve = 0
                    else:
                        # Revert: remove e2, add back e1
                        remove_edge(u2, v2)
                        add_edge(u1, v1)
                        no_improve += 1
                else:
                    # Invalid, revert
                    remove_edge(u2, v2)
                    add_edge(u1, v1)
                    no_improve += 1
            else:
                # Add back e1
                add_edge(u1, v1)
                no_improve += 1
        else:
            # Perturbation: remove a random edge
            existing_edges = []
            for i in range(n):
                temp = adj[i]
                while temp:
                    lb = temp & (-temp)
                    j = lb.bit_length() - 1
                    if i < j:
                        existing_edges.append((i, j))
                    temp ^= lb
            
            if existing_edges:
                e = random.choice(existing_edges)
                remove_edge(e[0], e[1])
                no_improve += 1
            else:
                no_improve += 1
    
    # Final polish: try to add any remaining valid edges
    adj = best_adj[:]
    for u in range(n):
        for v in range(u + 1, n):
            if not has_edge(u, v):
                if is_add_valid(u, v):
                    add_edge(u, v)
    
    final_count = count_edges()
    if final_count > best_edges:
        best_adj = adj[:]
        best_edges = final_count
    
    # Convert to list of edges
    result = get_edges_from_adj(best_adj)
    return result
