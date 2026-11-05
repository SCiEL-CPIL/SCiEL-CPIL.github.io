import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_edges = 0
    best_graph = []
    
    # Helper: check if adding (u,v) is valid
    def is_valid_add(neigh, u, v):
        # C = N(u) ∩ N(v)
        C = neigh[u] & neigh[v]
        # Optimization: if bit_count(C) < 4, no K2,2 possible (need 4 distinct vertices)
        if C.bit_count() < 4:
            return True
        
        # Get list of vertices in C
        verts = []
        temp = C
        while temp:
            lsb = temp & -temp
            idx = (lsb.bit_length() - 1)
            verts.append(idx)
            temp ^= lsb
            
        # Check pairs for K2,2 in induced subgraph of C
        k = len(verts)
        for i in range(k):
            x = verts[i]
            nx = neigh[x] & C
            for j in range(i+1, k):
                y = verts[j]
                # Common neighbors of x and y within C
                # We need at least 2 common neighbors in C
                common = nx & neigh[y]
                if common.bit_count() >= 2:
                    return False
        return True

    # Helper: find a violation (returns edge to remove or None)
    def find_violation(neigh):
        for u in range(n):
            nu = neigh[u]
            for v in range(u+1, n):
                nv = neigh[v]
                common_uv = nu & nv
                if common_uv.bit_count() < 2:
                    continue
                for w in range(v+1, n):
                    nw = neigh[w]
                    # C = N(u) ∩ N(v) ∩ N(w)
                    C = common_uv & nw
                    # Remove u, v, w from C? 
                    # u, v, w cannot be in N(u) etc, so they are not in C.
                    # Wait, N(u) does not contain u. So C does not contain u, v, w.
                    if C.bit_count() >= 3:
                        # Found K3,3
                        # Remove a random edge from the K3,3
                        # The K3,3 edges are between {u,v,w} and C
                        # Pick a random vertex from C and a random vertex from {u,v,w}
                        verts = []
                        temp = C
                        while temp:
                            lsb = temp & -temp
                            idx = (lsb.bit_length() - 1)
                            verts.append(idx)
                            temp ^= lsb
                        c_rand = random.choice(verts)
                        u_rand = random.choice([u, v, w])
                        return (u_rand, c_rand)
        return None

    # Main loop
    while time.time() < deadline:
        # Random initialization
        # Start with p=0.6 to get dense graph, then repair
        p = 0.6
        neigh = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    neigh[u] |= (1 << v)
                    neigh[v] |= (1 << u)
        
        # Repair phase
        # Limit repair steps to avoid infinite loops
        repair_limit = 100
        steps = 0
        while steps < repair_limit and time.time() < deadline:
            viol = find_violation(neigh)
            if viol is None:
                break
            u, v = viol
            neigh[u] &= ~(1 << v)
            neigh[v] &= ~(1 << u)
            steps += 1
        
        if time.time() >= deadline:
            break
            
        # If we didn't fix all violations, skip this iteration or continue?
        # If find_violation still finds something, the graph is invalid.
        # We should only proceed to greedy add if it's valid.
        if find_violation(neigh) is not None:
            continue

        # Greedy Add phase
        # We will repeatedly try to add edges.
        # To speed up, we can maintain a list of candidate edges.
        # But for n=20, just iterating all pairs is fine.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle order to avoid bias
            pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
            random.shuffle(pairs)
            for u, v in pairs:
                if not (neigh[u] & (1 << v)):
                    if is_valid_add(neigh, u, v):
                        neigh[u] |= (1 << v)
                        neigh[v] |= (1 << u)
                        improved = True
        
        # Count edges
        edge_count = sum(neigh[u].bit_count() for u in range(n)) // 2
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_graph = []
            for u in range(n):
                for v in range(u+1, n):
                    if neigh[u] & (1 << v):
                        best_graph.append([u, v])

    return best_graph
