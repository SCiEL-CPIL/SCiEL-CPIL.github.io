"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    n = 40
    best_edges = []
    best_count = 0
    
    # Precompute bit masks for efficiency
    full_mask = (1 << n) - 1
    
    def neighbors_list(adj_bits, v):
        """Get list of neighbors from bitset."""
        result = []
        temp = adj_bits[v]
        while temp:
            lsb = temp & -temp
            result.append(lsb.bit_length() - 1)
            temp ^= lsb
        return result
    
    def creates_c6(adj_bits, u, v):
        """Check if adding edge (u,v) creates a C6 using bitsets."""
        nu = adj_bits[u] & ~(1 << v)
        nv = adj_bits[v] & ~(1 << u)
        
        x1s = neighbors_list(adj_bits, u)
        x1s = [x for x in x1s if x != v]
        x4s = neighbors_list(adj_bits, v)
        x4s = [x for x in x4s if x != u]
        
        for x1 in x1s:
            for x4 in x4s:
                if x1 == x4:
                    # Need triangle x1-x2-x3-x1 in G \ {u,v}
                    # x2 in N(x1)\{u,v,x1}, x3 in N(x2)\{u,v,x1,x2} cap N(x1)
                    A = adj_bits[x1] & ~(1 << u) & ~(1 << v) & ~(1 << x1)
                    temp_a = A
                    while temp_a:
                        lsb = temp_a & -temp_a
                        x2 = lsb.bit_length() - 1
                        # x3 must be in N(x2) and N(x1), not in {u,v,x1,x2}
                        B = adj_bits[x2] & adj_bits[x1] & ~(1 << u) & ~(1 << v) & ~(1 << x1) & ~(1 << x2)
                        if B != 0:
                            return True
                        temp_a ^= lsb
                else:
                    # Need path x1-x2-x3-x4
                    # x2 in N(x1)\{u,v,x4}, x3 in N(x2)\{u,v,x1} cap N(x4)
                    A = adj_bits[x1] & ~(1 << u) & ~(1 << v) & ~(1 << x4)
                    B = adj_bits[x4] & ~(1 << u) & ~(1 << v) & ~(1 << x1)
                    temp_a = A
                    while temp_a:
                        lsb = temp_a & -temp_a
                        x2 = lsb.bit_length() - 1
                        if (adj_bits[x2] & B) != 0:
                            return True
                        temp_a ^= lsb
        return False
    
    def try_add_edge(adj_bits, edges, u, v):
        """Try to add edge (u,v), return True if successful."""
        if (adj_bits[u] >> v) & 1:
            return False
        if creates_c6(adj_bits, u, v):
            return False
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        edges.append([u, v])
        return True
    
    def local_search(adj_bits, edges, max_iters=200):
        """Try edge swaps: remove one edge, add two that don't create C6."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try removing each edge and see if we can add two back
            for i in range(len(edges) - 1, -1, -1):
                if time.time() > deadline - 10.0:
                    return adj_bits, edges
                u, v = edges[i]
                # Temporarily remove
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                edges.pop(i)
                
                # Try to find two edges we can add
                added = 0
                candidates_to_try = []
                for a in range(n):
                    for b in range(a + 1, n):
                        if not ((adj_bits[a] >> b) & 1):
                            candidates_to_try.append((a, b))
                
                random.shuffle(candidates_to_try)
                for a, b in candidates_to_try:
                    if added >= 2:
                        break
                    if not creates_c6(adj_bits, a, b):
                        adj_bits[a] |= (1 << b)
                        adj_bits[b] |= (1 << a)
                        edges.append([a, b])
                        added += 1
                
                if added < 2:
                    # Restore original edge
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    edges.append([u, v])
                else:
                    improved = True
                    break
        
        return adj_bits, edges
    
    # Run multiple restarts
    for restart in range(30):
        if time.time() > deadline - 15.0:
            break
        
        # Initialize with 8 K5s
        adj_bits = [0] * n
        edges = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    edges.append([u, v])
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
        
        # Build candidate list with heuristic ordering
        # Prioritize: edges between different K5 components, lower degree first
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_bits[u] >> v) & 1:
                    # Score: prefer cross-component, lower degree sum
                    comp_u = u // 5
                    comp_v = v // 5
                    cross = 0 if comp_u == comp_v else 1
                    deg_u = bin(adj_bits[u]).count('1')
                    deg_v = bin(adj_bits[v]).count('1')
                    score = cross * 100 - (deg_u + deg_v)
                    candidates.append((score, u, v))
        
        # Sort by score descending (cross-component, low degree first)
        candidates.sort(key=lambda x: -x[0])
        candidates = [(u, v) for _, u, v in candidates]
        
        # Add some randomization to avoid deterministic traps
        for i in range(len(candidates) - 1, 0, -1):
            j = random.randint(0, i)
            candidates[i], candidates[j] = candidates[j], candidates[i]
        
        # Greedy addition
        current_count = len(edges)
        for u, v in candidates:
            if time.time() > deadline - 10.0:
                break
            if try_add_edge(adj_bits, edges, u, v):
                current_count += 1
        
        # Local search refinement
        if time.time() < deadline - 10.0:
            adj_bits, edges = local_search(adj_bits, edges, max_iters=100)
            current_count = len(edges)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = edges
    
    return best_edges
# EVOLVE-BLOCK-END
