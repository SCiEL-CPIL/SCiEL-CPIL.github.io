"""Combined program: greedy C4-free construction with local search using bitmasks."""
import time
import random

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def check_c4_free(adj, n):
        """Check C4-freeness: no two vertices share 2+ common neighbors."""
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if common.bit_count() >= 2:
                    return False
        return True

    def min_degree(adj, n):
        return min(adj[i].bit_count() for i in range(n))

    def repair_c4s(adj, n, max_iter=50):
        """Repair C4s by removing edges. Returns True if C4-free."""
        for _ in range(max_iter):
            if elapsed() > deadline - 2:
                break
            found = False
            for i in range(n):
                if elapsed() > deadline - 2:
                    break
                for j in range(i + 1, n):
                    common = adj[i] & adj[j]
                    if common.bit_count() >= 2:
                        # Remove an edge to break the C4
                        tmp = common
                        lsb = tmp & -tmp
                        w = lsb.bit_length() - 1
                        adj[i] &= ~(1 << w)
                        adj[w] &= ~(1 << i)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        return check_c4_free(adj, n)

    def add_edges_to_boost_min_degree(adj, n, target, max_rounds=30):
        """Try to add edges to boost minimum degree while maintaining C4-freeness."""
        for _ in range(max_rounds):
            if elapsed() > deadline - 2:
                break
            
            degs = [adj[i].bit_count() for i in range(n)]
            min_deg = min(degs)
            if min_deg >= target:
                return True
            
            # Find vertices below target
            low_verts = [i for i in range(n) if degs[i] < target]
            if not low_verts:
                return True
            
            # Try to add edges for low-degree vertices
            improved = False
            random.shuffle(low_verts)
            
            for u in low_verts:
                if degs[u] >= target:
                    continue
                if elapsed() > deadline - 2:
                    break
                
                mask = (1 << n) - 1
                non_neighbors = (~adj[u]) & mask
                non_neighbors &= ~(1 << u)
                
                # Get list of non-neighbors
                nn_list = []
                tmp = non_neighbors
                while tmp:
                    lsb = tmp & -tmp
                    idx = lsb.bit_length() - 1
                    nn_list.append(idx)
                    tmp ^= lsb
                
                random.shuffle(nn_list)
                
                for v in nn_list:
                    if degs[v] >= target and degs[u] >= target:
                        break
                    # Check if adding (u,v) would create a C4
                    common = adj[u] & adj[v]
                    if common.bit_count() < 2:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degs[u] += 1
                        degs[v] += 1
                        improved = True
                        if degs[u] >= target:
                            break
            
            if not improved:
                break
        
        return min_degree(adj, n) >= target

    def local_search(adj, n, target, max_iters=100):
        """Local search: swap edges to improve minimum degree."""
        degs = [adj[i].bit_count() for i in range(n)]
        
        for _ in range(max_iters):
            if elapsed() > deadline - 2:
                break
            
            min_deg = min(degs)
            if min_deg >= target:
                return True
            
            # Find a low-degree vertex
            low_verts = [i for i in range(n) if degs[i] <= min_deg + 1]
            random.shuffle(low_verts)
            
            for u in low_verts:
                if degs[u] >= target:
                    continue
                
                # Try to swap: remove edge (u, x) and add edge (u, y)
                # where x is a neighbor and y is a non-neighbor
                neighbors = []
                tmp = adj[u]
                while tmp:
                    lsb = tmp & -tmp
                    idx = lsb.bit_length() - 1
                    neighbors.append(idx)
                    tmp ^= lsb
                
                mask = (1 << n) - 1
                non_neighbors = (~adj[u]) & mask
                non_neighbors &= ~(1 << u)
                
                nn_list = []
                tmp = non_neighbors
                while tmp:
                    lsb = tmp & -tmp
                    idx = lsb.bit_length() - 1
                    nn_list.append(idx)
                    tmp ^= lsb
                
                random.shuffle(neighbors)
                random.shuffle(nn_list)
                
                for x in neighbors:
                    for y in nn_list:
                        # Check if swap is valid
                        # After removing (u,x) and adding (u,y):
                        # Need to verify no C4 is created
                        
                        # Remove (u,x) temporarily
                        adj[u] &= ~(1 << x)
                        adj[x] &= ~(1 << u)
                        degs[u] -= 1
                        degs[x] -= 1
                        
                        # Check if we can add (u,y)
                        common = adj[u] & adj[y]
                        if common.bit_count() < 2:
                            # Check that removing (u,x) doesn't create issues
                            # and adding (u,y) is safe
                            adj[u] |= (1 << y)
                            adj[y] |= (1 << u)
                            degs[u] += 1
                            degs[y] += 1
                            
                            # Verify C4-free
                            if check_c4_free(adj, n):
                                if min(degs) >= target:
                                    return True
                            else:
                                # Revert
                                adj[u] &= ~(1 << y)
                                adj[y] &= ~(1 << u)
                                degs[u] -= 1
                                degs[y] -= 1
                        else:
                            # Revert
                            adj[u] |= (1 << x)
                            adj[x] |= (1 << u)
                            degs[u] += 1
                            degs[x] += 1
                            continue
                        
                        # Revert if not successful
                        adj[u] &= ~(1 << y)
                        adj[y] &= ~(1 << u)
                        degs[u] -= 1
                        degs[y] -= 1
                        adj[u] |= (1 << x)
                        adj[x] |= (1 << u)
                        degs[u] += 1
                        degs[x] += 1
                        
                        break
                    break
        
        return min_degree(adj, n) >= target

    def try_construct(n, attempts=30):
        target = n - 39
        if target < 1:
            target = 1
        
        # Start with a slightly denser graph
        p = (target + 1) / n
        if p > 0.5: p = 0.5
        
        best_adj = None
        best_md = -1

        for attempt in range(attempts):
            if elapsed() > deadline - 2:
                break

            # 1. Initialize random graph using bitmasks
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

            # 2. Repair: Remove edges to break C4s
            if not repair_c4s(adj, n, max_iter=80):
                continue

            md = min_degree(adj, n)
            if md > best_md:
                best_md = md
                best_adj = adj[:]

            # 3. If min degree is too low, try to add edges
            if md < target:
                if add_edges_to_boost_min_degree(adj, n, target, max_rounds=20):
                    if check_c4_free(adj, n):
                        md = min_degree(adj, n)
                        if md > best_md:
                            best_md = md
                            best_adj = adj[:]
                        if md >= target:
                            edges = []
                            for i in range(n):
                                for j in range(i + 1, n):
                                    if adj[i] & (1 << j):
                                        edges.append([i, j])
                            return n, edges

            # 4. Local search to improve
            if md < target:
                if local_search(adj, n, target, max_iters=50):
                    if check_c4_free(adj, n):
                        md = min_degree(adj, n)
                        if md > best_md:
                            best_md = md
                            best_adj = adj[:]
                        if md >= target:
                            edges = []
                            for i in range(n):
                                for j in range(i + 1, n):
                                    if adj[i] & (1 << j):
                                        edges.append([i, j])
                            return n, edges

            # Final check
            if check_c4_free(adj, n):
                md = min_degree(adj, n)
                if md > best_md:
                    best_md = md
                    best_adj = adj[:]
                if md >= target:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return n, edges

        if best_adj is not None and best_md >= target:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if best_adj[i] & (1 << j):
                        edges.append([i, j])
            return n, edges
            
        return None

    # Try from largest n down
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        result = try_construct(n)
        if result is not None:
            return {'n': result[0], 'edges': result[1]}

    # Fallback: trivial graph
    return {'n': 39, 'edges': []}
