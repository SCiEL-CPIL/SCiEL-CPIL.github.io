"""C4-free graph construction: sparse start + directed edge addition + targeted repair."""
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
            ai = adj[i]
            for j in range(i + 1, n):
                common = ai & adj[j]
                if common.bit_count() >= 2:
                    return False
        return True

    def is_c4_free_after_add(adj, n, u, v):
        """Check if adding edge (u,v) creates a C4."""
        common = adj[u] & adj[v]
        return common.bit_count() < 2

    def min_degree(adj, n):
        return min(adj[i].bit_count() for i in range(n))

    def find_c4s(adj, n):
        """Find all C4s (pairs of vertices with 2+ common neighbors)."""
        c4s = []
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                common = ai & adj[j]
                if common.bit_count() >= 2:
                    # Get the common neighbors
                    cn = []
                    tmp = common
                    while tmp:
                        lsb = tmp & -tmp
                        idx = lsb.bit_length() - 1
                        cn.append(idx)
                        tmp ^= lsb
                    c4s.append((i, j, cn))
        return c4s

    def repair_c4s_smart(adj, n, max_iter=300):
        """Repair C4s by removing edges strategically to preserve degree."""
        for _ in range(max_iter):
            if elapsed() > deadline - 3:
                break
            c4s = find_c4s(adj, n)
            if not c4s:
                return True
            
            # For each C4, decide which edge to remove
            # Strategy: remove edge from the lower-degree endpoint to one of the common neighbors
            # that has the highest degree (to preserve balance)
            best_removal = None
            best_score = -1
            
            for i, j, cn in c4s:
                deg_i = adj[i].bit_count()
                deg_j = adj[j].bit_count()
                
                # Try removing (i, w) or (j, w) for w in cn
                for w in cn:
                    # Score: prefer removing edge from lower-degree vertex
                    if deg_i <= deg_j:
                        # Remove (i, w)
                        # Penalty: how many other edges incident to i and w
                        score = deg_i * 10 + deg_j * 5 - adj[w].bit_count() * 2
                    else:
                        # Remove (j, w)
                        score = deg_j * 10 + deg_i * 5 - adj[w].bit_count() * 2
                    
                    if score > best_score:
                        best_score = score
                        if deg_i <= deg_j:
                            best_removal = (i, w)
                        else:
                            best_removal = (j, w)
            
            if best_removal is None:
                break
                
            u, v = best_removal
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        
        return check_c4_free(adj, n)

    def directed_edge_addition(adj, n, target, max_rounds=200):
        """Add edges directed at boosting minimum degree, maintaining C4-freeness."""
        for _ in range(max_rounds):
            if elapsed() > deadline - 3:
                break
            
            degs = [adj[i].bit_count() for i in range(n)]
            min_deg = min(degs)
            if min_deg >= target:
                return True
            
            # Find vertices below target, sorted by degree (lowest first)
            low_verts = sorted(range(n), key=lambda i: degs[i])
            low_verts = [i for i in low_verts if degs[i] < target]
            if not low_verts:
                return True
            
            improved = False
            random.shuffle(low_verts)
            
            for u in low_verts:
                if degs[u] >= target:
                    continue
                if elapsed() > deadline - 3:
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
                
                # Sort non-neighbors by degree (prefer connecting to high-degree vertices
                # that have room, i.e., low common-neighbor count with u)
                nn_list.sort(key=lambda v: adj[v].bit_count(), reverse=True)
                random.shuffle(nn_list)  # add randomness
                
                for v in nn_list:
                    if degs[u] >= target:
                        break
                    # Check if adding (u,v) would create a C4
                    if is_c4_free_after_add(adj, n, u, v):
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

    def targeted_local_search(adj, n, target, max_iters=300):
        """Local search: swap edges to improve minimum degree while maintaining C4-freeness."""
        degs = [adj[i].bit_count() for i in range(n)]
        
        for _ in range(max_iters):
            if elapsed() > deadline - 3:
                break
            
            min_deg = min(degs)
            if min_deg >= target:
                return True
            
            # Find the lowest-degree vertex
            low_verts = [i for i in range(n) if degs[i] <= min_deg + 1]
            random.shuffle(low_verts)
            
            for u in low_verts:
                if degs[u] >= target:
                    continue
                
                # Get neighbors and non-neighbors
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
                
                if not neighbors or not nn_list:
                    continue
                
                random.shuffle(neighbors)
                random.shuffle(nn_list)
                
                found_swap = False
                for x in neighbors:
                    if found_swap:
                        break
                    # Check if removing (u,x) is safe (doesn't create new C4s elsewhere)
                    # and if there's a valid y to add
                    valid_ys = []
                    for y in nn_list:
                        if is_c4_free_after_swap(adj, n, u, x, y):
                            # Additional check: does removing (u,x) and adding (u,y)
                            # maintain C4-freeness globally?
                            valid_ys.append(y)
                    
                    if not valid_ys:
                        continue
                    
                    random.shuffle(valid_ys)
                    
                    for y in valid_ys:
                        # Perform the swap
                        adj[u] &= ~(1 << x)
                        adj[x] &= ~(1 << u)
                        degs[u] -= 1
                        degs[x] -= 1
                        
                        adj[u] |= (1 << y)
                        adj[y] |= (1 << u)
                        degs[u] += 1
                        degs[y] += 1
                        
                        # Verify C4-freeness after swap
                        if check_c4_free(adj, n):
                            new_min = min(degs)
                            if new_min >= min_deg:
                                found_swap = True
                                break
                            else:
                                # Revert
                                adj[u] &= ~(1 << y)
                                adj[y] &= ~(1 << u)
                                degs[u] -= 1
                                degs[y] -= 1
                                adj[u] |= (1 << x)
                                adj[x] |= (1 << u)
                                degs[u] += 1
                                degs[x] += 1
                        else:
                            # Revert
                            adj[u] &= ~(1 << y)
                            adj[y] &= ~(1 << u)
                            degs[u] -= 1
                            degs[y] -= 1
                            adj[u] |= (1 << x)
                            adj[x] |= (1 << u)
                            degs[u] += 1
                            degs[x] += 1
                        break
                    if found_swap:
                        break
                if found_swap:
                    break
        
        return min_degree(adj, n) >= target

    def is_c4_free_after_swap(adj, n, u, x, y):
        """Check if swapping (u,x)->(u,y) maintains C4-freeness."""
        # Temporarily remove (u,x)
        adj[u] &= ~(1 << x)
        adj[x] &= ~(1 << u)
        
        # Check if (u,y) would create C4
        ok = is_c4_free_after_add(adj, n, u, y)
        
        # Restore (u,x)
        adj[u] |= (1 << x)
        adj[x] |= (1 << u)
        
        return ok

    def try_construct(n, attempts=80):
        target = max(1, n - 39)
        
        best_adj = None
        best_md = -1

        for attempt in range(attempts):
            if elapsed() > deadline - 3:
                break

            # Sparse random start
            p = 0.3 + 0.3 * (attempt / max(1, attempts))  # vary density
            if p > 0.6:
                p = 0.6
            
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

            # Repair C4s
            if not repair_c4s_smart(adj, n, max_iter=300):
                continue

            md = min_degree(adj, n)
            if md > best_md:
                best_md = md
                best_adj = adj[:]

            # Directed edge addition
            if md < target:
                if directed_edge_addition(adj, n, target, max_rounds=150):
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

            # Local search
            if md < target:
                if targeted_local_search(adj, n, target, max_iters=200):
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
    time_per_n = {47: 45, 46: 45, 45: 30, 44: 20, 43: 10, 42: 5, 41: 3, 40: 2}
    
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if elapsed() > deadline - 5:
            break
        old_deadline = deadline
        deadline = min(deadline, start_time + time_per_n.get(n, 10))
        
        result = try_construct(n)
        if result is not None:
            return {'n': result[0], 'edges': result[1]}
        
        deadline = old_deadline

    # Fallback: trivial graph
    return {'n': 39, 'edges': []}
