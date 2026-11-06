import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    def check_k4(n, adj):
        # Returns (u, v, w) if K4 found, else None
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if (au >> v) & 1:
                    common = au & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c -= lsb
                        # Check if w is connected to any other node in common
                        # common includes w? No, common is N(u) & N(v).
                        # adj[w] & common checks if w has a neighbor in common.
                        # Since w is in common, we must ensure we don't count w itself if it were in adj[w] (it's not).
                        # But we need an edge (w, x) where x is in common and x != w.
                        # adj[w] & common does exactly this.
                        if adj[w] & common:
                            return (u, v, w)
        return None

    def check_ind6(n, adj):
        # Returns list of 6 vertices if Ind6 found, else None
        mask = (1 << n) - 1
        comp = [0] * n
        for i in range(n):
            comp[i] = (mask ^ adj[i]) & ~(1 << i)
        
        def backtrack(candidates, k, path):
            if k == 0:
                return path
            if len(candidates) < k:
                return None
            # Heuristic: pick vertex with highest degree in induced subgraph first?
            # For speed, just iterate.
            for i in range(len(candidates)):
                v = candidates[i]
                new_cands = [c for c in candidates[i+1:] if (comp[v] >> c) & 1]
                res = backtrack(new_cands, k-1, path + [v])
                if res:
                    return res
            return None
            
        return backtrack(list(range(n)), 6, [])

    def is_valid(n, adj):
        if check_k4(n, adj):
            return False
        if check_ind6(n, adj):
            return False
        return True

    def local_search(n, time_budget):
        deadline_ls = time.time() + time_budget
        # Initialize
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        while time.time() < deadline_ls:
            k4 = check_k4(n, adj)
            if k4:
                u, v, w = k4
                # Flip a random edge in the K4
                # The K4 is u, v, w, and some x in common(u,v) connected to w.
                # We just flip one of the edges (u,v), (u,w), (v,w)
                choice = random.randint(0, 2)
                if choice == 0:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                elif choice == 1:
                    adj[u] ^= (1 << w)
                    adj[w] ^= (1 << u)
                else:
                    adj[v] ^= (1 << w)
                    adj[w] ^= (1 << v)
                continue
            
            ind6 = check_ind6(n, adj)
            if ind6:
                # Flip a random edge within the independent set
                u, v = random.sample(ind6, 2)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                continue
            
            return adj
        
        return None

    def try_extend_to_36(adj35):
        # Try to add vertex 35 to a valid 35-vertex graph
        n = 35
        adj = [0] * 36
        for i in range(n):
            adj[i] = adj35[i]
        
        # Constraints for vertex 35:
        # 1. No K4: For every triangle (u,v,w) in G35, 35 can connect to at most 2 of {u,v,w}.
        # 2. No Ind6: For every independent set of size 5 in G35, 35 must connect to at least 1 of them.
        
        # Precompute triangles and independent 5-sets
        triangles = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c -= lsb
                        if w > v:
                            triangles.append((u, v, w))
        
        ind5s = []
        # Find all independent sets of size 5
        # This is K5 in complement
        mask = (1 << n) - 1
        comp = [0] * n
        for i in range(n):
            comp[i] = (mask ^ adj[i]) & ~(1 << i)
            
        def find_k5(candidates, k, path):
            if k == 0:
                ind5s.append(tuple(path))
                return
            if len(candidates) < k:
                return
            for i in range(len(candidates)):
                v = candidates[i]
                new_cands = [c for c in candidates[i+1:] if (comp[v] >> c) & 1]
                find_k5(new_cands, k-1, path + [v])
        
        find_k5(list(range(n)), 5, [])
        
        # Backtracking search for edges of vertex 35
        # Variables: x[0..34]
        # x[i] = 1 if edge (35, i) exists
        
        # Pruning:
        # If we assign x, check if any triangle is fully connected to 35 (3 edges).
        # If we assign x, check if any ind5 is fully disconnected from 35 (0 edges).
        
        solution = [0] * n
        
        def backtrack(idx, connected_count):
            if idx == n:
                return True
            
            # Try 0 (no edge)
            solution[idx] = 0
            # Check constraints
            valid = True
            for t in triangles:
                if t[0] == idx or t[1] == idx or t[2] == idx:
                    # Count how many of t are connected to 35
                    cnt = 0
                    for v in t:
                        if v < idx:
                            cnt += solution[v]
                        elif v == idx:
                            cnt += 0
                    if cnt == 3:
                        valid = False
                        break
            if valid:
                for s in ind5s:
                    if idx in s:
                        cnt = 0
                        for v in s:
                            if v < idx:
                                cnt += solution[v]
                        if cnt == 0:
                            valid = False
                            break
            if valid:
                if backtrack(idx + 1, connected_count):
                    return True
            
            # Try 1 (edge exists)
            solution[idx] = 1
            valid = True
            for t in triangles:
                if t[0] == idx or t[1] == idx or t[2] == idx:
                    cnt = 0
                    for v in t:
                        if v < idx:
                            cnt += solution[v]
                        elif v == idx:
                            cnt += 1
                    if cnt == 3:
                        valid = False
                        break
            if valid:
                for s in ind5s:
                    if idx in s:
                        cnt = 0
                        for v in s:
                            if v < idx:
                                cnt += solution[v]
                        if cnt == 0:
                            valid = False
                            break
            if valid:
                if backtrack(idx + 1, connected_count + 1):
                    return True
            
            return False
        
        if backtrack(0, 0):
            # Apply solution
            for i in range(n):
                if solution[i]:
                    adj[i] |= (1 << 35)
                    adj[35] |= (1 << i)
            return adj
        return None

    best_n = 0
    best_adj = None

    # 1. Try to find 35-vertex graph
    adj35 = local_search(35, 20)
    if adj35 and is_valid(35, adj35):
        best_n = 35
        best_adj = adj35
        # 2. Try to extend to 36
        adj36 = try_extend_to_36(adj35)
        if adj36 and is_valid(36, adj36):
            best_n = 36
            best_adj = adj36

    # 3. If 36 not found, try direct local search on 36
    if best_n < 36 and time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 10:
            adj36_direct = local_search(36, remaining - 5)
            if adj36_direct and is_valid(36, adj36_direct):
                best_n = 36
                best_adj = adj36_direct

    # 4. Fallback to 35 if not found yet
    if best_n < 35 and time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 5:
            adj35_fallback = local_search(35, remaining - 2)
            if adj35_fallback and is_valid(35, adj35_fallback):
                best_n = 35
                best_adj = adj35_fallback

    # 5. Fallback to 34
    if best_n < 34 and time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 2:
            adj34 = local_search(34, remaining)
            if adj34 and is_valid(34, adj34):
                best_n = 34
                best_adj = adj34

    # Convert to edges
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}
