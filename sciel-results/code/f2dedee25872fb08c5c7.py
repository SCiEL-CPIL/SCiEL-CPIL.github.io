import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def popcount(x):
        return bin(x).count('1')

    def build_adj_bitset(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def is_triangle_free(adj, n):
        for i in range(n):
            neighbors = adj[i]
            while neighbors:
                lsb = neighbors & (-neighbors)
                j = lsb.bit_length() - 1
                neighbors ^= lsb
                common = adj[j] & adj[i] & ~(1 << i)
                if common:
                    return False
        return True

    def find_independent_set_heuristic(adj, n, k):
        """
        Returns an independent set of size k if found by greedy heuristic, else None.
        """
        if n < k:
            return None
        
        remaining = (1 << n) - 1
        indep_set = []
        
        for _ in range(k):
            if not remaining:
                break
            min_deg = n
            min_v = -1
            temp = remaining
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = popcount(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
                    if deg == 0:
                        break
            
            if min_v == -1:
                break
                
            indep_set.append(min_v)
            remaining &= ~(1 << min_v)
            remaining &= ~adj[min_v]
            
        if len(indep_set) >= k:
            return indep_set
        return None

    def verify_adj_exact(n, adj):
        """
        Exact check if alpha < 10.
        """
        if not is_triangle_free(adj, n):
            return False
            
        best_alpha = [0]
        
        def dfs(candidates, current_size):
            if current_size > best_alpha[0]:
                best_alpha[0] = current_size
            if best_alpha[0] >= 10:
                return
            if not candidates:
                return
            if current_size + popcount(candidates) <= best_alpha[0]:
                return
            
            # Pick vertex with max degree in candidates
            max_deg = -1
            v = -1
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                vv = lsb.bit_length() - 1
                temp ^= lsb
                deg = popcount(adj[vv] & candidates)
                if deg > max_deg:
                    max_deg = deg
                    v = vv
            
            if v == -1:
                return
                
            bit_v = 1 << v
            rest = candidates & ~bit_v
            
            # Branch 1: Include v
            dfs(rest & ~adj[v], current_size + 1)
            if best_alpha[0] >= 10:
                return
            
            # Branch 2: Exclude v
            dfs(rest, current_size)
            
        dfs((1 << n) - 1, 0)
        return best_alpha[0] < 10

    def andrasfai_graph(k):
        n = 3 * k
        edges = []
        for a in range(3):
            for i in range(k):
                for b in range(3):
                    if (b - a) % 3 != 1:
                        continue
                    for j in range(k):
                        if (j - i) % k in (0, 1):
                            u = a * k + i
                            v = b * k + j
                            if u < v:
                                edges.append([u, v])
        return {'n': n, 'edges': edges}

    best_n = 27
    best_graph = andrasfai_graph(9)
    
    # Store candidates for later exact verification
    candidates = {}
    for n in range(28, 41):
        candidates[n] = None

    # Try to find graphs for n=28 to 40
    for target_n in range(28, 41):
        if time.time() - start_time > deadline - 15:
            break
            
        found_for_n = False
        for restart in range(15):
            if time.time() - start_time > deadline - 10:
                break
                
            random.seed(target_n * 100 + restart)
            
            adj = [0] * target_n
            # Start with random graph
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if random.random() < 0.4:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Phase 1: Remove triangles
            for _ in range(1000):
                if time.time() - start_time > deadline - 10:
                    break
                triangle = None
                for i in range(target_n):
                    neighbors = adj[i]
                    while neighbors:
                        lsb1 = neighbors & (-neighbors)
                        j = lsb1.bit_length() - 1
                        neighbors ^= lsb1
                        common = adj[j] & adj[i] & ~(1 << i) & ~(1 << j)
                        if common:
                            lsb2 = common & (-common)
                            k = lsb2.bit_length() - 1
                            triangle = (i, j, k)
                            break
                    if triangle:
                        break
                if triangle is None:
                    break
                i, j, k = triangle
                choice = random.randint(0, 2)
                if choice == 0:
                    adj[i] &= ~(1 << j); adj[j] &= ~(1 << i)
                elif choice == 1:
                    adj[i] &= ~(1 << k); adj[k] &= ~(1 << i)
                else:
                    adj[j] &= ~(1 << k); adj[k] &= ~(1 << j)
                    
            if not is_triangle_free(adj, target_n):
                continue

            # Phase 2: Reduce alpha
            for _ in range(200):
                if time.time() - start_time > deadline - 10:
                    break
                indep_set = find_independent_set_heuristic(adj, target_n, 10)
                if indep_set is None:
                    break
                
                added_edge = False
                random.shuffle(indep_set)
                for idx_u in range(len(indep_set)):
                    u = indep_set[idx_u]
                    for idx_v in range(idx_u + 1, len(indep_set)):
                        v = indep_set[idx_v]
                        if adj[u] & (1 << v):
                            continue
                        if (adj[u] & adj[v]) == 0:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            added_edge = True
                            break
                    if added_edge:
                        break
                if not added_edge:
                    break

            # If heuristic says alpha < 10, store as candidate
            if is_triangle_free(adj, target_n) and find_independent_set_heuristic(adj, target_n, 10) is None:
                if candidates[target_n] is None:
                    candidates[target_n] = adj
                    found_for_n = True
                # We can keep searching for better ones if time permits, 
                # but for now, one candidate per n is enough to check.

    # Exact verification of candidates in descending order of n
    for n in range(40, 27, -1):
        if candidates[n] is not None:
            if time.time() - start_time > deadline - 5:
                break
            if verify_adj_exact(n, candidates[n]):
                best_n = n
                # Convert to edges
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if candidates[n][i] & (1 << j):
                            edges.append([i, j])
                best_graph = {'n': n, 'edges': edges}
                break

    return best_graph

