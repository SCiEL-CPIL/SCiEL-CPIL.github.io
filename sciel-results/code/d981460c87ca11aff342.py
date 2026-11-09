import time
import random

def construct():
    n = 18
    start = time.time()
    deadline = start + 140
    
    def is_k4_free(adj):
        for a in range(n):
            for b in range(a+1, n):
                if not (adj[a] & (1 << b)):
                    continue
                common_ab = adj[a] & adj[b]
                cn = common_ab
                while cn:
                    c = (cn & -cn).bit_length() - 1
                    cn &= cn - 1
                    if c <= b:
                        continue
                    if cn & adj[c]:
                        return False
        return True

    def compute_L_fast(adj):
        """Compute largest triangle-free induced subgraph size using branch and bound."""
        # Precompute tri_mask
        tri_mask = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    tri_mask[u][v] = tri_mask[v][u] = adj[u] & adj[v]
        
        best = 0
        
        # DFS: build a triangle-free set, track size
        # At each step, try adding a vertex that doesn't create a triangle
        # Use pruning: if remaining vertices + current_size <= best, stop
        
        def dfs(current_mask, current_size, available_mask):
            nonlocal best
            if current_size > best:
                best = current_size
            
            if available_mask == 0:
                return
            
            # Pruning: even if we add all available, can't beat best
            if current_size + bin(available_mask).count('1') <= best:
                return
            
            # Try adding each vertex from available
            am = available_mask
            while am:
                v = (am & -am).bit_length() - 1
                am &= am - 1
                
                # Check if adding v creates a triangle with any two vertices in current_mask
                # v creates triangle if there exist u,w in current_mask with u~w and (u~v and w~v)
                # i.e., N(v) & current_mask contains an edge
                nv_in_current = adj[v] & current_mask
                # Check if nv_in_current has any edge
                nm = nv_in_current
                has_edge = False
                while nm:
                    u = (nm & -nm).bit_length() - 1
                    nm &= nm - 1
                    if adj[u] & nv_in_current:
                        has_edge = True
                        break
                if has_edge:
                    continue
                
                # Adding v is safe. New available: remove v and any vertex that would form triangle with v and some future vertex
                # Actually, we need to ensure that any future vertex w added doesn't form triangle with v and some other vertex in current+{v}
                # Simpler: just remove v from available, and continue
                new_mask = current_mask | (1 << v)
                new_avail = available_mask & ~(1 << v)
                dfs(new_mask, current_size + 1, new_avail)
        
        all_mask = (1 << n) - 1
        dfs(0, 0, all_mask)
        return best

    def make_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Turan graph T(18,3)
    adj_turan = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if i % 3 != j % 3:
                adj_turan[i] |= (1 << j)
                adj_turan[j] |= (1 << i)
    
    best_L = compute_L_fast(adj_turan)
    best_adj = adj_turan[:]
    print(f"Turan L = {best_L}", flush=True)
    
    def gen_random_k4_free(density=0.5):
        adj = [0]*n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        changed = True
        iters = 0
        while changed and iters < 500:
            changed = False
            iters += 1
            for a in range(n):
                if changed:
                    break
                for b in range(a+1, n):
                    if changed:
                        break
                    if not (adj[a] & (1 << b)):
                        continue
                    common_ab = adj[a] & adj[b]
                    cn = common_ab
                    while cn:
                        c = (cn & -cn).bit_length() - 1
                        cn &= cn - 1
                        if c <= b:
                            continue
                        if cn & adj[c]:
                            adj[a] &= ~(1 << c)
                            adj[c] &= ~(1 << a)
                            changed = True
                            break
                    if changed:
                        break
        return adj

    def local_search(adj, max_time=5):
        current_adj = adj[:]
        current_L = compute_L_fast(current_adj)
        
        while time.time() < max_time:
            improved = False
            for i in range(n):
                if time.time() > max_time:
                    break
                for j in range(i+1, n):
                    if time.time() > max_time:
                        break
                    old_state = bool(current_adj[i] & (1 << j))
                    if old_state:
                        current_adj[i] &= ~(1 << j)
                        current_adj[j] &= ~(1 << i)
                    else:
                        current_adj[i] |= (1 << j)
                        current_adj[j] |= (1 << i)
                    
                    if is_k4_free(current_adj):
                        L = compute_L_fast(current_adj)
                        if L < current_L:
                            current_L = L
                            improved = True
                        else:
                            if old_state:
                                current_adj[i] |= (1 << j)
                                current_adj[j] |= (1 << i)
                            else:
                                current_adj[i] &= ~(1 << j)
                                current_adj[j] &= ~(1 << i)
                    else:
                        if old_state:
                            current_adj[i] |= (1 << j)
                            current_adj[j] |= (1 << i)
                        else:
                            current_adj[i] &= ~(1 << j)
                            current_adj[j] &= ~(1 << i)
            if not improved:
                break
        return current_adj, current_L

    random.seed(42)
    
    for trial in range(30):
        if time.time() > deadline - 15:
            break
        density = 0.3 + 0.4 * (trial % 5) / 5
        adj = gen_random_k4_free(density)
        if not is_k4_free(adj):
            continue
        L = compute_L_fast(adj)
        if L < best_L:
            best_L = L
            best_adj = adj[:]
            print(f"Trial {trial}: L = {L}", flush=True)
        if time.time() < deadline - 20:
            adj_ls, L_ls = local_search(adj, max_time=min(5, deadline - time.time() - 20))
            if L_ls < best_L:
                best_L = L_ls
                best_adj = adj_ls[:]
                print(f"Trial {trial} LS: L = {L_ls}", flush=True)
    
    if time.time() < deadline - 20:
        adj_ls, L_ls = local_search(adj_turan, max_time=min(10, deadline - time.time() - 20))
        if L_ls < best_L:
            best_L = L_ls
            best_adj = adj_ls[:]
            print(f"Turan LS: L = {L_ls}", flush=True)
    
    return make_edge_list(best_adj)
