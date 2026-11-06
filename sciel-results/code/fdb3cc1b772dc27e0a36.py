import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    n = 18
    
    # Bitmask helpers
    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    d_candidates = common & adj[c]
                    d_candidates &= ~((1 << (c + 1)) - 1)
                    if d_candidates:
                        return True
        return False

    def find_k4(adj):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    d_candidates = common & adj[c]
                    d_candidates &= ~((1 << (c + 1)) - 1)
                    if d_candidates:
                        d_bit = d_candidates & (-d_candidates)
                        d = d_bit.bit_length() - 1
                        return (a, b, c, d)
        return None

    def max_triangle_free_size(adj, time_limit=2.0):
        best = [0]
        t0 = time.time()
        timeout = [False]
        
        # Order vertices by degree (high to low) for better pruning
        deg = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -deg[v])
        
        # Precompute neighbor masks
        # adj is already a list of masks
        
        def backtrack(idx, current_mask, current_size):
            if timeout[0]:
                return
            if time.time() - t0 > time_limit:
                timeout[0] = True
                return
            
            if idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Pruning
            if current_size + (n - idx) <= best[0]:
                return
            
            v = order[idx]
            
            # Option 1: Exclude v
            backtrack(idx + 1, current_mask, current_size)
            
            # Option 2: Include v
            if not timeout[0]:
                # Check if adding v creates a triangle
                N_v = adj[v]
                C = current_mask & N_v
                valid = True
                if C:
                    tmp_c = C
                    while tmp_c:
                        bit = tmp_c & (-tmp_c)
                        u = bit.bit_length() - 1
                        tmp_c ^= bit
                        if adj[u] & C:
                            valid = False
                            break
                
                if valid:
                    backtrack(idx + 1, current_mask | (1 << v), current_size + 1)
        
        backtrack(0, 0, 0)
        return best[0]

    def make_k4_free(adj):
        for _ in range(500):
            k4 = find_k4(adj)
            if k4 is None:
                return True
            a, b, c, d = k4
            pair = random.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
            x, y = pair
            adj[x] &= ~(1 << y)
            adj[y] &= ~(1 << x)
        return not has_k4(adj)

    def random_k4_free_graph():
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if not make_k4_free(adj):
            # Fallback to Paley if random fails
            return construct_paley_17()
        return adj

    def construct_paley_17():
        adj = [0] * n
        residues = {1, 4, 9, 16, 8, 2, 15, 13}
        for i in range(17):
            for j in range(i + 1, 17):
                diff = (j - i) % 17
                if diff in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    # Initial candidates
    candidates = []
    candidates.append(construct_paley_17())
    for _ in range(5):
        candidates.append(random_k4_free_graph())
        
    # Evaluate initial candidates
    best_adj = None
    best_score = 999
    for adj in candidates:
        score = max_triangle_free_size(adj, time_limit=1.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            
    # Local Search
    while time.time() < deadline:
        improved = False
        adj = best_adj[:]
        
        # Try random mutations
        for _ in range(100):
            if time.time() > deadline:
                break
                
            move_type = random.choice(['add', 'remove', 'swap'])
            new_adj = adj[:]
            
            if move_type == 'add':
                non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj[i] >> j) & 1)]
                if not non_edges: continue
                u, v = random.choice(non_edges)
                # Check K4
                common = adj[u] & adj[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                creates_k4 = False
                tmp_c = common
                while tmp_c:
                    bit = tmp_c & (-tmp_c)
                    w = bit.bit_length() - 1
                    tmp_c ^= bit
                    if adj[w] & common:
                        creates_k4 = True
                        break
                if not creates_k4:
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
                else:
                    continue
                    
            elif move_type == 'remove':
                edges = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]
                if not edges: continue
                u, v = random.choice(edges)
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                
            else: # swap
                edges = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]
                non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj[i] >> j) & 1)]
                if not edges or not non_edges: continue
                u, v = random.choice(edges)
                # Find a non-edge involving u or v
                candidates_for_swap = []
                for x, y in non_edges:
                    if x == u or x == v or y == u or y == v:
                        candidates_for_swap.append((x, y))
                if not candidates_for_swap: continue
                x, y = random.choice(candidates_for_swap)
                
                # Remove (u,v), add (x,y)
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                new_adj[x] |= (1 << y)
                new_adj[y] |= (1 << x)
                
                # Check K4 in new_adj
                if has_k4(new_adj):
                    continue

            score = max_triangle_free_size(new_adj, time_limit=1.0)
            if score < best_score:
                best_score = score
                best_adj = new_adj
                improved = True
                adj = new_adj[:] # Restart from new best
                
        if not improved:
            # Try a random restart
            if random.random() < 0.1:
                new_adj = random_k4_free_graph()
                score = max_triangle_free_size(new_adj, time_limit=1.0)
                if score < best_score:
                    best_score = score
                    best_adj = new_adj

    return edges_from_adj(best_adj)

if __name__ == "__main__":
    edges = construct()
    print(f"Found graph with {len(edges)} edges")
