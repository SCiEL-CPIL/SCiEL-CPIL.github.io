import random
import time

def construct():
    n = 40
    target_m = 401
    deadline = time.time() + 145.0
    
    # State variables
    # adj[u] is a set of neighbors
    # book[u][v] is the book size for edge (u,v), stored in a dict or 2D list
    # Using a 2D list of size 40x40 for book sizes is fine (1600 ints)
    
    def init_random_graph():
        adj = [set() for _ in range(n)]
        book = [[0]*n for _ in range(n)]
        edges = []
        
        # Generate random graph
        all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs[:target_m]:
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
            
        # Compute initial book sizes
        freq = [0] * (n + 1)
        for u, v in edges:
            b = len(adj[u] & adj[v])
            book[u][v] = b
            book[v][u] = b
            freq[b] += 1
            
        max_b = 0
        for k in range(n, -1, -1):
            if freq[k] > 0:
                max_b = k
                break
                
        return adj, book, edges, freq, max_b

    def compute_sum_squares(book, edges):
        s = 0
        for u, v in edges:
            s += book[u][v] ** 2
        return s

    def local_search(adj, book, edges, freq, max_b, time_limit):
        best_max = max_b
        best_edges = edges[:]
        best_book = [row[:] for row in book]
        best_freq = freq[:]
        best_adj = [set(s) for s in adj]
        
        current_max = max_b
        current_sum_sq = compute_sum_squares(book, edges)
        
        # Simulated Annealing parameters
        T = 50.0
        cooling = 0.999
        
        while time.time() < time_limit:
            # Pick an edge to remove
            # Prefer edges with high book size
            candidates = [e for e in edges if book[e[0]][e[1]] >= current_max - 2]
            if not candidates:
                candidates = edges
            u, v = random.choice(candidates)
            
            # Pick a non-edge to add
            # Try a few random non-edges
            found_add = False
            for _ in range(50):
                a = random.randrange(n)
                b = random.randrange(n)
                if a == b or b in adj[a]:
                    continue
                
                # Evaluate swap (u,v) -> (a,b)
                # 1. Remove (u,v)
                C_rm = adj[u] & adj[v]
                old_b_uv = book[u][v]
                
                # Check if this removal reduces max
                # We need to simulate the freq changes
                
                new_freq = freq[:]
                # Decrement old book of (u,v)
                new_freq[old_b_uv] -= 1
                
                # Decrement books of edges in C_rm
                for c in C_rm:
                    old_b_uc = book[u][c]
                    new_freq[old_b_uc] -= 1
                    new_freq[old_b_uc - 1] += 1
                    
                    old_b_vc = book[v][c]
                    new_freq[old_b_vc] -= 1
                    new_freq[old_b_vc - 1] += 1
                
                # Find new max after removal
                new_max_after_rm = current_max
                if new_freq[new_max_after_rm] == 0:
                    while new_max_after_rm > 0 and new_freq[new_max_after_rm] == 0:
                        new_max_after_rm -= 1
                
                # 2. Add (a,b)
                C_add = adj[a] & adj[b]
                new_b_ab = len(C_add)
                
                # Increment book of (a,b)
                new_freq[new_b_ab] += 1
                
                # Increment books of edges in C_add
                for c in C_add:
                    old_b_ac = book[a][c]
                    new_freq[old_b_ac] -= 1
                    new_freq[old_b_ac + 1] += 1
                    
                    old_b_bc = book[b][c]
                    new_freq[old_b_bc] -= 1
                    new_freq[old_b_bc + 1] += 1
                
                # Find new max after addition
                new_max = new_max_after_rm
                if new_b_ab > new_max:
                    new_max = new_b_ab
                for c in C_add:
                    if book[a][c] + 1 > new_max:
                        new_max = book[a][c] + 1
                    if book[b][c] + 1 > new_max:
                        new_max = book[b][c] + 1
                
                # Calculate new sum of squares
                # Change in sum sq:
                # - old_b_uv^2
                # - sum_{c in C_rm} (book[u][c]^2 - (book[u][c]-1)^2) = sum (2*book[u][c] - 1)
                # - sum_{c in C_rm} (book[v][c]^2 - (book[v][c]-1)^2) = sum (2*book[v][c] - 1)
                # + new_b_ab^2
                # + sum_{c in C_add} ((book[a][c]+1)^2 - book[a][c]^2) = sum (2*book[a][c] + 1)
                # + sum_{c in C_add} ((book[b][c]+1)^2 - book[b][c]^2) = sum (2*book[b][c] + 1)
                
                delta_sq = -old_b_uv**2
                for c in C_rm:
                    delta_sq -= (2*book[u][c] - 1)
                    delta_sq -= (2*book[v][c] - 1)
                delta_sq += new_b_ab**2
                for c in C_add:
                    delta_sq += (2*book[a][c] + 1)
                    delta_sq += (2*book[b][c] + 1)
                
                new_sum_sq = current_sum_sq + delta_sq
                
                # Acceptance criterion
                accept = False
                if new_max < current_max:
                    accept = True
                elif new_max == current_max:
                    if new_sum_sq < current_sum_sq:
                        accept = True
                    else:
                        # Simulated annealing step
                        delta_E = new_sum_sq - current_sum_sq
                        if random.random() < pow(2.718, -delta_E / T):
                            accept = True
                else:
                    # new_max > current_max
                    delta_E = (new_max - current_max) * 100 + (new_sum_sq - current_sum_sq)
                    if random.random() < pow(2.718, -delta_E / T):
                        accept = True
                
                if accept:
                    # Apply changes
                    adj[u].remove(v)
                    adj[v].remove(u)
                    edges.remove((u, v))
                    for c in C_rm:
                        book[u][c] -= 1
                        book[c][u] -= 1
                        book[v][c] -= 1
                        book[c][v] -= 1
                    book[u][v] = 0
                    book[v][u] = 0
                    
                    adj[a].add(b)
                    adj[b].add(a)
                    edges.append((a, b))
                    book[a][b] = new_b_ab
                    book[b][a] = new_b_ab
                    for c in C_add:
                        book[a][c] += 1
                        book[c][a] += 1
                        book[b][c] += 1
                        book[c][b] += 1
                        
                    freq = new_freq
                    current_max = new_max
                    current_sum_sq = new_sum_sq
                    
                    if current_max < best_max:
                        best_max = current_max
                        best_edges = edges[:]
                        best_book = [row[:] for row in book]
                        best_freq = freq[:]
                        best_adj = [set(s) for s in adj]
                        
                    if best_max <= 7:
                        return best_adj, best_book, best_edges, best_freq, best_max
                        
                    found_add = True
                    break
            
            T *= cooling
            if T < 0.1:
                T = 50.0 # Restart temperature if stuck? Or break?
                # Let's just break if T is very low and no improvement for a while
                # For simplicity, just continue cooling
                if T < 0.01:
                    break

        return best_adj, best_book, best_edges, best_freq, best_max

    best_overall_max = float('inf')
    best_overall_edges = None
    
    # Multiple restarts
    while time.time() < deadline - 10:
        adj, book, edges, freq, max_b = init_random_graph()
        
        if max_b < best_overall_max:
            best_overall_max = max_b
            best_overall_edges = edges[:]
            
        if best_overall_max <= 7:
            break
            
        # Run local search
        time_left = deadline - time.time() - 5
        if time_left < 10:
            break
            
        adj, book, edges, freq, max_b = local_search(adj, book, edges, freq, max_b, time.time() + time_left)
        
        if max_b < best_overall_max:
            best_overall_max = max_b
            best_overall_edges = edges[:]
            
        if best_overall_max <= 7:
            break

    result = [[u, v] for u, v in best_overall_edges]
    
    # Verification
    assert len(result) == 401
    assert len(set(tuple(sorted(e)) for e in result)) == 401
    
    return result
