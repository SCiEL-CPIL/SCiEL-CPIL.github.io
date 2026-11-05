import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 18

    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        return is_independent(common, adj)

    def largest_tf(adj, time_limit=None):
        best = [0]
        full_mask = (1 << n) - 1
        
        # Greedy Lower Bound - multiple random orders
        for _ in range(3):
            degrees = [bin(adj[i]).count('1') for i in range(n)]
            order = list(range(n))
            random.shuffle(order)
            order.sort(key=lambda x: -degrees[x])
            
            greedy_set = 0
            for v in order:
                if is_independent(adj[v] & greedy_set, adj):
                    greedy_set |= (1 << v)
            sz = bin(greedy_set).count('1')
            if sz > best[0]:
                best[0] = sz
        
        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = bin(S_mask).count('1')
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return

            # Select vertex with max degree in C
            max_deg = -1
            chosen_v = -1
            c = C_mask
            while c:
                lsb = c & -c
                vid = lsb.bit_length() - 1
                deg_in_c = bin(adj[vid] & C_mask).count('1')
                if deg_in_c > max_deg:
                    max_deg = deg_in_c
                    chosen_v = vid
                c ^= lsb
                
            v = chosen_v
            v_bit = 1 << v
            
            # Branch 1: Exclude v
            bnb(S_mask, C_mask & ~v_bit)
            
            # Branch 2: Include v
            if is_independent(adj[v] & S_mask, adj):
                bnb(S_mask | v_bit, C_mask & ~v_bit)
                
        bnb(0, full_mask)
        return best[0]

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    random.seed(42)
    best_adj = [0] * n
    best_score = n

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute all possible edges as list
    edge_set = set(all_edges)
    
    def greedy_fill(adj, edge_order):
        for u, v in edge_order:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    def local_search(adj, current_score, time_limit):
        """Try to improve by adding or removing edges."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            random.shuffle(all_edges)
            for u, v in all_edges:
                if time.time() > time_limit:
                    break
                has_edge = (adj[u] >> v) & 1
                if has_edge:
                    # Try removing edge
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    if random.random() < 0.15:
                        new_score = largest_tf(adj, time_limit=time_limit)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                        else:
                            # Revert
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                    else:
                        # Keep removal (denser isn't always better for TF)
                        pass
                else:
                    # Try adding edge
                    if can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        if random.random() < 0.15:
                            new_score = largest_tf(adj, time_limit=time_limit)
                            if new_score < current_score:
                                current_score = new_score
                                improved = True
                            else:
                                # Revert
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                        else:
                            pass
        return current_score

    # Main search loop
    while time.time() < deadline - 10.0:
        adj = [0] * n
        random.shuffle(all_edges)
        greedy_fill(adj, all_edges)
        
        # Evaluate with time limit
        eval_limit = min(deadline - 5.0, time.time() + 3.0)
        score = largest_tf(adj, time_limit=eval_limit)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            print(f"New best: {best_score}")
        
        # Local search if time permits
        if time.time() < deadline - 15.0:
            ls_limit = min(deadline - 10.0, time.time() + 8.0)
            score = local_search(adj, score, ls_limit)
            if score < best_score:
                best_score = score
                best_adj = adj[:]
                print(f"LS improved: {best_score}")

    # Final polish: try more aggressive local search on best
    if time.time() < deadline - 3.0:
        adj = best_adj[:]
        polish_limit = deadline - 2.0
        score = local_search(adj, best_score, polish_limit)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            print(f"Polish improved: {best_score}")

    edges = get_edge_list(best_adj)
    print(f"Final: score={best_score}, edges={len(edges)}")
    return edges
