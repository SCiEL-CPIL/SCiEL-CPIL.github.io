import time
import random
import sys

def construct():
    n = 20
    deadline = time.time() + 110.0
    
    # Bitmask representation: adj[i] is an integer bitmask of neighbors of vertex i
    bits = [1 << i for i in range(n)]
    
    def popcount(x):
        return bin(x).count('1')
    
    def can_add(u, v, adj):
        if u == v:
            return False
        if adj[u] & bits[v]:
            return False
        
        # Check common neighbors of u and v: must be <= 2
        cu = adj[u] & adj[v]
        if popcount(cu) > 2:
            return False
        
        # Check all pairs involving u or v with their neighbors
        # For each neighbor w of v (w != u): common(u, w) must be <= 1 (since v will be added)
        nw = adj[v] & ~bits[u]
        w = nw
        while w:
            lb = w & (-w)
            wi = lb.bit_length() - 1
            cw = adj[u] & adj[wi]
            if popcount(cw) > 1:
                return False
            w ^= lb
        
        # For each neighbor w of u (w != v): common(v, w) must be <= 1
        nw = adj[u] & ~bits[v]
        w = nw
        while w:
            lb = w & (-w)
            wi = lb.bit_length() - 1
            cw = adj[v] & adj[wi]
            if popcount(cw) > 1:
                return False
            w ^= lb
        
        return True
    
    def add_edge(u, v, adj):
        adj[u] |= bits[v]
        adj[v] |= bits[u]
    
    def remove_edge(u, v, adj):
        adj[u] &= ~bits[v]
        adj[v] &= ~bits[u]
    
    def copy_adj(a):
        return a[:]
    
    def count_edges_fast(a):
        cnt = 0
        for u in range(n):
            for v in range(u+1, n):
                if a[u] & bits[v]:
                    cnt += 1
        return cnt
    
    def greedy_init():
        """Random greedy initialization"""
        adj = [0] * n
        edges_to_try = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edges_to_try)
        for u, v in edges_to_try:
            if can_add(u, v, adj):
                add_edge(u, v, adj)
        return adj
    
    def local_search(adj, deadline):
        """Local search: try to add missing edges, then do swaps"""
        current_count = count_edges_fast(adj)
        
        while time.time() < deadline:
            # Try to add any missing edge
            added = False
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & bits[v])]
            random.shuffle(missing)
            
            for u, v in missing:
                if can_add(u, v, adj):
                    add_edge(u, v, adj)
                    added = True
                    current_count += 1
            
            if not added:
                # Try 1-for-1 swaps: remove one edge, add another
                all_edges = [(u, v) for u in range(n) for v in range(u+1, n) if adj[u] & bits[v]]
                random.shuffle(all_edges)
                
                found_swap = False
                for rem_u, rem_v in all_edges:
                    if time.time() > deadline:
                        break
                    
                    remove_edge(rem_u, rem_v, adj)
                    current_count -= 1
                    
                    # Try to add a missing edge
                    missing_after = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & bits[v])]
                    random.shuffle(missing_after)
                    
                    best_add = None
                    for add_u, add_v in missing_after:
                        if can_add(add_u, add_v, adj):
                            best_add = (add_u, add_v)
                            break
                    
                    if best_add:
                        add_edge(best_add[0], best_add[1], adj)
                        current_count += 1
                        found_swap = True
                        break
                    else:
                        # Rollback
                        add_edge(rem_u, rem_v, adj)
                        current_count += 1
                
                if not found_swap:
                    # Try 1-for-2 swaps: remove one edge, add two
                    for rem_u, rem_v in all_edges:
                        if time.time() > deadline:
                            break
                        
                        remove_edge(rem_u, rem_v, adj)
                        current_count -= 1
                        
                        added_edges = []
                        missing_after = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & bits[v])]
                        random.shuffle(missing_after)
                        
                        for add_u, add_v in missing_after:
                            if len(added_edges) >= 2:
                                break
                            if can_add(add_u, add_v, adj):
                                add_edge(add_u, add_v, adj)
                                added_edges.append((add_u, add_v))
                                current_count += 1
                        
                        if len(added_edges) == 2:
                            # Net gain of 1, accept
                            break
                        else:
                            # Rollback
                            for eu, ev in added_edges:
                                remove_edge(eu, ev, adj)
                                current_count -= 1
                            add_edge(rem_u, rem_v, adj)
                            current_count += 1
                    else:
                        # No improvement possible, break
                        break
            
            # Check for improvement
            if current_count > count_edges_fast(adj):
                pass  # Already updated in place
        
        return adj, current_count
    
    # Multi-start local search
    best_adj = [0] * n
    best_count = 0
    
    # Run multiple restarts
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random greedy initialization
        adj = greedy_init()
        current_count = count_edges_fast(adj)
        
        # Local search
        adj, current_count = local_search(adj, deadline)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = copy_adj(adj)
    
    # Final polish: try to add any remaining missing edges to best_adj
    missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (best_adj[u] & bits[v])]
    random.shuffle(missing)
    for u, v in missing:
        if can_add(u, v, best_adj):
            add_edge(u, v, best_adj)
            best_count += 1
    
    # Return edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & bits[v]:
                edges.append([u, v])
    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
