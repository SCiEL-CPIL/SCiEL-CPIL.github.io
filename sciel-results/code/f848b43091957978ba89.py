import time
import random
import sys

def construct():
    n = 20
    deadline = time.time() + 110.0
    
    # Bitmask representation: adj[i] is an integer bitmask of neighbors of vertex i
    adj = [0] * n
    best_adj = [0] * n
    best_edges = 0
    
    # Precompute bit for each vertex
    bits = [1 << i for i in range(n)]
    
    def popcount(x):
        return bin(x).count('1')
    
    def common_neighbors(u, v):
        return adj[u] & adj[v]
    
    def count_edges(a):
        return sum(popcount(a[u] & bits[u+1] if u < n-1 else a[u]) for u in range(n)) // 2
        # Actually simpler: sum popcount of each row / 2
        # But let's just count upper triangle
        cnt = 0
        for u in range(n):
            mask = a[u] & ~((bits[u+1] - 1) | bits[u]) if u < n-1 else a[u] & ~bits[u]
            # Simpler: for u < v
            for v in range(u+1, n):
                if a[u] & bits[v]:
                    cnt += 1
        return cnt
    
    def count_edges_fast(a):
        cnt = 0
        for u in range(n):
            for v in range(u+1, n):
                if a[u] & bits[v]:
                    cnt += 1
        return cnt
    
    def can_add(u, v, a):
        if u == v:
            return False
        if a[u] & bits[v]:
            return False
        
        # Check common neighbors of u and v: must be <= 2
        cu = a[u] & a[v]
        if popcount(cu) > 2:
            return False
        
        # Check all pairs involving u or v with their neighbors
        # For each neighbor w of v (w != u): common(u, w) must be <= 1 (since v will be added)
        nw = a[v] & ~bits[u]
        w = nw
        while w:
            lb = w & (-w)
            wi = lb.bit_length() - 1
            # common(u, wi) in current graph
            cw = a[u] & a[wi]
            if popcount(cw) > 1:
                return False
            w ^= lb
        
        # For each neighbor w of u (w != v): common(v, w) must be <= 1
        nw = a[u] & ~bits[v]
        w = nw
        while w:
            lb = w & (-w)
            wi = lb.bit_length() - 1
            cw = a[v] & a[wi]
            if popcount(cw) > 1:
                return False
            w ^= lb
        
        return True
    
    def add_edge(u, v, a):
        a[u] |= bits[v]
        a[v] |= bits[u]
    
    def remove_edge(u, v, a):
        a[u] &= ~bits[v]
        a[v] &= ~bits[u]
    
    def copy_adj(a):
        return a[:]
    
    # Initialize with Clebsch graph on vertices 0..15
    def get_rep(x):
        return min(x, x ^ 31)
    
    clebsch_edges = set()
    for u in range(32):
        for b in range(5):
            v = u ^ (1 << b)
            if u < v:
                cu = get_rep(u)
                cv = get_rep(v)
                if cu != cv:
                    clebsch_edges.add((cu, cv))
    
    for u, v in clebsch_edges:
        add_edge(u, v, adj)
    
    # Add vertices 16-19: connect each to 2 adjacent vertices in Clebsch
    cleb_edge_list = list(clebsch_edges)
    random.shuffle(cleb_edge_list)
    for i in range(4):
        new_v = 16 + i
        # Pick a random edge from Clebsch
        idx = (i * 7) % len(cleb_edge_list)
        u, v = cleb_edge_list[idx]
        add_edge(new_v, u, adj)
        add_edge(new_v, v, adj)
    
    best_adj = copy_adj(adj)
    best_edges = count_edges_fast(adj)
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Simulated annealing parameters
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    current_edges = best_edges
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Try to add a random edge
        u, v = random.choice(all_edges)
        if not (adj[u] & bits[v]):
            if can_add(u, v, adj):
                add_edge(u, v, adj)
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = copy_adj(adj)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Try 1-for-2 swap
        if random.random() < 0.5:
            # Pick a random existing edge to remove
            existing = [(u, v) for u, v in all_edges if adj[u] & bits[v]]
            if existing:
                rem_u, rem_v = random.choice(existing)
                remove_edge(rem_u, rem_v, adj)
                current_edges -= 1
                
                added = []
                # Try to add up to 2 edges
                for _ in range(2):
                    non_edges = [(u, v) for u, v in all_edges if not (adj[u] & bits[v])]
                    if not non_edges:
                        break
                    e1 = random.choice(non_edges)
                    if can_add(e1[0], e1[1], adj):
                        add_edge(e1[0], e1[1], adj)
                        added.append(e1)
                        current_edges += 1
                
                if len(added) == 2:
                    # Net gain of 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = copy_adj(adj)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # Rollback
                    for eu, ev in added:
                        remove_edge(eu, ev, adj)
                        current_edges -= 1
                    add_edge(rem_u, rem_v, adj)
                    current_edges += 1
                    no_improve_count += 1
        
        # 1-for-1 swap (exploration) with simulated annealing acceptance
        if random.random() < 0.15:
            existing = [(u, v) for u, v in all_edges if adj[u] & bits[v]]
            if existing:
                rem_u, rem_v = random.choice(existing)
                remove_edge(rem_u, rem_v, adj)
                current_edges -= 1
                
                non_edges = [(u, v) for u, v in all_edges if not (adj[u] & bits[v])]
                if non_edges:
                    add_u, add_v = random.choice(non_edges)
                    if can_add(add_u, add_v, adj):
                        add_edge(add_u, add_v, adj)
                        current_edges += 1
                        # Edge count unchanged; accept with some probability for exploration
                        # Since edge count is same, always accept to explore
                    else:
                        add_edge(rem_u, rem_v, adj)
                        current_edges += 1
                else:
                    add_edge(rem_u, rem_v, adj)
                    current_edges += 1
        
        # Periodic restart if stuck: try a different initial configuration
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            # Reset to best and perturb
            adj = copy_adj(best_adj)
            current_edges = best_edges
            # Remove a few random edges to create space
            for _ in range(random.randint(2, 5)):
                existing = [(u, v) for u, v in all_edges if adj[u] & bits[v]]
                if existing:
                    ru, rv = random.choice(existing)
                    remove_edge(ru, rv, adj)
                    current_edges -= 1
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
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
