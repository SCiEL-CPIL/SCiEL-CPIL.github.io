import time
import random
import math

def construct():
    n = 20
    start_time = time.time()
    time_limit = 118.0  # Safety margin

    # Bitset helpers
    def popcount(x):
        return bin(x).count('1')

    def count_edges(adj):
        return sum(popcount(x) for x in adj) // 2

    def is_k23_free(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj[u] & adj[v]) >= 3:
                    return False
        return True

    def valid_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        if adj[u] & (1 << v):
            return False
        
        # 1. u and v must have at most 2 common neighbors
        if popcount(adj[u] & adj[v]) > 2:
            return False
        
        # 2. For each w in N(u), w and v must have at most 1 common neighbor
        # Note: When we add (u,v), v becomes a neighbor of u.
        # For a fixed w in N(u) (w != v):
        # New N(u) = N(u) old U {v}
        # N(w) is unchanged.
        # Common neighbors of w and u: (N(w) & N(u)_old) U (N(w) & {v})
        # If v in N(w), then v is a new common neighbor.
        # So we need | (N(w) & N(u)_old) U (N(w) & {v}) | <= 2
        # This is |N(w) & N(u)_old| + (1 if v in N(w) else 0) <= 2
        
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if w == v: continue
            common_wu = popcount(adj[w] & adj[u])
            # v is in N(w) if adj[w] & (1 << v)
            if adj[w] & (1 << v):
                if common_wu + 1 > 2:
                    return False
            else:
                if common_wu > 2:
                    return False
        
        # 3. Symmetric check for N(v)
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if w == u: continue
            common_wv = popcount(adj[w] & adj[v])
            if adj[w] & (1 << u):
                if common_wv + 1 > 2:
                    return False
            else:
                if common_wv > 2:
                    return False
        
        return True

    def valid_swap(adj, u1, v1, u2, v2):
        """Check if removing (u1,v1) and adding (u2,v2) keeps graph K2,3-free."""
        if adj[u1] & (1 << v1):
            pass # It is an edge
        else:
            return False
        if adj[u2] & (1 << v2):
            return False # It is already an edge
        
        # Remove (u1, v1)
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        
        # Check if (u2, v2) can be added
        res = valid_add(adj, u2, v2)
        
        # Revert removal
        adj[u1] |= (1 << v1)
        adj[v1] |= (1 << u1)
        
        return res

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_edges(adj):
        edges = []
        for u in range(n):
            temp = adj[u]
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append((u, v))
                temp -= lsb
        return edges

    def get_non_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    edges.append((u, v))
        return edges

    def greedy_fill(adj):
        """Greedily add all possible edges in random order."""
        non_edges = get_non_edges(adj)
        random.shuffle(non_edges)
        for u, v in non_edges:
            if valid_add(adj, u, v):
                add_edge(adj, u, v)

    # Initial seed: 5 disjoint K4s
    def make_k4_union():
        adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    add_edge(adj, base + a, base + b)
        return adj

    best_edges = 0
    best_adj = None

    # First, establish baseline
    adj0 = make_k4_union()
    greedy_fill(adj0)
    best_edges = count_edges(adj0)
    best_adj = adj0[:]

    # Multi-restart simulated annealing
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break

        # Different starting points
        if r == 0:
            adj = make_k4_union()
        elif r % 5 == 0:
            # Start from best found, then perturb
            adj = best_adj[:]
            edges = get_edges(adj)
            random.shuffle(edges)
            for u, v in edges[:4]:
                remove_edge(adj, u, v)
        else:
            # Random regular-ish graph
            adj = [0] * n
            # Try to add random edges until stuck
            for _ in range(100):
                non_edges = get_non_edges(adj)
                if not non_edges: break
                u, v = random.choice(non_edges)
                if valid_add(adj, u, v):
                    add_edge(adj, u, v)

        greedy_fill(adj)
        current_edges = count_edges(adj)

        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

        # Simulated annealing
        T = 1.0
        T_min = 0.05
        cooling = 0.999
        steps = 0

        while T > T_min and (time.time() - start_time < time_limit):
            steps += 1
            move_type = random.random()

            if move_type < 0.3:
                # Try to add a random non-edge
                non_edges = get_non_edges(adj)
                if not non_edges:
                    T *= cooling
                    continue
                u, v = random.choice(non_edges)
                if valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                # else: reject

            elif move_type < 0.5:
                # Try to remove a random edge (accept with SA probability)
                edges = get_edges(adj)
                if not edges:
                    T *= cooling
                    continue
                u, v = random.choice(edges)
                # Delta E = -1 (worse), accept with prob exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    remove_edge(adj, u, v)
                    current_edges -= 1

            else:
                # Swap: remove one edge, try to add another
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or not non_edges:
                    T *= cooling
                    continue
                
                u1, v1 = random.choice(edges)
                u2, v2 = random.choice(non_edges)
                
                # Check validity of swap
                if valid_swap(adj, u1, v1, u2, v2):
                    # Perform swap
                    remove_edge(adj, u1, v1)
                    add_edge(adj, u2, v2)
                    # Edge count unchanged, but structure changed
                    # Accept with high probability to explore
                    if random.random() < 0.9:
                        pass # keep
                    else:
                        # Revert
                        remove_edge(adj, u2, v2)
                        add_edge(adj, u1, v1)
                # else: reject

            T *= cooling

    # Final polish: systematic 2-opt and greedy
    if best_adj:
        greedy_fill(best_adj)
        final_edges = count_edges(best_adj)
        if final_edges > best_edges:
            best_edges = final_edges
            best_adj = adj[:]

        # Systematic polish: try all possible single edge additions
        changed = True
        while changed:
            changed = False
            non_edges = get_non_edges(best_adj)
            for u, v in non_edges:
                if time.time() - start_time > time_limit:
                    break
                if valid_add(best_adj, u, v):
                    add_edge(best_adj, u, v)
                    final_edges += 1
                    changed = True
            if time.time() - start_time > time_limit:
                break

        # Systematic polish: try 2-edge swaps (remove one, add one) to escape local optima
        # We do a few passes of this
        for _ in range(10):
            if time.time() - start_time > time_limit:
                break
            improved = False
            edges = get_edges(best_adj)
            non_edges = get_non_edges(best_adj)
            
            # To speed up, we might not check ALL pairs, but for N=20, 190 edges max, 190 non-edges.
            # 190*190 = 36100 checks. Each check is O(N). 36100*20 = 722,000 ops. Fast.
            for u1, v1 in edges:
                if time.time() - start_time > time_limit:
                    break
                for u2, v2 in non_edges:
                    if time.time() - start_time > time_limit:
                        break
                    if valid_swap(best_adj, u1, v1, u2, v2):
                        # Perform swap
                        remove_edge(best_adj, u1, v1)
                        add_edge(best_adj, u2, v2)
                        # Check if this leads to further improvements via greedy
                        save_adj = best_adj[:]
                        greedy_fill(save_adj)
                        save_edges = count_edges(save_adj)
                        if save_edges > final_edges:
                            best_adj = save_adj
                            final_edges = save_edges
                            improved = True
                            break
                        else:
                            # Revert
                            remove_edge(best_adj, u2, v2)
                            add_edge(best_adj, u1, v1)
                if improved:
                    break
            if not improved:
                break

        if final_edges > best_edges:
            best_edges = final_edges

    # Convert to list of edges
    result = []
    for u, v in get_edges(best_adj):
        result.append([u, v])
    return result
