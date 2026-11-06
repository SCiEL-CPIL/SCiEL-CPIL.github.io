import time
import random

def construct():
    """
    Constructs a C6-free graph on 40 vertices with a high number of edges.
    Uses structured initialization with K5 cliques, greedy filling, 
    and local search with 2-add 1-remove moves.
    """
    N = 40
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit

    def check_c6_fast(u, v, adj):
        """
        Check if adding edge (u,v) creates a C6.
        C6: u-a-x-y-b-v.
        For a in N(u), b in N(v), a!=b, b!=u, a!=v:
          Need x in N(a) \ {u,v,b} and y in N(b) \ {u,v,a} such that (x,y) is an edge.
        """
        nu = adj[u]
        nv = adj[v]
        
        # a in N(u), a != v
        a_mask = nu & ~(1 << v)
        while a_mask:
            a_bit = a_mask & (-a_mask)
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            # b in N(v), b != u, b != a
            b_mask = nv & ~(1 << u) & ~(1 << a)
            while b_mask:
                b_bit = b_mask & (-b_mask)
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                # x in N(a) \ {u,v,b}
                x_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << b)
                if not x_mask:
                    continue
                # y in N(b) \ {u,v,a}
                y_mask = adj[b] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if not y_mask:
                    continue
                
                # Check if any edge (x,y) with x in x_mask, y in y_mask, x != y
                xm = x_mask
                while xm:
                    x_bit = xm & (-xm)
                    x = x_bit.bit_length() - 1
                    xm ^= x_bit
                    if adj[x] & (y_mask & ~(1 << x)):
                        return True
        
        return False

    def structured_init():
        """Initialize with 8 K5 cliques."""
        adj = [0] * N
        edge_set = set()
        
        # 8 groups of 5, connect within groups as K5
        for g in range(8):
            group = list(range(g*5, g*5+5))
            for i in range(5):
                for j in range(i+1, 5):
                    u, v = group[i], group[j]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
        
        return edge_set, adj

    def greedy_fill(edge_set, adj, rng):
        """Greedily add edges in random order."""
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        rng.shuffle(all_edges)
        
        for u, v in all_edges:
            if (u, v) in edge_set:
                continue
            if not check_c6_fast(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
        
        return edge_set, adj

    def local_search(edge_set, adj, rng, max_iters=2000):
        """
        Local search with:
        1. Try to add any non-edge.
        2. If stuck, try 2-add 1-remove move.
        """
        current_edges = set(edge_set)
        current_adj = list(adj)
        current_count = len(current_edges)
        no_improve = 0
        max_no_improve = 100
        
        for it in range(max_iters):
            if time.time() >= deadline:
                break
            
            improved = False
            
            # 1. Try to add a random non-edge
            # We can also try all non-edges if count is low, but random is faster
            added = False
            for _ in range(10): # Try a few random additions
                u, v = rng.randint(0, N-1), rng.randint(0, N-1)
                if u == v:
                    continue
                u, v = min(u,v), max(u,v)
                if (u, v) in current_edges:
                    continue
                if not check_c6_fast(u, v, current_adj):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    current_count += 1
                    improved = True
                    added = True
                    no_improve = 0
                    break
            
            if improved:
                continue
            
            no_improve += 1
            if no_improve > max_no_improve:
                break
            
            # 2. Try 2-add 1-remove move
            # Remove e1, add e2 and e3
            if current_count == 0:
                break
            
            # Sample a random edge to remove
            e_remove = rng.choice(list(current_edges))
            u1, v1 = e_remove
            
            # Temporarily remove e1
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            current_edges.discard(e_remove)
            
            # Try to add two edges
            added_count = 0
            added_edges = []
            
            for _ in range(50): # Try to find 2 edges to add
                u2, v2 = rng.randint(0, N-1), rng.randint(0, N-1)
                if u2 == v2:
                    continue
                u2, v2 = min(u2,v2), max(u2,v2)
                if (u2, v2) == (u1, v1) or (u2, v2) in current_edges:
                    continue
                
                if not check_c6_fast(u2, v2, current_adj):
                    current_adj[u2] |= (1 << v2)
                    current_adj[v2] |= (1 << u2)
                    current_edges.add((u2, v2))
                    added_edges.append((u2, v2))
                    added_count += 1
                    if added_count == 2:
                        break
            
            if added_count >= 2:
                # We removed 1, added 2. Net gain +1.
                current_count += added_count - 1
                no_improve = 0
                improved = True
            else:
                # Revert: remove added edges, re-add e1
                for ue, ve in added_edges:
                    current_adj[ue] &= ~(1 << ve)
                    current_adj[ve] &= ~(1 << ue)
                    current_edges.discard((ue, ve))
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
                current_edges.add(e_remove)
        
        return current_edges, current_adj, current_count

    # Phase 1: Structured Init + Greedy Fill
    edge_set, adj = structured_init()
    rng = random.Random(42)
    edge_set, adj = greedy_fill(edge_set, adj, rng)
    best_count = len(edge_set)
    best_edges = list(edge_set)
    best_adj = list(adj)
    
    # Phase 2: Local Search
    if time.time() < deadline:
        new_edges, new_adj, new_count = local_search(edge_set, adj, rng, max_iters=5000)
        if new_count > best_count:
            best_count = new_count
            best_edges = list(new_edges)
            best_adj = list(new_adj)
    
    # Phase 3: Restart with different seeds if time permits
    for seed in range(10):
        if time.time() >= deadline:
            break
        
        # Restart with slightly perturbed structured init
        # e.g., randomize the clique assignments or start with random edges
        # For now, just re-run local search with new RNG on the current best
        # Or, try a fresh greedy with different seed on top of K5s
        
        rng2 = random.Random(seed * 1000 + 7)
        # Re-initialize adj/edges from best to avoid drift, or just perturb best?
        # Let's try to improve the current best further with a new RNG stream
        new_edges, new_adj, new_count = local_search(best_edges, best_adj, rng2, max_iters=2000)
        
        if new_count > best_count:
            best_count = new_count
            best_edges = list(new_edges)
            best_adj = list(new_adj)
            
            # If we improved, we might want to do a greedy fill again on the new state?
            # The local search handles additions.
    
    # Final polish: try to add any remaining edges
    if best_edges and time.time() < deadline:
        adj = [0] * N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Try all non-edges one last time
        for u in range(N):
            for v in range(u+1, N):
                if (u,v) not in best_edges and not check_c6_fast(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    best_edges.append((u, v))
                    best_count += 1

    return [[u, v] for u, v in best_edges]
