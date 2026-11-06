"""Construct C6-free graph on 40 vertices with maximum edges."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit

    n = 40

    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    n_pairs = len(all_pairs)

    # Use bitmasks for fast adjacency checks
    # adj[i] is an integer where bit j is set if edge (i,j) exists

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def has_c6(adj, u, v):
        """
        Check if adding edge (u,v) to graph with adjacency adj creates a C6.
        A C6 containing edge (u,v) has the form u-a-b-c-d-v-u
        where a is neighbor of u (not v), d is neighbor of v (not u),
        a != d, b is neighbor of a (not u, not v, not d), 
        c is neighbor of d (not u, not v, not a), and b is neighbor of c.
        
        Actually the cycle is u-a-b-c-d-v-u, so:
        - a adjacent to u
        - b adjacent to a
        - c adjacent to b  
        - d adjacent to c
        - v adjacent to d
        - u adjacent to v (the edge we're adding)
        
        So we need: a in N(u)\{v}, d in N(v)\{u}, a!=d,
        b in N(a)\{u,v,d}, c in N(d)\{u,v,a}, and b adjacent to c.
        """
        if (adj[u] >> v) & 1:
            return True  # Already an edge
        
        # Get neighbors of u and v excluding each other
        nu_mask = adj[u] & ~(1 << v)
        nv_mask = adj[v] & ~(1 << u)
        
        # Get lists of neighbors for a and d
        a_list = []
        tmp = nu_mask
        while tmp:
            a = (tmp & -tmp).bit_length() - 1
            a_list.append(a)
            tmp &= tmp - 1
            
        d_list = []
        tmp = nv_mask
        while tmp:
            d = (tmp & -tmp).bit_length() - 1
            d_list.append(d)
            tmp &= tmp - 1

        for a in a_list:
            # Precompute base mask for b (neighbors of a excluding u, v, a)
            b_base = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
            for d in d_list:
                if a == d:
                    continue
                # Exclude d from b candidates
                b_mask = b_base & ~(1 << d)
                if b_mask == 0:
                    continue
                
                # Find c: neighbor of d, not in {u,v,a,d}
                c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if c_mask == 0:
                    continue
                
                # Check if there exists b in b_mask such that b is connected to any c in c_mask
                b_tmp = b_mask
                while b_tmp:
                    b = (b_tmp & -b_tmp).bit_length() - 1
                    b_tmp &= b_tmp - 1
                    if adj[b] & c_mask:
                        return True
        return False

    def try_add_edge(adj, u, v):
        """Try to add edge (u,v). Returns True if added."""
        if not has_c6(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True
        return False

    def remove_edge(adj, u, v):
        """Remove edge (u,v)."""
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def add_edge(adj, u, v):
        """Add edge (u,v) without checking."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def get_edges(adj):
        """Get list of edges from adjacency masks."""
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def find_c6_containing_edge(adj, u, v):
        """Find a C6 containing edge (u,v) if one exists. Returns list of 6 vertices or None."""
        if (adj[u] >> v) & 1:
            nu_mask = adj[u] & ~(1 << v)
            nv_mask = adj[v] & ~(1 << u)
            
            a_list = []
            tmp = nu_mask
            while tmp:
                a = (tmp & -tmp).bit_length() - 1
                a_list.append(a)
                tmp &= tmp - 1
                
            d_list = []
            tmp = nv_mask
            while tmp:
                d = (tmp & -tmp).bit_length() - 1
                d_list.append(d)
                tmp &= tmp - 1

            for a in a_list:
                b_base = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                for d in d_list:
                    if a == d:
                        continue
                    b_mask = b_base & ~(1 << d)
                    if b_mask == 0:
                        continue
                    
                    c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                    if c_mask == 0:
                        continue
                    
                    b_tmp = b_mask
                    while b_tmp:
                        b = (b_tmp & -b_tmp).bit_length() - 1
                        b_tmp &= b_tmp - 1
                        if adj[b] & c_mask:
                            c_tmp = adj[b] & c_mask
                            c = (c_tmp & -c_tmp).bit_length() - 1
                            return [u, a, b, c, d, v]
        return None

    def simulated_annealing(adj, max_iter=50000):
        """
        Simulated annealing on the C6-free graph problem.
        Moves: add an edge, remove an edge, or swap (remove one, add another).
        Objective: maximize number of edges subject to C6-free constraint.
        """
        T = 0.5
        T_min = 0.001
        cool_rate = 0.9995
        
        current_count = count_edges(adj)
        best_adj = adj[:]
        best_count = current_count
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
                
            move_type = random.random()
            if move_type < 0.4:
                # Try to add an edge
                u, v = random.choice(all_pairs)
                if (adj[u] >> v) & 1:
                    continue
                if has_c6(adj, u, v):
                    # Can't add this edge - it would create a C6
                    # But in SA, we might still want to consider it with penalty
                    # For simplicity, skip invalid moves
                    continue
                new_count = current_count + 1
                delta = 1
                if delta > 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                else:
                    if random.random() < min(1.0, delta / max(T, 1e-10)):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        current_count = new_count
            elif move_type < 0.7:
                # Try to remove an edge
                u, v = random.choice(all_pairs)
                if not ((adj[u] >> v) & 1):
                    continue
                new_count = current_count - 1
                delta = -1
                if delta > 0 or random.random() < min(1.0, delta / max(T, 1e-10)):
                    remove_edge(adj, u, v)
                    current_count = new_count
            else:
                # Swap: remove one edge, try to add another
                u1, v1 = random.choice(all_pairs)
                if not ((adj[u1] >> v1) & 1):
                    continue
                u2, v2 = random.choice(all_pairs)
                if (adj[u2] >> v2) & 1:
                    continue
                # Remove u1-v1 first
                remove_edge(adj, u1, v1)
                # Try to add u2-v2
                if not has_c6(adj, u2, v2):
                    add_edge(adj, u2, v2)
                    # Count stays the same
                else:
                    # Can't add, put back u1-v1
                    add_edge(adj, u1, v1)
            
            # Cool down
            T = max(T * cool_rate, T_min)
        
        return best_adj, best_count

    def local_search(adj, max_iter=5000):
        """
        Local search: iteratively try to add any missing edge that doesn't create C6.
        Uses a more aggressive approach with multiple passes.
        """
        improved = True
        pass_count = 0
        while improved and pass_count < max_iter:
            improved = False
            pass_count += 1
            shuffled = all_pairs[:]
            random.shuffle(shuffled)
            for u, v in shuffled:
                if time.time() > deadline:
                    return adj
                if not ((adj[u] >> v) & 1):
                    if try_add_edge(adj, u, v):
                        improved = True
        return adj

    def greedy_construct(seed_offset=0):
        """Greedy construction with random ordering."""
        random.seed(12345 + seed_offset)
        adj = [0] * n
        shuffled = all_pairs[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if time.time() > deadline:
                break
            try_add_edge(adj, u, v)
        
        return adj

    def perturb_and_recover(best_adj):
        """Perturb the best solution and try to recover/improve."""
        adj = best_adj[:]
        
        # Remove k random edges
        k = random.randint(2, 6)
        current_edges = get_edges(adj)
        if len(current_edges) > k:
            removed = random.sample(current_edges, k)
            for u, v in removed:
                remove_edge(adj, u, v)
        
        # Try to re-add edges greedily
        shuffled = all_pairs[:]
        random.shuffle(shuffled)
        for u, v in shuffled:
            if time.time() > deadline:
                break
            if not ((adj[u] >> v) & 1):
                try_add_edge(adj, u, v)
        
        # Local search
        adj = local_search(adj)
        
        return adj

    random.seed(42)
    best_adj = None
    best_count = 0
    
    # Phase 1: Multiple random restarts with greedy construction + local search
    n_restarts = 80
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        adj = greedy_construct(restart)
        adj = local_search(adj)
        
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
        
        if best_count >= 105:
            if restart >= 10:
                break
    
    # Phase 2: Simulated annealing from best solution
    if best_adj is not None:
        sa_adj, sa_count = simulated_annealing(best_adj[:], max_iter=30000)
        if sa_count > best_count:
            best_count = sa_count
            best_adj = sa_adj
    
    # Phase 3: More perturbations from best
    if best_adj is not None:
        for perturb_round in range(40):
            if time.time() > deadline:
                break
            
            adj = perturb_and_recover(best_adj)
            current_count = count_edges(adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
    
    # Final local search
    if best_adj is not None:
        best_adj = local_search(best_adj)
        best_count = count_edges(best_adj)
    
    return get_edges(best_adj)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Edges: {len(edges)}, Time: {t1-t0:.2f}s")
