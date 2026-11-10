import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = 20
    total_possible = n * (n - 1) // 2
    
    def popcount(x):
        return bin(x).count('1')
        
    def can_add(adj, common, u, v):
        # Check if adding edge (u,v) keeps graph K2,3-free
        if (adj[u] >> v) & 1:
            return False
            
        # 1. Pair (u,v) common neighbors must be <= 2
        if common[u][v] > 2:
            return False
            
        # 2. For all w in N(v), pair (u,w) gains v as common neighbor.
        #    So current common(u,w) must be <= 1.
        w = adj[v]
        while w:
            lsb = w & (-w)
            w ^= lsb
            wi = lsb.bit_length() - 1
            if common[u][wi] > 1:
                return False
                
        # 3. For all w in N(u), pair (v,w) gains u as common neighbor.
        #    So current common(v,w) must be <= 1.
        w = adj[u]
        while w:
            lsb = w & (-w)
            w ^= lsb
            wi = lsb.bit_length() - 1
            if common[v][wi] > 1:
                return False
                
        return True

    def add_edge(adj, common, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        # Update common neighbor counts
        common[u][v] += 1
        common[v][u] += 1
        # For each neighbor of u (excluding v), common count with v increases by 1
        w = adj[u] & ~(1 << v)
        while w:
            lsb = w & (-w)
            w ^= lsb
            wi = lsb.bit_length() - 1
            common[v][wi] += 1
            common[wi][v] += 1
        # For each neighbor of v (excluding u), common count with u increases by 1
        w = adj[v] & ~(1 << u)
        while w:
            lsb = w & (-w)
            w ^= lsb
            wi = lsb.bit_length() - 1
            common[u][wi] += 1
            common[wi][u] += 1
            
    def remove_edge(adj, common, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        # Update common neighbor counts
        common[u][v] -= 1
        common[v][u] -= 1
        # For each neighbor of u (excluding v), common count with v decreases by 1
        w = adj[u]
        while w:
            lsb = w & (-w)
            w ^= lsb
            wi = lsb.bit_length() - 1
            common[v][wi] -= 1
            common[wi][v] -= 1
        # For each neighbor of v (excluding u), common count with u decreases by 1
        w = adj[v]
        while w:
            lsb = w & (-w)
            w ^= lsb
            wi = lsb.bit_length() - 1
            common[u][wi] -= 1
            common[wi][u] -= 1
            
    def count_edges(adj):
        return sum(popcount(adj[i]) for i in range(n)) // 2
        
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v_mask = adj[u]
            while v_mask:
                lsb = v_mask & (-v_mask)
                v_mask ^= lsb
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
        return edges

    def init_common(adj):
        common = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                c = popcount(adj[u] & adj[v])
                common[u][v] = c
                common[v][u] = c
        return common

    def greedy_build(adj, common):
        # Randomly add edges while maintaining K2,3-free property
        order = list(range(n * (n - 1) // 2))
        random.shuffle(order)
        # Map index to (u,v) pair
        pair_map = {}
        idx = 0
        for u in range(n):
            for v in range(u + 1, n):
                pair_map[idx] = (u, v)
                idx += 1
                
        for i in order:
            if time.time() > deadline:
                break
            u, v = pair_map[i]
            if can_add(adj, common, u, v):
                add_edge(adj, common, u, v)

    def local_search(adj, common, max_iters=10000):
        # Try to improve by adding edges
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try to add a random missing edge
            for _ in range(200):
                if time.time() > deadline:
                    return adj, common
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if (adj[u] >> v) & 1:
                    continue
                if can_add(adj, common, u, v):
                    add_edge(adj, common, u, v)
                    improved = True
                    break
        return adj, common

    def simulated_annealing(adj, common, max_iters=30000, initial_temp=1.0, cooling=0.9998):
        # Simulated annealing: accept worse solutions with probability exp(-delta/temp)
        # Moves: add edge, remove edge, or swap (remove one, add another)
        current_adj = adj[:]
        current_common = [row[:] for row in common]
        current_edges = count_edges(current_adj)
        best_adj = current_adj[:]
        best_common = [row[:] for row in current_common]
        best_edges = current_edges
        temp = initial_temp
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to add a random missing edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if (current_adj[u] >> v) & 1:
                    continue
                if can_add(current_adj, current_common, u, v):
                    add_edge(current_adj, current_common, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                        best_common = [row[:] for row in current_common]
                        
            elif move_type < 0.75:
                # Try to remove a random edge (to escape local optima)
                # Pick a random existing edge
                all_edges = []
                for u in range(n):
                    v_mask = current_adj[u]
                    while v_mask:
                        lsb = v_mask & (-v_mask)
                        v_mask ^= lsb
                        v = lsb.bit_length() - 1
                        if u < v:
                            all_edges.append((u, v))
                if not all_edges:
                    continue
                u, v = random.choice(all_edges)
                remove_edge(current_adj, current_common, u, v)
                new_edges = current_edges - 1
                # Accept with probability exp(-1/T)
                if random.random() < math.exp(-1.0 / temp):
                    current_edges = new_edges
                else:
                    add_edge(current_adj, current_common, u, v)
                    
            else:
                # Swap: remove one edge, try to add another
                # Pick a random existing edge to remove
                all_edges = []
                for u in range(n):
                    v_mask = current_adj[u]
                    while v_mask:
                        lsb = v_mask & (-v_mask)
                        v_mask ^= lsb
                        v = lsb.bit_length() - 1
                        if u < v:
                            all_edges.append((u, v))
                if not all_edges:
                    continue
                u_rem, v_rem = random.choice(all_edges)
                
                # Pick a random missing edge to add
                u_add = random.randint(0, n - 1)
                v_add = random.randint(0, n - 1)
                if u_add == v_add or (u_add == u_rem and v_add == v_rem) or (u_add == v_rem and v_add == u_rem):
                    continue
                if (current_adj[u_add] >> v_add) & 1:
                    continue
                    
                # Remove the first edge
                remove_edge(current_adj, current_common, u_rem, v_rem)
                
                # Check if we can add the second edge
                if can_add(current_adj, current_common, u_add, v_add):
                    add_edge(current_adj, current_common, u_add, v_add)
                    # Net change: 0 edges.
                    # Accept swap with probability 0.5 to explore
                    if random.random() < 0.5:
                        pass
                    else:
                        remove_edge(current_adj, current_common, u_add, v_add)
                        add_edge(current_adj, current_common, u_rem, v_rem)
                else:
                    # Revert
                    add_edge(current_adj, current_common, u_rem, v_rem)
                    
            temp *= cooling
            if temp < 0.001:
                temp = 0.001
                
        return best_adj, best_common, best_edges

    def paley_seed():
        # Construct a Paley graph on 19 vertices (q=19, p≡3 mod 4)
        # Vertex 19 is connected to all others
        adj = [0] * n
        common = [[0]*n for _ in range(n)]
        
        # Paley graph on 19 vertices: connect u,v if (v-u) is a quadratic residue mod 19
        # Quadratic residues mod 19: 1,4,5,6,7,9,11,16,17
        qr = set()
        for x in range(1, 19):
            qr.add((x*x) % 19)
        
        for u in range(19):
            for v in range(u+1, 19):
                diff = (v - u) % 19
                if diff in qr or (19 - diff) in qr:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Connect vertex 19 to all others
        for u in range(19):
            adj[19] |= (1 << u)
            adj[u] |= (1 << 19)
        
        common = init_common(adj)
        return adj, common

    # Multi-start approach with simulated annealing
    best_adj = [0] * n
    best_common = [[0]*n for _ in range(n)]
    best_edges = 0
    
    # Start with Paley seed
    adj0, common0 = paley_seed()
    adj0, common0 = local_search(adj0, common0)
    sa_adj, sa_common, sa_edges = simulated_annealing(adj0, common0)
    
    if sa_edges > best_edges:
        best_edges = sa_edges
        best_adj = sa_adj[:]
        best_common = [row[:] for row in sa_common]
    
    # Additional random starts
    num_starts = 20
    for s in range(num_starts):
        if time.time() > deadline:
            break
            
        adj = [0] * n
        common = [[0]*n for _ in range(n)]
        greedy_build(adj, common)
        adj, common = local_search(adj, common)
        
        # Apply simulated annealing
        sa_adj, sa_common, sa_edges = simulated_annealing(adj, common)
        
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_adj = sa_adj[:]
            best_common = [row[:] for row in sa_common]
            
    # Final exhaustive polish: try all possible edge additions
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if time.time() > deadline - 2:
                        break
                    if (best_adj[u] >> v) & 1:
                        continue
                    if can_add(best_adj, best_common, u, v):
                        add_edge(best_adj, best_common, u, v)
                        best_edges += 1
                        improved = True
                if time.time() > deadline - 2:
                    break
                    
    # Convert best_adj to edge list
    return adj_to_edges(best_adj)
# EVOLVE-BLOCK-END
