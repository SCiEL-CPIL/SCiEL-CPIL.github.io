"""Combined program: set-based edges from inspiration, diverse seeds + SA from current."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill
    
    n = 40
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def would_create_c6(adj, u, v):
        """Check if adding edge (u, v) creates a C6 using bit operations."""
        Nu = adj[u]
        Nv = adj[v]
        
        a_mask = Nu & ~(1 << v)
        while a_mask:
            lsb_a = a_mask & -a_mask
            a = lsb_a.bit_length() - 1
            a_mask ^= lsb_a
            
            d_mask = Nv & ~(1 << u) & ~(1 << a)
            while d_mask:
                lsb_d = d_mask & -d_mask
                d = lsb_d.bit_length() - 1
                d_mask ^= lsb_d
                
                Na = adj[a]
                Nd = adj[d]
                
                b_mask = Na & ~(1 << u) & ~(1 << v) & ~(1 << d)
                c_mask = Nd & ~(1 << u) & ~(1 << v) & ~(1 << a)
                
                # Iterate over the smaller mask for efficiency
                if b_mask.bit_count() > c_mask.bit_count():
                    b_mask, c_mask = c_mask, b_mask
                
                b_temp = b_mask
                while b_temp:
                    lsb_b = b_temp & -b_temp
                    b = lsb_b.bit_length() - 1
                    b_temp ^= lsb_b
                    
                    if adj[b] & c_mask:
                        return True
        
        return False
    
    def build_adj(edges_set):
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def init_seed(restart_idx):
        """Create diverse initial seeds."""
        edges = set()
        
        if restart_idx % 5 == 0:
            # 8 K5s
            for k in range(8):
                for a in range(5):
                    for b in range(a + 1, 5):
                        u = 5 * k + a
                        v = 5 * k + b
                        edges.add((u, v))
        elif restart_idx % 5 == 1:
            # Random sparse graph
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.08:
                        edges.add((u, v))
        elif restart_idx % 5 == 2:
            # Turan-like: partition into 5 groups of 8, connect within groups
            group_size = 8
            for g in range(5):
                for a in range(group_size):
                    for b in range(a + 1, group_size):
                        u = g * group_size + a
                        v = g * group_size + b
                        edges.add((u, v))
        elif restart_idx % 5 == 3:
            # Random + some structure
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.12:
                        edges.add((u, v))
            # Add a few K4s
            for k in range(5):
                base = k * 8
                for a in range(4):
                    for b in range(a + 1, 4):
                        u = base + a
                        v = base + b
                        if (u, v) not in edges and (v, u) not in edges:
                            edges.add((min(u,v), max(u,v)))
        else:
            # Medium density random
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.18:
                        edges.add((u, v))
        
        return edges
    
    def greedy_add(adj, edges_set):
        """Add all possible non-C6-creating edges in random order."""
        non_edges = []
        for u, v in all_possible_edges:
            if not (adj[u] >> v) & 1:
                non_edges.append((u, v))
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > deadline - 3.0:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges_set.add((u, v))
        
        return edges_set
    
    def local_search_swap(adj, edges_set):
        """Local search: remove 1 edge, try to add 2 new edges (swap improvement)."""
        current_edges = set(edges_set)
        
        consecutive_failures = 0
        max_failures = 30
        
        while time.time() < deadline - 3.0:
            if consecutive_failures > max_failures:
                break
            
            # Pick a random edge to remove
            edge_list = list(current_edges)
            if not edge_list:
                break
            
            e_rem = random.choice(edge_list)
            u_rem, v_rem = e_rem
            
            # Save state
            old_adj = adj[:]
            old_edges = set(current_edges)
            
            # Remove edge
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_edges.discard(e_rem)
            
            # Try to add 2 non-edges
            non_edges = []
            for u, v in all_possible_edges:
                if not (adj[u] >> v) & 1:
                    non_edges.append((u, v))
            random.shuffle(non_edges)
            
            added = 0
            for u, v in non_edges:
                if added >= 2:
                    break
                if time.time() > deadline - 3.0:
                    break
                if (u, v) in current_edges:
                    continue
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    added += 1
            
            if added >= 2:
                consecutive_failures = 0
            else:
                # Revert
                adj = old_adj
                current_edges = old_edges
                consecutive_failures += 1
        
        return current_edges
    
    def simulated_annealing(adj, edges_set):
        """Simulated Annealing with structured moves."""
        current_edges = set(edges_set)
        
        T = 3.0
        cooling = 0.997
        min_T = 0.05
        max_steps = 3000
        
        for step in range(max_steps):
            if time.time() > deadline - 3.0:
                break
            if T < min_T:
                break
                
            # Randomly remove 1-2 edges
            k_remove = random.choice([1, 1, 2])
            edge_list = list(current_edges)
            if len(edge_list) < k_remove:
                break
            indices = random.sample(range(len(edge_list)), k_remove)
            
            removed = []
            for idx in sorted(indices, reverse=True):
                u, v = edge_list[idx]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                removed.append((u, v))
                current_edges.discard((u, v))
            
            new_count = len(current_edges)
            
            # Try to add edges
            added_edges = []
            attempts = 0
            max_add_attempts = 30
            
            while attempts < max_add_attempts and len(added_edges) < 3:
                ru = random.randint(0, n - 1)
                rv = random.randint(0, n - 1)
                if ru == rv:
                    attempts += 1
                    continue
                if ru > rv:
                    ru, rv = rv, ru
                    
                if (adj[ru] >> rv) & 1:
                    attempts += 1
                    continue
                    
                if not would_create_c6(adj, ru, rv):
                    adj[ru] |= (1 << rv)
                    adj[rv] |= (1 << ru)
                    current_edges.add((ru, rv))
                    added_edges.append((ru, rv))
                    new_count += 1
                attempts += 1
            
            delta = new_count - (new_count - len(added_edges) + k_remove)
            # delta = len(added_edges) - k_remove
            delta = len(added_edges) - k_remove
            
            if delta >= 0 or random.random() < pow(2.718, delta / T):
                pass  # Accept
            else:
                # Revert
                for a, b in added_edges:
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    current_edges.discard((a, b))
                for u, v in removed:
                    current_edges.add((u, v))
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            T *= cooling
        
        return current_edges
    
    def final_polish(adj, edges_set):
        """Final high-precision polish: try all non-edges systematically."""
        current_edges = set(edges_set)
        
        # Multiple passes of greedy
        for _ in range(3):
            non_edges = []
            for u, v in all_possible_edges:
                if not (adj[u] >> v) & 1:
                    non_edges.append((u, v))
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() > deadline - 1.0:
                    return current_edges
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
        
        return current_edges
    
    # Main loop
    best_edges = set()
    best_count = 0
    
    num_restarts = 40
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize with diverse seed
        current_edges = init_seed(restart)
        adj = build_adj(current_edges)
        
        # Greedy phase
        current_edges = greedy_add(adj, current_edges)
        
        # Local search phase
        if time.time() < deadline - 8.0:
            current_edges = local_search_swap(adj, current_edges)
        
        # SA phase
        if time.time() < deadline - 5.0:
            current_edges = simulated_annealing(adj, current_edges)
        
        # Final polish
        if time.time() < deadline - 2.0:
            current_edges = final_polish(adj, current_edges)
        
        current_count = len(current_edges)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = set(current_edges)
    
    # Return as list of [u, v]
    return [[u, v] for u, v in best_edges]
