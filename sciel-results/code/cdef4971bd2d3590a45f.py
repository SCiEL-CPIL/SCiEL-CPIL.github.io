import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for safety
    
    n = 18
    best_score = 18
    best_edges = []
    
    # Helper: Get edges from adjacency list (list of bitmasks)
    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            # Only consider j > i to avoid duplicates
            # We can iterate through set bits
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append([i, j])
                m ^= lsb
        return edges

    # Helper: Check if graph is K4-free using bitmask
    def is_k4_free_mask(adj_mask):
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask[u] >> v) & 1:
                    continue
                # Common neighbors of u and v
                common = adj_mask[u] & adj_mask[v]
                # Remove u and v from common
                common &= ~(1 << u)
                common &= ~(1 << v)
                
                # A K4 exists if any two vertices in common are adjacent.
                # Iterate through vertices w in common
                w = common
                while w:
                    # Get lowest set bit
                    lsb = w & -w
                    w ^= lsb
                    wi = lsb.bit_length() - 1
                    # Check if any other vertex in common is adjacent to wi
                    if adj_mask[wi] & (common & ~(1 << wi)):
                        return False
        return True

    # Helper: Find maximum triangle-free induced subgraph size
    # Uses bitmask Branch and Bound
    def max_triangle_free_size_mask(adj_mask):
        best = 0
        
        # Precompute for each vertex v, the set of vertices that form a triangle with v?
        # Actually, the condition for adding v to S is:
        # v must not be connected to two adjacent vertices in S.
        # Let N_v_S = adj_mask[v] & S
        # For any u in N_v_S, if adj_mask[u] & (N_v_S \ {u}) != 0, then triangle.
        
        def dfs(current_set, candidates):
            nonlocal best
            if not candidates:
                if current_set.bit_count() > best:
                    best = current_set.bit_count()
                return
            
            # Upper bound
            if current_set.bit_count() + candidates.bit_count() <= best:
                return
            
            # Pick a vertex from candidates to branch on
            # Heuristic: pick the vertex with the highest degree in the induced subgraph of candidates?
            # Or just the first one.
            v = (candidates & -candidates).bit_length() - 1
            v_mask = 1 << v
            
            # Branch 1: Exclude v
            dfs(current_set, candidates & ~v_mask)
            
            # Branch 2: Include v
            # Check if v can be added to current_set without creating a triangle.
            neighbors_in_current = adj_mask[v] & current_set
            
            valid = True
            check = neighbors_in_current
            while check:
                lsb = check & -check
                check ^= lsb
                ui = lsb.bit_length() - 1
                if adj_mask[ui] & (neighbors_in_current & ~lsb):
                    valid = False
                    break
            
            if valid:
                new_current = current_set | v_mask
                # Update candidates: remove v. 
                # We don't strictly need to prune other candidates here for correctness, 
                # just for speed. The check in the next step will handle it.
                # However, if we include v, any w in candidates that forms a triangle 
                # with v and some u in current_set cannot be added LATER.
                # Let's prune candidates to improve speed.
                # w is forbidden if (w,v) in E AND (w,u) in E AND (v,u) in E for some u in current_set.
                # This is equivalent to: w is in adj_mask[v] AND w is in (union of adj_mask[u] for u in N_v_current).
                
                forbidden = 0
                # N_v_current is neighbors of v in current_set
                # For each u in N_v_current, w must not be in adj_mask[u]
                # So w is forbidden if w is in adj_mask[v] AND w is in (OR of adj_mask[u] for u in N_v_current)
                
                # Let's compute the union of adj_mask[u] for u in N_v_current
                union_adj = 0
                check_u = neighbors_in_current
                while check_u:
                    lsb_u = check_u & -check_u
                    check_u ^= lsb_u
                    ui = lsb_u.bit_length() - 1
                    union_adj |= adj_mask[ui]
                    
                # Forbidden vertices are those in adj_mask[v] AND union_adj
                forbidden = adj_mask[v] & union_adj
                
                new_candidates = (candidates & ~v_mask) & ~forbidden
                dfs(new_current, new_candidates)
                
        dfs(0, (1 << n) - 1)
        return best

    # Initial Construction: Turan Graph T_3(18)
    # Partition vertices into 3 parts of size 6
    # Edges exist between parts, not within.
    def generate_turan_adj_mask(partition):
        adj_mask = [0]*n
        for i in range(n):
            for j in range(i+1, n):
                if partition[i] != partition[j]:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        return adj_mask

    # Initial partition: 0,1,2 for part 0; 3,4,5 for part 1; 6,7,8 for part 2... etc
    # Just assign i % 3
    initial_partition = [i % 3 for i in range(n)]
    
    adj_mask = generate_turan_adj_mask(initial_partition)
    current_score = max_triangle_free_size_mask(adj_mask)
    
    if current_score is not None:
        best_score = current_score
        best_edges = get_edges_from_adj(adj_mask)
    else:
        # Should not happen for Turan, but fallback
        return []

    # Simulated Annealing on the partition
    # State: partition list of length 18 with values 0, 1, 2
    # Move: Swap partition of two random vertices
    # Or: Move a vertex to a different partition
    
    current_partition = initial_partition[:]
    current_adj = adj_mask[:]
    current_temp = 1.0
    min_temp = 0.01
    
    # To speed up, we can precompute the effect of moves?
    # No, just recompute score if K4-free (which is always true for 3-partite)
    
    # Note: Any graph derived from a 3-partition where edges are complete between parts is K4-free.
    # So we don't need to check K4-free.
    
    improved = True
    while time.time() < deadline - 5:
        # Try a move
        u, v = random.sample(range(n), 2)
        
        # Option 1: Swap partition of u and v
        # Option 2: Change partition of u to a random other part
        
        if random.random() < 0.5:
            new_partition = current_partition[:]
            new_partition[u], new_partition[v] = new_partition[v], new_partition[u]
        else:
            new_partition = current_partition[:]
            # Change u to a random part different from current
            old_part = new_partition[u]
            new_part = random.choice([p for p in range(3) if p != old_part])
            new_partition[u] = new_part
            
        new_adj = generate_turan_adj_mask(new_partition)
        new_score = max_triangle_free_size_mask(new_adj)
        
        if new_score is None:
            continue
            
        # Acceptance criterion
        if new_score < current_score:
            current_partition = new_partition
            current_adj = new_adj
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = get_edges_from_adj(new_adj)
                improved = True
            # Cool down slightly
            current_temp = max(min_temp, current_temp * 0.99)
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if delta == 0:
                # Accept to explore
                if random.random() < 0.1:
                    current_partition = new_partition
                    current_adj = new_adj
            else:
                # delta > 0
                prob = random.random()
                if prob < (2.71828 ** (-delta / current_temp)):
                    current_partition = new_partition
                    current_adj = new_adj
                    current_score = new_score
            # Cool down
            current_temp = max(min_temp, current_temp * 0.995)
            
        # Restart temperature if stuck?
        # Simple check: if no improvement for a long time, maybe randomize?
        # For now, just continue.
        
    return best_edges

# Fallback if construct fails or returns empty
if __name__ == "__main__":
    # This block is just for local testing, not used by evaluator
    pass
