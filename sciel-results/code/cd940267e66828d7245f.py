import time

N = 20
NUM_EDGES = N * (N - 1) // 2

edge_list = []
edge_index = {}
idx = 0
for u in range(N):
    for v in range(u + 1, N):
        edge_list.append((u, v))
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
        idx += 1

eu = [0] * NUM_EDGES
ev = [0] * NUM_EDGES
for i, (u, v) in enumerate(edge_list):
    eu[i] = u
    ev[i] = v

def construct():
    start_time = time.time()
    time_limit = 115
    
    best_count = 0
    best_edges_mask = 0
    
    # Global variables for B&B
    # S: current set of edges (bitmask)
    # C: candidate set of edges (bitmask)
    # common: 20x20 list of lists
    # adj: list of 20 bitmasks
    
    def can_add(common, adj, e_idx):
        u = eu[e_idx]
        v = ev[e_idx]
        
        if common[u][v] > 2:
            return False
            
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            if common[u][w] > 1:
                return False
            mask_v ^= lsb
            
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if common[v][w] > 1:
                return False
            mask_u ^= lsb
            
        return True

    def apply_edge(common, adj, e_idx, sign):
        # sign: 1 for add, -1 for remove
        u = eu[e_idx]
        v = ev[e_idx]
        
        # Update common
        # For w in N(v) (before update), common[u][w] += sign
        # We need the neighbors *before* the update.
        # If adding, adj[v] does not include u yet.
        # If removing, adj[v] includes u.
        
        # To be safe, we calculate based on current adj, but we must be careful about the edge (u,v) itself.
        # When adding (u,v):
        #   N(v) does not contain u.
        #   N(u) does not contain v.
        #   We add (u,v).
        #   New N(v) contains u. New N(u) contains v.
        #   For any w in OLD N(v), u and w now share v. So common[u][w] += 1.
        #   For any w in OLD N(u), v and w now share u. So common[v][w] += 1.
        #   common[u][v] is unchanged (count of common neighbors).
        
        # When removing (u,v):
        #   N(v) contains u.
        #   N(u) contains v.
        #   We remove (u,v).
        #   For any w in NEW N(v) (which is OLD N(v) \ {u}), u and w no longer share v. common[u][w] -= 1.
        #   For any w in NEW N(u) (which is OLD N(u) \ {v}), v and w no longer share u. common[v][w] -= 1.
        
        if sign == 1:
            mask_v = adj[v]
            while mask_v:
                lsb = mask_v & -mask_v
                w = lsb.bit_length() - 1
                common[u][w] += 1
                common[w][u] += 1
                mask_v ^= lsb
                
            mask_u = adj[u]
            while mask_u:
                lsb = mask_u & -mask_u
                w = lsb.bit_length() - 1
                common[v][w] += 1
                common[w][v] += 1
                mask_u ^= lsb
                
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        else:
            # Remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            mask_v = adj[v]
            while mask_v:
                lsb = mask_v & -mask_v
                w = lsb.bit_length() - 1
                common[u][w] -= 1
                common[w][u] -= 1
                mask_v ^= lsb
                
            mask_u = adj[u]
            while mask_u:
                lsb = mask_u & -mask_u
                w = lsb.bit_length() - 1
                common[v][w] -= 1
                common[w][v] -= 1
                mask_u ^= lsb

    def branch_and_bound(S, C, common, adj, depth):
        nonlocal best_count, best_edges_mask
        
        if time.time() - start_time > time_limit:
            return
            
        current_count = bin(S).count('1')
        if current_count > best_count:
            best_count = current_count
            best_edges_mask = S
            
        if C == 0:
            return
            
        # Upper bound: current_count + number of candidates
        # Tighter bound: Greedy coloring of conflict graph on C
        # For now, use a simple bound: current_count + C.bit_count()
        # If this is <= best_count, prune
        if current_count + C.bit_count() <= best_count:
            return
            
        # Build conflict graph on C
        # We need to pick a vertex to branch on.
        # Let's pick the first set bit in C.
        lsb = C & -C
        e_idx = lsb.bit_length() - 1
        
        # Branch 1: Include e_idx
        if can_add(common, adj, e_idx):
            # Apply
            apply_edge(common, adj, e_idx, 1)
            S_new = S | lsb
            
            # Update C: remove e_idx, and any edges that conflict with e_idx given S_new
            C_new = C & ~lsb
            # We need to check which edges in C_new are still compatible with S_new
            # This is O(|C| * cost_check)
            # To speed up, we can just filter C_new
            C_filtered = 0
            mask = C_new
            while mask:
                lsb2 = mask & -mask
                e2_idx = lsb2.bit_length() - 1
                if can_add(common, adj, e2_idx):
                    C_filtered |= lsb2
                mask ^= lsb2
                
            branch_and_bound(S_new, C_filtered, common, adj, depth + 1)
            
            # Undo
            apply_edge(common, adj, e_idx, -1)
            
        # Branch 2: Exclude e_idx
        C_excluded = C & ~lsb
        branch_and_bound(S, C_excluded, common, adj, depth + 1)

    # Initialization
    common = [[0] * N for _ in range(N)]
    adj = [0] * N
    S = 0
    C = (1 << NUM_EDGES) - 1
    
    # To improve performance, we can sort edges or use a better branching variable.
    # Let's just run the B&B.
    branch_and_bound(S, C, common, adj, 0)
    
    # Convert best_edges_mask to list of edges
    result_edges = []
    for i in range(NUM_EDGES):
        if best_edges_mask & (1 << i):
            result_edges.append([eu[i], ev[i]])
            
    return result_edges
