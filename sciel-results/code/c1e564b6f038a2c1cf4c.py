import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Fast validity checker using bitmasks
    def is_valid_bitmask(n, adj_masks):
        """Check if graph with adjacency masks has no K4 and no independent set of 6."""
        # Check for K4: for each pair (i,j) that is an edge, check if they share 2+ common neighbors
        for i in range(n):
            for j in range(i+1, n):
                if not (adj_masks[i] >> j) & 1:
                    continue
                # Common neighbors of i and j
                common = adj_masks[i] & adj_masks[j]
                # Check if any two common neighbors are adjacent
                common_bits = []
                for k in range(n):
                    if k != i and k != j and (common >> k) & 1:
                        common_bits.append(k)
                for a_idx in range(len(common_bits)):
                    for b_idx in range(a_idx+1, len(common_bits)):
                        a, b = common_bits[a_idx], common_bits[b_idx]
                        if (adj_masks[a] >> b) & 1:
                            return False  # Found K4: i, j, a, b
        # Check for independent set of 6: K6 in complement
        # Complement adjacency: ~adj_mask with self-loops removed
        comp_masks = []
        for i in range(n):
            comp = (~adj_masks[i]) & ((1 << n) - 1)
            comp &= ~(1 << i)  # Remove self-loop
            comp_masks.append(comp)
        for i in range(n):
            for j in range(i+1, n):
                if not (comp_masks[i] >> j) & 1:
                    continue
                common = comp_masks[i] & comp_masks[j]
                common_bits = []
                for k in range(n):
                    if k != i and k != j and (common >> k) & 1:
                        common_bits.append(k)
                for a_idx in range(len(common_bits)):
                    for b_idx in range(a_idx+1, len(common_bits)):
                        a, b = common_bits[a_idx], common_bits[b_idx]
                        if not ((comp_masks[a] >> b) & 1):
                            continue
                        # Check if a and b have 2 more common neighbors in complement
                        common_ab = comp_masks[a] & comp_masks[b]
                        common_ab_bits = []
                        for c in range(n):
                            if c not in (i, j, a, b) and (common_ab >> c) & 1:
                                common_ab_bits.append(c)
                        for c_idx in range(len(common_ab_bits)):
                            for d_idx in range(c_idx+1, len(common_ab_bits)):
                                c, d = common_ab_bits[c_idx], common_ab_bits[d_idx]
                                if (comp_masks[c] >> d) & 1:
                                    return False  # Found independent set of 6
        return True
    
    def edges_from_masks(n, adj_masks):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    # Start with a small valid graph and grow it
    best_n = 1
    best_adj = [0]
    
    # Try to build up from n=1 to n=35
    for target_n in range(2, 36):
        if time.time() > deadline - 10.0:
            break
        
        found = False
        # Try multiple strategies to extend the graph
        for strategy in range(20):
            if time.time() > deadline - 5.0:
                break
            
            if strategy == 0:
                # Greedy: add vertex with max degree that doesn't create K4 or Ind6
                # Start from best_adj if possible
                if len(best_adj) < target_n - 1:
                    continue
                adj = list(best_adj) + [0]  # Start with current best + new isolated vertex
            else:
                # Random initialization with bias
                adj = []
                for i in range(target_n):
                    mask = 0
                    for j in range(target_n):
                        if i != j and random.random() < 0.45:
                            mask |= (1 << j)
                    adj.append(mask)
                # Make symmetric
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if (adj[i] >> j) & 1:
                            adj[j] |= (1 << i)
                        else:
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
            
            # Verify initial state
            if not is_valid_bitmask(target_n, adj):
                # Try to repair
                repaired = False
                for step in range(500):
                    if time.time() > deadline - 3.0:
                        break
                    u = random.randint(0, target_n-1)
                    v = random.randint(0, target_n-1)
                    if u == v:
                        continue
                    old = (adj[u] >> v) & 1
                    if old:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    
                    if is_valid_bitmask(target_n, adj):
                        repaired = True
                        break
                    # Revert if this made things worse (heuristic: just revert)
                    if old:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                
                if not repaired:
                    continue
            
            if is_valid_bitmask(target_n, adj):
                found = True
                best_n = target_n
                best_adj = adj
                break
        
        if not found and target_n > best_n + 2:
            # Can't extend further, stop
            break
    
    edges = edges_from_masks(best_n, best_adj)
    return {'n': best_n, 'edges': edges}
