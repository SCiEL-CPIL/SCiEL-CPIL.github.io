import random
import itertools
import time

# Precompute Q3 properties
q3_edges = []
q3_adj = [0]*8
for i in range(8):
    adj = 0
    for j in range(8):
        if i != j and bin(i ^ j).count('1') == 1:
            adj |= (1 << j)
    q3_adj[i] = adj

# Precompute all subsets of size 8
n = 16
all_subsets = list(itertools.combinations(range(n), 8))
num_subsets = len(all_subsets)

# Precompute which subsets contain which edges
edge_to_subsets = {}
for idx, subset in enumerate(all_subsets):
    for i in range(8):
        for j in range(i+1, 8):
            u, v = subset[i], subset[j]
            if u > v:
                u, v = v, u
            if (u, v) not in edge_to_subsets:
                edge_to_subsets[(u, v)] = []
            edge_to_subsets[(u, v)].append(idx)

def contains_q3_fast(subset, adj_bits):
    """
    Fast check if induced subgraph on 8 vertices contains Q3.
    """
    local_adj = [0]*8
    for i in range(8):
        vi = subset[i]
        for j in range(8):
            if i != j and (adj_bits[vi] >> subset[j]) & 1:
                local_adj[i] |= (1 << j)
    
    # Degree check
    degs = [bin(local_adj[i]).count('1') for i in range(8)]
    if min(degs) < 3:
        return False
    
    total_deg = sum(degs)
    if total_deg < 24:
        return False
    
    mapping = [-1]*8
    used_mask = 0
    
    def backtrack(q3_v):
        nonlocal used_mask
        if q3_v == 8:
            return True
        
        candidates = []
        for s_idx in range(8):
            if (used_mask >> s_idx) & 1:
                continue
            valid = True
            for u in range(q3_v):
                if (q3_adj[u] >> q3_v) & 1:
                    if not ((local_adj[mapping[u]] >> s_idx) & 1):
                        valid = False
                        break
            if valid:
                candidates.append(s_idx)
        
        # Sort candidates by degree descending?
        # Higher degree vertices are more constrained?
        # Actually, if a vertex has high degree, it's more likely to be part of a Q3.
        # But for pruning, maybe low degree fails faster?
        # Let's try high degree first.
        candidates.sort(key=lambda x: degs[x], reverse=True)
        
        for s_idx in candidates:
            mapping[q3_v] = s_idx
            used_mask |= (1 << s_idx)
            if backtrack(q3_v + 1):
                return True
            used_mask &= ~(1 << s_idx)
            mapping[q3_v] = -1
        return False
    
    # Symmetry breaking: Fix Q3 vertex 0 to local vertex with max degree
    max_deg = max(degs)
    # Find all indices with max degree
    max_indices = [i for i in range(8) if degs[i] == max_deg]
    
    for max_idx in max_indices:
        mapping[0] = max_idx
        used_mask = (1 << max_idx)
        if backtrack(1):
            return True
        mapping[0] = -1
        used_mask = 0
        
    return False

def construct():
    start_time = time.time()
    time_limit = 118
    
    adj_bits = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.8:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    subset_q3_status = [False] * num_subsets
    current_q3_count = 0
    
    for idx, subset in enumerate(all_subsets):
        if contains_q3_fast(subset, adj_bits):
            subset_q3_status[idx] = True
            current_q3_count += 1
    
    best_adj_bits = adj_bits[:]
    best_q3_count = current_q3_count
    best_edges = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
    
    T = 1.0
    T_min = 0.001
    cooling = 0.99
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        u, v = random.choice(all_edges)
        is_present = (adj_bits[u] >> v) & 1
        
        rel_subsets = edge_to_subsets.get((u, v), [])
        
        delta_p = 0
        status_changes = {} # idx -> new_status
        
        if is_present:
            # Removing
            for idx in rel_subsets:
                if subset_q3_status[idx]:
                    # Check if still Q3
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                    still_q3 = contains_q3_fast(all_subsets[idx], adj_bits)
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    
                    if not still_q3:
                        delta_p -= 1
                        status_changes[idx] = False
        else:
            # Adding
            for idx in rel_subsets:
                if not subset_q3_status[idx]:
                    # Check if becomes Q3
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    becomes_q3 = contains_q3_fast(all_subsets[idx], adj_bits)
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                    
                    if becomes_q3:
                        delta_p += 1
                        status_changes[idx] = True
        
        cur_edges = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
        cur_score = 100 * current_q3_count - cur_edges
        
        new_q3_count = current_q3_count + delta_p
        new_edges = cur_edges + (1 if not is_present else -1)
        new_score = 100 * new_q3_count - new_edges
        
        delta_score = new_score - cur_score
        
        if delta_score < 0 or random.random() < pow(2.718, -delta_score / T):
            # Accept
            if is_present:
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
            else:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
            
            # Apply status changes
            for idx, new_status in status_changes.items():
                subset_q3_status[idx] = new_status
            
            current_q3_count += delta_p
            
            cur_edges = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
            if current_q3_count == 0 and cur_edges > best_edges:
                best_edges = cur_edges
                best_adj_bits = adj_bits[:]
            
            T *= cooling
    
    adj_bits = best_adj_bits
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
