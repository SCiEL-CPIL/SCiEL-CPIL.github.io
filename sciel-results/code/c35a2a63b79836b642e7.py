import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    def time_ok():
        return time.time() < deadline
    
    # Build Paley(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj_bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    # Precompute K4s and Ind4s for the seed graph
    full_mask = (1 << n) - 1
    comp_bits = [0] * n
    for i in range(n):
        comp_bits[i] = (full_mask ^ adj_bits[i]) & ~(1 << i)
    
    def compute_k4s(n, adj_bits):
        k4s = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_bits[i] >> j) & 1:
                    common = adj_bits[i] & adj_bits[j]
                    for k in range(j + 1, n):
                        if (common >> k) & 1:
                            common2 = common & adj_bits[k]
                            for l in range(k + 1, n):
                                if (common2 >> l) & 1:
                                    k4s.append((i, j, k, l))
        return k4s
    
    def compute_ind4s(n, adj_bits):
        full_mask = (1 << n) - 1
        comp_bits = [0] * n
        for i in range(n):
            comp_bits[i] = (full_mask ^ adj_bits[i]) & ~(1 << i)
        ind4s = []
        for i in range(n):
            for j in range(i + 1, n):
                if (comp_bits[i] >> j) & 1:
                    common = comp_bits[i] & comp_bits[j]
                    for k in range(j + 1, n):
                        if (common >> k) & 1:
                            common2 = common & comp_bits[k]
                            for l in range(k + 1, n):
                                if (common2 >> l) & 1:
                                    ind4s.append((i, j, k, l))
        return ind4s
    
    seed_k4s = compute_k4s(n, adj_bits)
    seed_ind4s = compute_ind4s(n, adj_bits)
    
    def has_k5_bitset(n, adj_bits):
        """Fast K5 check using bitsets."""
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_bits[i] >> j) & 1:
                    common = adj_bits[i] & adj_bits[j]
                    for k in range(j + 1, n):
                        if (common >> k) & 1:
                            common2 = common & adj_bits[k]
                            for l in range(k + 1, n):
                                if (common2 >> l) & 1:
                                    common3 = common2 & adj_bits[l]
                                    if common3 & ~((1 << (l + 1)) - 1):
                                        return True
        return False
    
    def has_ind5_bitset(n, adj_bits):
        """Fast independent 5-set check using bitsets."""
        comp_bits = [0] * n
        full_mask = (1 << n) - 1
        for i in range(n):
            comp_bits[i] = (full_mask ^ adj_bits[i]) & ~(1 << i)
        
        for i in range(n):
            for j in range(i + 1, n):
                if (comp_bits[i] >> j) & 1:
                    common = comp_bits[i] & comp_bits[j]
                    for k in range(j + 1, n):
                        if (common >> k) & 1:
                            common2 = common & comp_bits[k]
                            for l in range(k + 1, n):
                                if (common2 >> l) & 1:
                                    common3 = common2 & comp_bits[l]
                                    if common3 & ~((1 << (l + 1)) - 1):
                                        return True
        return False
    
    def is_valid(n, adj_bits):
        return not has_k5_bitset(n, adj_bits) and not has_ind5_bitset(n, adj_bits)
    
    # Verify seed
    if not is_valid(n, adj_bits):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_bits[i] >> j) & 1:
                    edges.append((i, j))
        return n, edges
    
    best_n = n
    best_adj_bits = list(adj_bits)
    
    def count_conflicts_k4(neighbor_mask, k4s):
        """Count how many K4s are entirely in the neighbor set."""
        count = 0
        for k4 in k4s:
            if all((neighbor_mask >> v) & 1 for v in k4):
                count += 1
        return count
    
    def count_conflicts_ind4(non_neighbor_mask, ind4s):
        """Count how many Ind4s are entirely in the non-neighbor set."""
        count = 0
        for ind4 in ind4s:
            if all((non_neighbor_mask >> v) & 1 for v in ind4):
                count += 1
        return count
    
    def try_add_vertex_optimized(n, adj_bits, k4s, ind4s, max_attempts=10000):
        """Try to add a vertex using precomputed K4s and Ind4s with smarter heuristics."""
        all_mask = (1 << n) - 1
        
        # Strategy 1: Greedy - find a vertex with minimal conflicts
        # Try to construct neighbor mask that avoids all K4s and Ind4s
        for attempt in range(max_attempts):
            if not time_ok():
                break
            
            if attempt < max_attempts // 3:
                # Phase 1: Random with bias toward balanced degree
                target_degree = n // 2
                neighbor_mask = 0
                for i in range(n):
                    if random.random() < 0.5:
                        neighbor_mask |= (1 << i)
            elif attempt < 2 * max_attempts // 3:
                # Phase 2: Start from a random mask, then flip bits to reduce conflicts
                neighbor_mask = 0
                for i in range(n):
                    if random.random() < 0.5:
                        neighbor_mask |= (1 << i)
                
                # Local improvement: flip bits to reduce total conflicts
                for _ in range(20):
                    if not time_ok():
                        break
                    old_conflicts = count_conflicts_k4(neighbor_mask, k4s) + count_conflicts_ind4(all_mask ^ neighbor_mask, ind4s)
                    v = random.randint(0, n - 1)
                    new_mask = neighbor_mask ^ (1 << v)
                    new_conflicts = count_conflicts_k4(new_mask, k4s) + count_conflicts_ind4(all_mask ^ new_mask, ind4s)
                    if new_conflicts < old_conflicts:
                        neighbor_mask = new_mask
            else:
                # Phase 3: Targeted - try to connect to vertices that break K4s
                neighbor_mask = 0
                for i in range(n):
                    if random.random() < 0.4:
                        neighbor_mask |= (1 << i)
            
            non_neighbor_mask = all_mask ^ neighbor_mask
            
            # Check K4s: no K4 should be entirely in neighbors
            valid = True
            for k4 in k4s:
                if all((neighbor_mask >> v) & 1 for v in k4):
                    valid = False
                    break
            
            if valid:
                # Check Ind4s: no Ind4 should be entirely in non-neighbors
                for ind4 in ind4s:
                    if all((non_neighbor_mask >> v) & 1 for v in ind4):
                        valid = False
                        break
            
            if valid:
                # Full validation
                new_adj_bits = list(adj_bits)
                new_adj_bits.append(neighbor_mask)
                for i in range(n):
                    if (neighbor_mask >> i) & 1:
                        new_adj_bits[i] |= (1 << n)
                
                if is_valid(n + 1, new_adj_bits):
                    return True, new_adj_bits
        
        return False, None
    
    current_n = n
    current_adj_bits = list(adj_bits)
    current_k4s = seed_k4s
    current_ind4s = seed_ind4s
    
    # Multiple rounds of vertex addition with restarts
    for round_num in range(20):
        if not time_ok():
            break
        
        # Try to add a vertex
        success, new_adj_bits = try_add_vertex_optimized(
            current_n, current_adj_bits, current_k4s, current_ind4s,
            max_attempts=5000
        )
        if success:
            current_adj_bits = new_adj_bits
            current_n += 1
            best_n = current_n
            best_adj_bits = list(current_adj_bits)
            # Recompute K4s and Ind4s
            current_k4s = compute_k4s(current_n, current_adj_bits)
            current_ind4s = compute_ind4s(current_n, current_adj_bits)
        else:
            # If stuck, try local search to find a better configuration
            if best_n > n and time_ok():
                found_improvement = False
                for _ in range(3000):
                    if not time_ok():
                        break
                    
                    # Try flipping one edge
                    v = random.randint(0, best_n - 1)
                    w = random.randint(0, best_n - 1)
                    if v == w:
                        continue
                    
                    new_adj_bits = list(best_adj_bits)
                    # Check if edge exists
                    has_edge = (new_adj_bits[v] >> w) & 1
                    new_adj_bits[v] ^= (1 << w)
                    new_adj_bits[w] ^= (1 << v)
                    
                    if is_valid(best_n, new_adj_bits):
                        # Try to add a vertex from this improved graph
                        k4s = compute_k4s(best_n, new_adj_bits)
                        ind4s = compute_ind4s(best_n, new_adj_bits)
                        success2, added = try_add_vertex_optimized(best_n, new_adj_bits, k4s, ind4s, max_attempts=1000)
                        if success2:
                            best_adj_bits = added
                            best_n += 1
                            current_n = best_n
                            current_adj_bits = best_adj_bits
                            current_k4s = compute_k4s(current_n, current_adj_bits)
                            current_ind4s = compute_ind4s(current_n, current_adj_bits)
                            found_improvement = True
                            break
                if not found_improvement:
                    break
            else:
                break
    
    # Final local search to try to squeeze out more vertices
    if time_ok() and best_n > n:
        for _ in range(5000):
            if not time_ok():
                break
            
            # Try flipping one edge in the best graph
            v = random.randint(0, best_n - 1)
            w = random.randint(0, best_n - 1)
            if v == w:
                continue
            
            new_adj_bits = list(best_adj_bits)
            new_adj_bits[v] ^= (1 << w)
            new_adj_bits[w] ^= (1 << v)
            
            if is_valid(best_n, new_adj_bits):
                k4s = compute_k4s(best_n, new_adj_bits)
                ind4s = compute_ind4s(best_n, new_adj_bits)
                success, added = try_add_vertex_optimized(best_n, new_adj_bits, k4s, ind4s, max_attempts=500)
                if success:
                    best_adj_bits = added
                    best_n += 1
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj_bits[i] >> j) & 1:
                edges.append((i, j))
    
    return best_n, edges
