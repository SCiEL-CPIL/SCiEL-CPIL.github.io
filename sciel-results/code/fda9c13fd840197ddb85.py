import random
import time
from itertools import combinations
from math import gcd, exp

def construct():
    deadline = time.time() + 145
    start_time = time.time()
    
    # 1. Generate Pool - use a structured set that creates many collinear triples
    pool = []
    # Use a grid with step 1 to create many collinear points
    for x in range(-40, 41):
        for y in range(-40, 41):
            pool.append((x, y))
    
    # Remove duplicates and shuffle
    pool = list(set(pool))
    random.shuffle(pool)
    n_pool = len(pool)
    
    # 2. Precompute line keys for all pairs
    def get_line_key(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0 and dy == 0:
            return None
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            g = 1
        dx_n = dx // g
        dy_n = dy // g
        if dx_n < 0 or (dx_n == 0 and dy_n < 0):
            dx_n = -dx_n
            dy_n = -dy_n
        c = dy_n * p1[0] - dx_n * p1[1]
        return (dx_n, dy_n, c)
    
    # Build line map: line_key -> set of point indices
    line_map = {}
    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            lk = get_line_key(pool[i], pool[j])
            if lk:
                if lk not in line_map:
                    line_map[lk] = set()
                line_map[lk].add(i)
                line_map[lk].add(j)
    
    # Build list of lines with >= 3 points
    lines = []
    for lk, idx_set in line_map.items():
        if len(idx_set) >= 3:
            lines.append(sorted(idx_set))
    
    # 3. Fast validity check using precomputed line membership
    # For each point, precompute which lines it's on
    point_lines = [[] for _ in range(n_pool)]
    for line_idx, line in enumerate(lines):
        for idx in line:
            point_lines[idx].append(line_idx)
    
    def is_valid_fast(selected_indices):
        """Check if selected set has no 4 collinear points."""
        sel_set = set(selected_indices)
        for line in lines:
            count = sum(1 for idx in line if idx in sel_set)
            if count >= 4:
                return False
        return True
    
    # 4. Count collinear triples in selected set
    def count_triples(selected_indices):
        """Count number of collinear triples in the selected set."""
        sel_set = set(selected_indices)
        triples = 0
        for line in lines:
            count = sum(1 for idx in line if idx in sel_set)
            if count >= 3:
                triples += count * (count - 1) * (count - 2) // 6
        return triples
    
    # 5. Exact Independence Number using Branch and Bound
    def calc_exact_in(selected_indices):
        """Calculate the largest subset with no 3 collinear points."""
        n = len(selected_indices)
        local_map = {idx: i for i, idx in enumerate(selected_indices)}
        
        # Build conflict structure: for each pair (i,j), which k make them collinear
        # conflict[i][j] = bitmask of k such that i,j,k are collinear
        conflict = [[0] * n for _ in range(n)]
        
        for line in lines:
            local_line = [local_map[idx] for idx in line if idx in local_map]
            if len(local_line) >= 3:
                for a in range(len(local_line)):
                    for b in range(a + 1, len(local_line)):
                        i, j = local_line[a], local_line[b]
                        for c in range(len(local_line)):
                            if c != a and c != b:
                                k = local_line[c]
                                conflict[i][j] |= (1 << k)
                                conflict[j][i] |= (1 << k)
        
        # Precompute triple masks for quick validity check
        triple_masks = []
        for line in lines:
            local_line = [local_map[idx] for idx in line if idx in local_map]
            if len(local_line) >= 3:
                for combo in combinations(local_line, 3):
                    mask = 0
                    for idx in combo:
                        mask |= (1 << idx)
                    triple_masks.append(mask)
        triple_masks = list(set(triple_masks))
        
        def is_valid_mask(mask):
            """Check if mask represents a valid set (no 3 collinear)."""
            for tm in triple_masks:
                if (mask & tm) == tm:
                    return False
            return True
        
        # Branch and bound
        best = 0
        full_mask = (1 << n) - 1
        
        def branch_and_bound(avail_mask, selected_mask, count):
            nonlocal best
            
            # Pruning: upper bound
            remaining = bin(avail_mask).count('1')
            if count + remaining <= best:
                return
            
            if avail_mask == 0:
                if count > best:
                    best = count
                return
            
            # Lower bound: greedy
            if count > best:
                best = count
            
            # Pick a node to branch on
            # Heuristic: pick node with most conflicts
            node_mask = avail_mask
            best_node = -1
            best_degree = -1
            while node_mask:
                bit = node_mask & -node_mask
                node = bit.bit_length() - 1
                # Count conflicts with other available nodes
                degree = 0
                other_mask = avail_mask & ~bit
                while other_mask:
                    obit = other_mask & -other_mask
                    onode = obit.bit_length() - 1
                    if conflict[node][onode] & avail_mask:
                        degree += 1
                    other_mask ^= obit
                if degree > best_degree:
                    best_degree = degree
                    best_node = node
                node_mask ^= bit
            
            if best_node == -1:
                return
            
            node_bit = 1 << best_node
            rest_avail = avail_mask & ~node_bit
            
            # Option 1: Don't select best_node
            branch_and_bound(rest_avail, selected_mask, count)
            
            # Option 2: Select best_node
            # Check if we can select it
            valid = True
            temp_sel = selected_mask
            while temp_sel:
                u_bit = temp_sel & -temp_sel
                u = u_bit.bit_length() - 1
                if conflict[u][best_node] & selected_mask:
                    valid = False
                    break
                temp_sel ^= u_bit
            
            if valid:
                new_selected = selected_mask | node_bit
                # When we select best_node, we need to prevent selecting any pair (u,v)
                # such that best_node, u, v are collinear.
                # This means for any u in rest_avail, we cannot select v if 
                # conflict[best_node][u] has v and v is in rest_avail.
                # But this is a pairwise constraint, not a simple removal.
                # For now, just recurse and let the validity check handle it.
                branch_and_bound(rest_avail, new_selected, count + 1)
        
        branch_and_bound(full_mask, 0, 0)
        return best
    
    # 6. Simulated Annealing
    def proxy_score(selected):
        """Higher is better: more triples, valid configuration."""
        if not is_valid_fast(selected):
            return -1000000
        return count_triples(selected)
    
    # Initial solution
    best_solution = random.sample(range(n_pool), 20)
    attempts = 0
    while not is_valid_fast(best_solution) and attempts < 1000:
        best_solution = random.sample(range(n_pool), 20)
        attempts += 1
    
    best_score = proxy_score(best_solution)
    current_solution = best_solution[:]
    current_score = best_score
    
    temp = 15.0
    cooling = 0.9995
    min_temp = 0.5
    
    while time.time() < deadline - 20:
        # Neighbor: swap one point
        sel_idx = random.randint(0, 19)
        current_set = set(current_solution)
        unselected = [i for i in range(n_pool) if i not in current_set]
        if not unselected:
            break
        new_point = random.choice(unselected)
        
        new_solution = current_solution[:]
        new_solution[sel_idx] = new_point
        
        new_score = proxy_score(new_solution)
        
        delta = new_score - current_score
        if delta >= 0:
            accept = True
        else:
            exponent = -delta / temp
            if exponent > 700:
                accept = False
            else:
                accept = random.random() < exp(-exponent)
        
        if accept:
            current_solution = new_solution
            current_score = new_score
            if current_score > best_score:
                best_solution = current_solution[:]
                best_score = current_score
        
        temp = max(temp * cooling, min_temp)
    
    # 7. Local search polish
    # Try single-point swaps to improve
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(20):
            current_set = set(best_solution)
            unselected = [j for j in range(n_pool) if j not in current_set]
            for new_pt in unselected[:50]:  # Limit candidates for speed
                candidate = best_solution[:]
                candidate[i] = new_pt
                if is_valid_fast(candidate):
                    new_score = count_triples(candidate)
                    if new_score > best_score:
                        best_solution = candidate
                        best_score = new_score
                        improved = True
    
    # 8. Verify and return
    if is_valid_fast(best_solution):
        return [[pool[i][0], pool[i][1]] for i in best_solution]
    else:
        # Fallback: return a known valid configuration
        # Points on y = x^2 for x = 0..19 have no 3 collinear
        fallback = [[x, x*x] for x in range(20)]
        return fallback
