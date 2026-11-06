import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Target ns in descending order
    target_ns = [26, 25, 24, 23, 22, 21, 20]
    
    # 1. Try General Local Search with delta evaluation (fast)
    for n in target_ns:
        if time.time() > deadline - 20:
            break
        result = local_search_delta(n, time_limit=15)
        if result is not None:
            return result

    # 2. Try Cyclic Search
    for n in target_ns:
        if time.time() > deadline:
            break
        result = search_cyclic(n, time_limit=5)
        if result is not None:
            return result

    # Fallback: valid coloring for n=11
    return {'n': 11, 'colors': [(i + j) % 5 for i in range(11) for j in range(i + 1, 11)]}

def search_cyclic(n, time_limit=5):
    k = n // 2
    # params: colors for differences 1..k
    # params[d-1] is the color for difference d (and -d)
    params = [random.randint(0, 4) for _ in range(k)]
    
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    def gen_colors(params):
        colors = [0] * num_edges
        for idx, (i, j) in enumerate(edges):
            diff = (j - i) % n
            # Map diff to 1..k
            # If diff == 0, not possible for i!=j
            # If diff > k, it represents - (n-diff)
            # We defined params[d-1] for difference d.
            # If diff <= k, color = params[diff-1]
            # If diff > k, color = params[(n-diff)-1]
            if diff <= k:
                d = diff
            else:
                d = n - diff
            colors[idx] = params[d - 1]
        return colors

    def calc_violations(colors):
        neighbors = [[0] * n for _ in range(5)]
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    cnt = common.bit_count()
                    if cnt > 1:
                        violations += cnt * (cnt - 1) // 2
        return violations

    current_colors = gen_colors(params)
    current_violations = calc_violations(current_colors)
    best_violations = current_violations
    best_params = params[:]
    
    if current_violations == 0:
        return {'n': n, 'colors': current_colors}
        
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.99
    
    while time.time() < start_time + time_limit:
        idx = random.randint(0, k - 1)
        old_val = params[idx]
        new_val = random.randint(0, 4)
        if new_val == old_val:
            continue
            
        params[idx] = new_val
        new_colors = gen_colors(params)
        new_violations = calc_violations(new_colors)
        
        delta = new_violations - current_violations
        
        if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
            current_violations = new_violations
            current_colors = new_colors
            if new_violations < best_violations:
                best_violations = new_violations
                best_params = params[:]
                if best_violations == 0:
                    return {'n': n, 'colors': current_colors}
        else:
            params[idx] = old_val
            
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    return None

def local_search_delta(n, time_limit=15):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Build edge index lookup
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
    
    # Initial random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Build neighbor bitmasks per color
    neighbors = [[0] * n for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
    
    # Count C4s: for each color, for each pair (u,v), count common neighbors
    # violations = sum over c, u<v of C(common_c(u,v), 2)
    def count_c4s():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    cnt = common.bit_count()
                    if cnt > 1:
                        total += cnt * (cnt - 1) // 2
        return total
    
    current_violations = count_c4s()
    best_violations = current_violations
    best_colors = colors[:]
    
    if current_violations == 0:
        return {'n': n, 'colors': colors}
    
    start_time = time.time()
    temperature = 2.0
    cooling_rate = 0.9995
    
    while time.time() < start_time + time_limit:
        e_idx = random.randint(0, num_edges - 1)
        u, v = edges[e_idx]
        old_color = colors[e_idx]
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
        
        # Compute delta in violations when changing edge (u,v) from old_color to new_color
        # Removing (u,v) from old_color: for each w, the pair (u,w) and (v,w) lose one common neighbor
        # Adding (u,v) to new_color: for each w, the pair (u,w) and (v,w) gain one common neighbor
        
        delta = 0
        
        # Effect of removing edge (u,v) from old_color
        # For each w != u,v: 
        #   pair (u,w): common neighbors in old_color decrease by 1 if w was neighbor of v in old_color
        #   pair (v,w): common neighbors in old_color decrease by 1 if w was neighbor of u in old_color
        # We need to track C(k,2) changes
        
        # More efficient: track the number of common neighbors for each pair
        # When we remove (u,v) from color c:
        #   For each w where (u,w) in c and (v,w) in c: pair (u,v) loses common neighbor w -> C(k,2) changes
        #   Actually, removing (u,v) means u and v are no longer neighbors in color c
        #   So pairs that had both u and v as common neighbors are affected
        
        # Let's use a different approach: 
        # For color c, let N_c(x) be the set of neighbors of x in color c.
        # A C4 in color c is a pair (a,b) with |N_c(a) ∩ N_c(b)| >= 2, contributing C(|N_c(a)∩N_c(b)|, 2).
        
        # When we change edge (u,v) from old_color to new_color:
        # In old_color: u and v are no longer connected.
        #   For any w: if w was in N_old(u) ∩ N_old(v), then w contributed to C4 count for pair (u,v).
        #   Actually, the C4s involving u and v as endpoints: pairs (u, x) for x in N_old(v), and (v, x) for x in N_old(u).
        
        # Let me think more carefully. The violation count is:
        # V = sum_c sum_{a<b} C(|N_c(a) ∩ N_c(b)|, 2)
        
        # Changing edge (u,v) from color oc to nc:
        # In color oc: remove (u,v). So N_oc(u) loses v, N_oc(v) loses u.
        # In color nc: add (u,v). So N_nc(u) gains v, N_nc(v) gains u.
        
        # For color oc, the pairs affected are those where one endpoint is u or v:
        #   Pairs (u, w) for w != u,v: |N_oc(u) ∩ N_oc(w)| changes
        #     Before: v was in N_oc(u). If v in N_oc(w), then v was a common neighbor.
        #     After: v not in N_oc(u).
        #     So if v in N_oc(w), then |N_oc(u)∩N_oc(w)| decreases by 1.
        #   Similarly for pairs (v, w).
        #   Pair (u,v): |N_oc(u)∩N_oc(w)| for w=u,v... actually pair (u,v) itself:
        #     |N_oc(u) ∩ N_oc(v)| - this changes because u is no longer in N_oc(v) and v no longer in N_oc(u),
        #     but the intersection N_oc(u) ∩ N_oc(v) doesn't include u or v (since no self-loops).
        #     Wait, N_oc(u) ∩ N_oc(v) = common neighbors of u and v in color oc.
        #     Removing (u,v) doesn't change the common neighbors of u and v! It only changes
        #     whether u is a neighbor of v and vice versa, but common neighbors are other vertices.
        #     So pair (u,v) is NOT affected in terms of its C4 contribution.
        
        # So for color oc, affected pairs are (u, w) and (v, w) for all w != u, v.
        # For pair (u, w): 
        #   Before: common = N_oc(u) ∩ N_oc(w). Since v in N_oc(u), if v in N_oc(w), v is common.
        #   After: v not in N_oc(u), so v not common.
        #   So |common| decreases by 1 if v in N_oc(w), else unchanged.
        
        # Similarly for color nc, affected pairs are (u, w) and (v, w) for all w != u, v.
        # For pair (u, w) in nc:
        #   Before: common = N_nc(u) ∩ N_nc(w). 
        #   After: v in N_nc(u), so if v in N_nc(w), v becomes common.
        #   So |common| increases by 1 if v in N_nc(w), else unchanged.
        
        # Let's compute delta:
        # For each w in range(n), w != u, w != v:
        #   In old_color: 
        #     if v in N_oc(w): pair (u,w) loses 1 common neighbor
        #       old_k = |N_oc(u) ∩ N_oc(w)|, new_k = old_k - 1
        #       delta += C(new_k, 2) - C(old_k, 2) = -(old_k - 1)
        #     if u in N_oc(w): pair (v,w) loses 1 common neighbor
        #       old_k = |N_oc(v) ∩ N_oc(w)|, new_k = old_k - 1
        #       delta += -(old_k - 1)
        #   In new_color:
        #     if v in N_nc(w): pair (u,w) gains 1 common neighbor
        #       old_k = |N_nc(u) ∩ N_nc(w)|, new_k = old_k + 1
        #       delta += C(new_k, 2) - C(old_k, 2) = old_k
        #     if u in N_nc(w): pair (v,w) gains 1 common neighbor
        #       old_k = |N_nc(v) ∩ N_nc(w)|, new_k = old_k + 1
        #       delta += old_k
        
        # This is still O(n) per move. Let's implement it.
        
        for w in range(n):
            if w == u or w == v:
                continue
            # Old color: removing (u,v)
            # Pair (u, w): if v in N_oc(w), common count decreases by 1
            if (1 << v) & neighbors[old_color][w]:
                # old_k = |N_oc(u) ∩ N_oc(w)|
                old_k = (neighbors[old_color][u] & neighbors[old_color][w]).bit_count()
                # After removal, k = old_k - 1
                # C(old_k-1, 2) - C(old_k, 2) = (old_k-1)(old_k-2)/2 - old_k(old_k-1)/2 = (old_k-1)(old_k-2-old_k)/2 = (old_k-1)(-2)/2 = -(old_k-1)
                delta -= (old_k - 1)
            # Pair (v, w): if u in N_oc(w), common count decreases by 1
            if (1 << u) & neighbors[old_color][w]:
                old_k = (neighbors[old_color][v] & neighbors[old_color][w]).bit_count()
                delta -= (old_k - 1)
            
            # New color: adding (u,v)
            # Pair (u, w): if v in N_nc(w), common count increases by 1
            if (1 << v) & neighbors[new_color][w]:
                old_k = (neighbors[new_color][u] & neighbors[new_color][w]).bit_count()
                # C(old_k+1, 2) - C(old_k, 2) = old_k
                delta += old_k
            # Pair (v, w): if u in N_nc(w), common count increases by 1
            if (1 << u) & neighbors[new_color][w]:
                old_k = (neighbors[new_color][v] & neighbors[new_color][w]).bit_count()
                delta += old_k
        
        # Apply the change
        colors[e_idx] = new_color
        neighbors[old_color][u] &= ~(1 << v)
        neighbors[old_color][v] &= ~(1 << u)
        neighbors[new_color][u] |= (1 << v)
        neighbors[new_color][v] |= (1 << u)
        
        current_violations += delta
        
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            if best_violations == 0:
                return {'n': n, 'colors': colors}
        
        # Simulated annealing acceptance
        if delta >= 0:
            if random.random() < pow(2.71828, -delta / max(temperature, 0.01)):
                pass  # accept
            else:
                # Revert
                colors[e_idx] = old_color
                neighbors[new_color][u] &= ~(1 << v)
                neighbors[new_color][v] &= ~(1 << u)
                neighbors[old_color][u] |= (1 << v)
                neighbors[old_color][v] |= (1 << u)
                current_violations -= delta
        
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
    
    return None
