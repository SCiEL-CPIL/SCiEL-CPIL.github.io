import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 140.0  # 10s safety margin
    
    # Lazebnik-Woldar construction for n=26
    def _build_lw(n):
        q = 5
        colors = []
        for i in range(n):
            ci = None if i == 25 else (i % 5, i // 5)
            for j in range(i + 1, n):
                cj = None if j == 25 else (j % 5, j // 5)
                if ci is None or cj is None:
                    colors.append(4)
                    continue
                dx = (cj[0] - ci[0]) % 5
                dy = (cj[1] - ci[1]) % 5
                if dx == 0:
                    c = 1
                elif dy == 0:
                    c = 0
                elif dx == dy:
                    c = 2
                elif (dx + dy) % 5 == 0:
                    c = 3
                else:
                    c = 4
                colors.append(c)
        return {'n': n, 'colors': colors}
    
    # Get baseline n=26
    base = _build_lw(26)
    best_n = 26
    best_colors = base['colors'].copy()
    
    # Precompute edge index lookup
    def make_index_map(n):
        idx_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx_map[(i, j)] = idx
                idx += 1
        return idx_map
    
    # Fast conflict count using numpy
    def count_conflicts_np(colors, n):
        adjs = [np.zeros((n, n), dtype=np.uint8) for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adjs[c][i][j] = 1
                adjs[c][j][i] = 1
                idx += 1
        
        conflicts = 0
        for c in range(5):
            A = adjs[c]
            A2 = A @ A
            np.fill_diagonal(A2, 0)
            excess = np.maximum(A2 - 1, 0)
            conflicts += int(np.sum(excess)) // 2
        return conflicts
    
    def is_valid(colors, n):
        return count_conflicts_np(colors, n) == 0
    
    # Verify baseline
    if not is_valid(best_colors, best_n):
        n = 11
        best_n = n
        best_colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    
    # Phase 1: Greedy extension
    # Try to extend from best_n to larger n using greedy vertex addition
    def greedy_extend(start_colors, start_n, target_n, max_time):
        """Try to extend coloring from start_n to target_n greedily."""
        if target_n <= start_n:
            return start_colors.copy(), start_n
        
        colors = start_colors.copy()
        current_n = start_n
        
        # Precompute index map for current size
        idx_map = make_index_map(current_n)
        
        for new_v in range(start_n, target_n):
            if time.time() > max_time:
                break
            
            # Add edges from new_v to all existing vertices
            # For each existing vertex u, choose color that minimizes conflicts
            # We need to find the color for edge (u, new_v) that creates fewest C4s
            
            # First, let's see what conflicts each color would create
            # A C4 in color c is formed if there exist vertices p, q such that:
            #   (p, new_v) has color c, (q, new_v) has color c, 
            #   (p, u) has color c, (q, u) has color c for some u
            # More precisely: if we assign color c to edge (u, new_v), then for any 
            # other vertex w, if edge (w, new_v) also has color c, and edges (u, w) and 
            # (some other) ... 
            # 
            # Actually, the C4 condition: vertices a, b, c, d form a C4 in color k if
            # edges (a,b), (b,c), (c,d), (d,a) all have color k.
            # 
            # When we add new vertex v with edges to existing vertices, a C4 involving v
            # must have v connected to two existing vertices, say u1 and u2, in color c.
            # Then we need two more vertices w1, w2 such that (u1,w1), (w1,v)... no.
            # 
            # A C4 involving v: v-u1-w1-u2-v or v-u1-w1-v... no, C4 is a 4-cycle.
            # v, u1, w, u2 form a C4 if edges (v,u1), (u1,w), (w,u2), (u2,v) all have same color.
            # 
            # So if we assign color c to edges (v, u1) and (v, u2), then for any w,
            # if edges (u1, w) and (u2, w) both have color c, we get a C4.
            # 
            # So the conflict from assigning color c to edge (v, u) is:
            #   for each other vertex w (w != u), if edge (v, w) has color c AND
            #   edges (u, w) has color c, then we have a potential C4.
            # But we're assigning colors one at a time, so we need to consider
            # the colors already assigned to (v, w) for w < u.
            
            # Greedy: for each vertex u (in order), choose color for (u, new_v)
            # that minimizes the number of "conflict pairs"
            
            # Track what colors we've assigned to (v, w) for w already processed
            v_colors = {}  # v_colors[w] = color of edge (w, new_v)
            
            # Process vertices in order 0, 1, ..., new_v-1
            for u in range(new_v):
                # For each possible color c, count how many conflicts it would create
                # A conflict: exists w (w != u, w < new_v) such that:
                #   v_colors[w] == c AND edge (u, w) has color c
                # This means v-u-w-v forms a path, and if we also had v-w and u-w in color c,
                # we'd have a C4: v, u, w, ... wait, that's only 3 vertices.
                # 
                # C4: v, u, w, x where edges (v,u), (u,w), (w,x), (x,v) all color c.
                # When adding v, the C4s involving v are of the form:
                # v, a, b, c where (v,a) and (v,c) are new edges (color c_k), 
                # and (a,b) and (b,c) are existing edges (color c_k).
                # 
                # So: for a given color k, if we set (v,u)=k and (v,w)=k for some w,
                # then for any b, if (u,b)=k and (w,b)=k, we have C4 v-u-b-w-v.
                # 
                # So the conflict count for setting (v,u)=k given current v_colors is:
                #   sum over w where v_colors[w]==k of: 
                #       number of b such that edge(u,b)=k AND edge(w,b)=k
                # 
                # But we also need to count C4s where u and w are the "new" edges.
                # Actually that's the same thing.
                
                best_color = 0
                min_conf = float('inf')
                
                for c in range(5):
                    conf = 0
                    for w in v_colors:
                        if v_colors[w] == c:
                            # Count common neighbors of u and w in color c
                            # among vertices 0..new_v-1
                            for b in range(new_v):
                                if b == u or b == w:
                                    continue
                                # Check if edge (u,b) has color c
                                idx_ub = make_idx(u, b, new_v)
                                if colors[idx_ub] == c:
                                    # Check if edge (w,b) has color c
                                    idx_wb = make_idx(w, b, new_v)
                                    if colors[idx_wb] == c:
                                        conf += 1
                    # Also, we need to check: does setting (v,u)=c create a C4
                    # where u is one of the "middle" vertices? 
                    # C4: v, a, u, b where (v,a)=c, (a,u)=c, (u,b)=c, (b,v)=c
                    # Here (v,a) and (v,b) are new edges. So if v_colors[a]==c and 
                    # v_colors[b]==c, and (a,u)=c and (u,b)=c, we have a C4.
                    # This is already counted above (with w=a, b=b, or w=b, b=a).
                    
                    if conf < min_conf:
                        min_conf = conf
                        best_color = c
                
                v_colors[u] = best_color
                # Add to colors list
                idx_vu = make_idx(u, new_v, new_v + 1)
                # We need to insert at the right position
                # Actually, let's just build the full list at the end
            
            # Now add all the new edges to colors
            # The new edges are (u, new_v) for u in 0..new_v-1
            # In lexicographic order, these come after all existing edges
            # Existing edges: all (i,j) with i<j<new_v
            # New edges: (0,new_v), (1,new_v), ..., (new_v-1, new_v)
            for u in range(new_v):
                colors.append(v_colors[u])
            
            current_n = new_v + 1
            idx_map = make_index_map(current_n)
        
        return colors, current_n
    
    def make_idx(i, j, n):
        """Get index of edge (i,j) in lexicographic list for n vertices."""
        if i > j:
            i, j = j, i
        # Number of edges before row i
        # Row k has (n-1-k) edges
        # Sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    # Phase 2: Local search with tabu and perturbation
    def local_search(colors, n, max_time):
        """Tabu search with perturbation to find valid coloring."""
        idx_map = make_index_map(n)
        current_conf = count_conflicts_np(colors, n)
        best_conf = current_conf
        best_colors = colors.copy()
        
        tabu_list = {}  # (idx) -> step when it becomes non-tabu
        tabu_tenure = 15
        current_step = 0
        
        while time.time() < max_time:
            if current_conf == 0:
                return colors, 0
            
            if current_conf < best_conf:
                best_conf = current_conf
                best_colors = colors.copy()
            
            # Choose move: change color of one edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            idx = idx_map[(i, j)]
            
            # Tabu check
            if idx in tabu_list and current_step < tabu_list[idx]:
                continue
            
            old_c = colors[idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            colors[idx] = new_c
            new_conf = count_conflicts_np(colors, n)
            
            # Aspiration: accept if it improves best
            if new_conf < best_conf or new_conf < current_conf:
                current_conf = new_conf
            else:
                colors[idx] = old_c
                # Try a different color
                new_c = random.randint(0, 4)
                if new_c == old_c:
                    continue
                colors[idx] = new_c
                new_conf = count_conflicts_np(colors, n)
                if new_conf < current_conf:
                    current_conf = new_conf
                else:
                    colors[idx] = old_c
                    continue
            
            tabu_list[idx] = current_step + tabu_tenure
            current_step += 1
            
            # Periodic perturbation
            if current_step % 100 == 0 and current_conf > 0:
                # Perturb: change 3-5 random edges
                for _ in range(random.randint(3, 5)):
                    pi = random.randint(0, n - 1)
                    pj = random.randint(0, n - 1)
                    if pi == pj:
                        continue
                    if pi > pj:
                        pi, pj = pj, pi
                    pidx = idx_map[(pi, pj)]
                    colors[pidx] = random.randint(0, 4)
                current_conf = count_conflicts_np(colors, n)
        
        # Return best found
        if best_conf < current_conf:
            return best_colors, best_conf
        return colors, current_conf
    
    # Try to extend to larger n
    for target_n in range(best_n + 1, 29):
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Strategy 1: Greedy extension
        greedy_colors, greedy_n = greedy_extend(best_colors, best_n, target_n, deadline - 20)
        if greedy_n == target_n:
            g_conf = count_conflicts_np(greedy_colors, target_n)
            if g_conf == 0:
                best_n = target_n
                best_colors = greedy_colors
                found = True
                break
        
        # Strategy 2: Local search from greedy result or random
        if not found:
            # Start from greedy if it got close, else random
            if greedy_n >= target_n - 2:
                init_colors = greedy_colors[:target_n * (target_n - 1) // 2]
                if len(init_colors) < target_n * (target_n - 1) // 2:
                    init_colors += [random.randint(0, 4) for _ in range(target_n * (target_n - 1) // 2 - len(init_colors))]
            else:
                init_colors = [random.randint(0, 4) for _ in range(target_n * (target_n - 1) // 2)]
            
            # Run local search
            ls_colors, ls_conf = local_search(init_colors, target_n, deadline - 10)
            if ls_conf == 0:
                best_n = target_n
                best_colors = ls_colors
                found = True
                break
        
        if not found and time.time() > deadline - 15:
            break
    
    # Final verification
    if not is_valid(best_colors, best_n):
        n = 11
        best_n = n
        best_colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    
    return {'n': best_n, 'colors': best_colors}
