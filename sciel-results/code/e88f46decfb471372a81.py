# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115
    n = 18

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4_fast(adj):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
                for c in range(b + 1, n):
                    if (common >> c) & 1:
                        if common & adj[c] & ~((1 << (c + 1)) - 1):
                            return True
        return False

    def find_all_triangles(adj):
        triangles = []
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
                tmp = common
                while tmp:
                    bit = tmp & (-tmp)
                    c = bit.bit_length() - 1
                    mask = (1 << a) | (1 << b) | (1 << c)
                    triangles.append(mask)
                    tmp &= tmp - 1
        return triangles

    def compute_triangle_packing_bound(adj):
        triangles = find_all_triangles(adj)
        if not triangles:
            return 0
        
        remaining = set(triangles)
        count = 0
        selected = []
        available = set(triangles)
        
        while available:
            t = min(available)
            selected.append(t)
            new_available = set()
            for u in available:
                if u == t:
                    continue
                if (u & t) == 0:
                    new_available.add(u)
            available = new_available
            count += 1
            
        return count

    def max_triangle_free_size(adj, time_limit=3.0):
        best = [0]
        deg = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: (-deg[v], v))
        t0 = time.time()
        timeout = [False]
        
        def has_triangle_with_v(v, mask):
            neighbors_in_mask = mask & adj[v]
            tmp = neighbors_in_mask
            while tmp:
                bit = tmp & (-tmp)
                u = bit.bit_length() - 1
                rest = tmp & ~bit
                if (adj[u] & rest) != 0:
                    return True
                tmp = rest
            return False

        def backtrack(idx, current_mask, current_size):
            if timeout[0]:
                return
            if time.time() - t0 > time_limit:
                timeout[0] = True
                return
            if idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            if current_size + (n - idx) <= best[0]:
                return
            
            v = order[idx]

            if not has_triangle_with_v(v, current_mask):
                backtrack(idx + 1, current_mask | (1 << v), current_size + 1)
                if timeout[0]:
                    return

            if current_size + (n - idx - 1) > best[0]:
                backtrack(idx + 1, current_mask, current_size)

        backtrack(0, 0, 0)
        return best[0]

    def make_paley17_plus_iso():
        adj = [0] * n
        residues = {1, 2, 4, 8, 9, 13, 15, 16}
        for i in range(17):
            for j in range(i + 1, 17):
                diff = (j - i) % 17
                if diff in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def make_complete_tripartite():
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                gi = i // 6
                gj = j // 6
                if gi != gj:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def make_paley17_shuffled():
        adj_base = make_paley17_plus_iso()
        perm = list(range(17))
        random.shuffle(perm)
        new_adj = [0] * n
        for i in range(17):
            for j in range(17):
                if i == j:
                    continue
                if (adj_base[i] >> j) & 1:
                    new_adj[perm[i]] |= (1 << perm[j])
                    new_adj[perm[j]] |= (1 << perm[i])
        return new_adj

    def remove_k4(adj):
        for _ in range(500):
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    if not ((adj[a] >> b) & 1):
                        continue
                    common = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
                    for c in range(b + 1, n):
                        if (common >> c) & 1:
                            remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                            if remaining:
                                d_bit = remaining & (-remaining)
                                d = d_bit.bit_length() - 1
                                pair = random.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
                                x, y = pair
                                adj[x] &= ~(1 << y)
                                adj[y] &= ~(1 << x)
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        return not has_k4_fast(adj)

    random.seed(42)
    
    candidates = []
    candidates.append(make_complete_tripartite())
    candidates.append(make_paley17_plus_iso())
    
    for _ in range(15):
        if time.time() > deadline - 50:
            break
        candidates.append(make_paley17_shuffled())
        
    for _ in range(20):
        if time.time() > deadline - 50:
            break
        density = 0.4 + random.random() * 0.4
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4(adj):
            candidates.append(adj)

    scored_candidates = []
    for adj in candidates:
        if time.time() > deadline - 40:
            break
        m = compute_triangle_packing_bound(adj)
        ub = n - m
        scored_candidates.append((ub, adj))
        
    scored_candidates.sort(key=lambda x: x[0])
    
    best_adj = None
    best_score = 18
    for ub, adj in scored_candidates:
        if time.time() > deadline - 30:
            break
        score = max_triangle_free_size(adj, time_limit=2.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]

    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(best_adj, time_limit=1.5)

    adj = best_adj[:]
    current_score = best_score
    best_adj_sa = adj[:]
    best_score_sa = current_score
    
    no_improve_count = 0
    max_no_improve = 800
    
    while time.time() < deadline - 8 and no_improve_count < max_no_improve:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        old_adj = adj[:]
        if (adj[i] >> j) & 1:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
        if has_k4_fast(adj):
            adj = old_adj
            no_improve_count += 1
            continue
            
        m = compute_triangle_packing_bound(adj)
        ub = n - m
        
        if ub >= current_score + 2:
            adj = old_adj
            no_improve_count += 1
            continue
            
        new_score = max_triangle_free_size(adj, time_limit=0.7)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score_sa:
                best_score_sa = current_score
                best_adj_sa = adj[:]
            no_improve_count = 0
        else:
            adj = old_adj
            no_improve_count += 1
            
        if no_improve_count > 300 and random.random() < 0.15:
            for _ in range(random.randint(2, 4)):
                i2 = random.randint(0, n - 1)
                j2 = random.randint(0, n - 1)
                if i2 == j2:
                    continue
                if i2 > j2:
                    i2, j2 = j2, i2
                if (adj[i2] >> j2) & 1:
                    adj[i2] &= ~(1 << j2)
                    adj[j2] &= ~(1 << i2)
                else:
                    adj[i2] |= (1 << j2)
                    adj[j2] |= (1 << i2)
            if has_k4_fast(adj):
                adj = best_adj_sa[:]
                current_score = best_score_sa
            else:
                m = compute_triangle_packing_bound(adj)
                ub = n - m
                if ub >= current_score + 2:
                    adj = best_adj_sa[:]
                    current_score = best_score_sa
                    no_improve_count += 1
                    continue
                    
                new_score = max_triangle_free_size(adj, time_limit=0.7)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score_sa:
                        best_score_sa = current_score
                        best_adj_sa = adj[:]
                    no_improve_count = 0
                else:
                    adj = best_adj_sa[:]
                    current_score = best_score_sa
                    no_improve_count += 1

    adj = best_adj_sa[:]
    for i in range(n):
        if time.time() > deadline - 3:
            break
        for j in range(i + 1, n):
            if time.time() > deadline - 3:
                break
            old_adj = adj[:]
            if (adj[i] >> j) & 1:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
            else:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            if has_k4_fast(adj):
                adj = old_adj
                continue
            
            m = compute_triangle_packing_bound(adj)
            ub = n - m
            if ub >= best_score_sa + 2:
                adj = old_adj
                continue
                
            score = max_triangle_free_size(adj, time_limit=0.5)
            if score < best_score_sa:
                best_score_sa = score
                best_adj_sa = adj[:]
            else:
                adj = old_adj
    
    if time.time() < deadline - 5:
        adj = best_adj_sa[:]
        for i in range(n):
            if time.time() > deadline - 3:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 3:
                    break
                old_adj = adj[:]
                if (adj[i] >> j) & 1:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                if has_k4_fast(adj):
                    adj = old_adj
                    continue
                    
                m = compute_triangle_packing_bound(adj)
                ub = n - m
                if ub >= best_score_sa + 2:
                    adj = old_adj
                    continue
                    
                score = max_triangle_free_size(adj, time_limit=0.4)
                if score < best_score_sa:
                    best_score_sa = score
                    best_adj_sa = adj[:]
                else:
                    adj = old_adj

    return edges_from_adj(best_adj_sa)
# EVOLVE-BLOCK-END
