import time
import itertools
from collections import deque

def construct():
    start_time = time.time()
    n = 16
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    edge_set = set(all_edges)
    m = len(all_edges)
    edge_idx = {e: i for i, e in enumerate(all_edges)}
    
    def is_q3(vertices):
        vset = set(vertices)
        edges = []
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                u, v = vertices[i], vertices[j]
                e = (u, v) if u < v else (v, u)
                if e in edge_set:
                    edges.append(e)
        if len(edges) != 12:
            return False
        deg = {v: 0 for v in vertices}
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != 3 for d in deg.values()):
            return False
        adj = {v: set() for v in vertices}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        color = {}
        for start in vertices:
            if start in color:
                continue
            color[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for nb in adj[node]:
                    if nb not in color:
                        color[nb] = color[node] ^ 1
                        queue.append(nb)
                    elif color[nb] == color[node]:
                        return False
        p0 = sum(1 for v in vertices if color[v] == 0)
        if p0 != 4:
            return False
        return True
    
    q3_edge_masks = []
    for combo in itertools.combinations(range(n), 8):
        if is_q3(combo):
            mask = 0
            for i in range(len(combo)):
                for j in range(i+1, len(combo)):
                    u, v = combo[i], combo[j]
                    e = (u, v) if u < v else (v, u)
                    if e in edge_set:
                        mask |= (1 << edge_idx[e])
            if mask != 0:
                q3_edge_masks.append(mask)
    
    edge_q3_list = [[] for _ in range(m)]
    for qi, mask in enumerate(q3_edge_masks):
        for i in range(m):
            if mask & (1 << i):
                edge_q3_list[i].append(qi)
    
    best_count = 0
    best_mask = 0
    
    # Multiple greedy attempts
    import random
    random.seed(42)
    
    for attempt in range(10):
        if attempt == 0:
            edge_order = sorted(range(m), key=lambda i: len(edge_q3_list[i]))
        else:
            edge_order = list(range(m))
            random.shuffle(edge_order)
        
        greedy_mask = 0
        for i in edge_order:
            ok = True
            for qi in edge_q3_list[i]:
                q3mask = q3_edge_masks[qi]
                other = q3mask & ~(1 << i)
                if (other & greedy_mask) == other:
                    ok = False
                    break
            if ok:
                greedy_mask |= (1 << i)
        
        cnt = bin(greedy_mask).count('1')
        if cnt > best_count:
            best_count = cnt
            best_mask = greedy_mask
    
    # Branch and bound
    included_mask = best_mask
    undecided_mask = ((1 << m) - 1) & ~best_mask
    excluded_mask = 0
    
    def search(included_mask, undecided_mask, excluded_mask, depth=0):
        nonlocal best_count, best_mask
        
        if time.time() - start_time > 115:
            return
        
        if undecided_mask == 0:
            cnt = bin(included_mask).count('1')
            if cnt > best_count:
                best_count = cnt
                best_mask = included_mask
            return
        
        # Upper bound
        ub = bin(included_mask).count('1')
        temp = undecided_mask
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            temp ^= lsb
            ok = True
            for qi in edge_q3_list[i]:
                q3mask = q3_edge_masks[qi]
                other = q3mask & ~lsb
                if (other & included_mask) == other:
                    ok = False
                    break
            if ok:
                ub += 1
        
        if ub <= best_count:
            return
        
        # Pick edge
        best_i = -1
        best_active = -1
        temp = undecided_mask
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            temp ^= lsb
            active = 0
            for qi in edge_q3_list[i]:
                if not (q3_edge_masks[qi] & excluded_mask):
                    active += 1
            if active > best_active:
                best_active = active
                best_i = i
        
        if best_i < 0:
            best_i = (undecided_mask & (-undecided_mask)).bit_length() - 1
        
        i = best_i
        bit = 1 << i
        new_undecided = undecided_mask & ~bit
        
        # Try include
        ok = True
        for qi in edge_q3_list[i]:
            q3mask = q3_edge_masks[qi]
            other = q3mask & ~bit
            if (other & included_mask) == other:
                ok = False
                break
        if ok:
            search(included_mask | bit, new_undecided, excluded_mask, depth + 1)
        
        # Try exclude
        search(included_mask, new_undecided, excluded_mask | bit, depth + 1)
    
    search(included_mask, undecided_mask, excluded_mask)
    
    result = []
    for i in range(m):
        if best_mask & (1 << i):
            result.append(list(all_edges[i]))
    return result
