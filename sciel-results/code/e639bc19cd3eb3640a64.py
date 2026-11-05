import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 145.0

    def check_regular(mask, adj):
        k = bin(mask).count('1')
        if k < 6:
            return False
        degs = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = bin(mask & adj[v]).count('1')
            degs.append(deg)
            m ^= lsb
        first = degs[0]
        for d in degs[1:]:
            if d != first:
                return False
        return True

    def full_check(n, adj):
        for k in range(6, n + 1):
            for comb in combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                if check_regular(mask, adj):
                    return False
        return True

    def try_solve(n):
        masks6 = []
        for comb in combinations(range(n), 6):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            masks6.append(mask)
        
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        is_bad = [False] * len(masks6)
        for i, mask in enumerate(masks6):
            if check_regular(mask, adj):
                is_bad[i] = True
        bad_count = sum(is_bad)
        
        best_adj = adj[:]
        best_bad_count = bad_count
        
        if bad_count == 0 and full_check(n, adj):
            return adj

        masks6_by_pair = {}
        for i, mask in enumerate(masks6):
            verts = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            for a in range(len(verts)):
                for b in range(a+1, len(verts)):
                    u = verts[a]
                    v = verts[b]
                    if u > v: u, v = v, u
                    if (u, v) not in masks6_by_pair:
                        masks6_by_pair[(u, v)] = []
                    masks6_by_pair[(u, v)].append(i)
        
        no_improve = 0
        max_no_improve = 500
        
        while time.time() - start_time < time_limit:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            if adj[u] & (1 << v):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                edge_present = False
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_present = True
            
            affected_indices = masks6_by_pair.get((u, v), [])
            delta = 0
            for idx in affected_indices:
                mask = masks6[idx]
                was_bad = is_bad[idx]
                now_bad = check_regular(mask, adj)
                if was_bad and not now_bad:
                    delta -= 1
                    is_bad[idx] = False
                elif not was_bad and now_bad:
                    delta += 1
                    is_bad[idx] = True
            
            new_bad_count = bad_count + delta
            
            if new_bad_count < bad_count:
                bad_count = new_bad_count
                no_improve = 0
                if bad_count == 0:
                    if full_check(n, adj):
                        return adj
            elif new_bad_count == bad_count:
                if random.random() < 0.5:
                    bad_count = new_bad_count
                    no_improve = 0
            else:
                # Reject
                if edge_present:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                
                for idx in affected_indices:
                    mask = masks6[idx]
                    is_bad[idx] = check_regular(mask, adj)
                
                no_improve += 1
                if no_improve > max_no_improve:
                    # Restart
                    adj = [0] * n
                    for i in range(n):
                        for j in range(i+1, n):
                            if random.random() < 0.5:
                                adj[i] |= (1 << j)
                                adj[j] |= (1 << i)
                    is_bad = [False] * len(masks6)
                    for i, mask in enumerate(masks6):
                        if check_regular(mask, adj):
                            is_bad[i] = True
                    bad_count = sum(is_bad)
                    no_improve = 0
                    if bad_count == 0 and full_check(n, adj):
                        return adj

            if bad_count < best_bad_count:
                best_bad_count = bad_count
                best_adj = adj[:]

        return None

    adj21 = try_solve(21)
    if adj21 is not None:
        edges = []
        for i in range(21):
            for j in range(i+1, 21):
                if adj21[i] & (1 << j):
                    edges.append([i, j])
        return {'n': 21, 'edges': edges}

    adj20 = try_solve(20)
    if adj20 is not None:
        edges = []
        for i in range(20):
            for j in range(i+1, 20):
                if adj20[i] & (1 << j):
                    edges.append([i, j])
        return {'n': 20, 'edges': edges}
    
    return {'n': 5, 'edges': []}
