import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    full_mask = (1 << n) - 1
    
    # Popcount lookup table
    popcnt = [0] * 65536
    for i in range(1, 65536):
        popcnt[i] = popcnt[i >> 1] + (i & 1)
    
    # Precompute all 4-sets and their masks
    all_4sets = list(combinations(range(n), 4))
    mask_of = {}
    for s in all_4sets:
        m = 0
        for v in s:
            m |= (1 << v)
        mask_of[s] = m
    
    # Precompute disjoint pairs of 4-sets
    disjoint_pairs = []
    for i, A in enumerate(all_4sets):
        A_mask = mask_of[A]
        for B in all_4sets[i+1:]:
            B_mask = mask_of[B]
            if A_mask & B_mask == 0:
                disjoint_pairs.append((A, B, A_mask, B_mask))
    
    adj_mask = [0] * n
    
    def set_edge(u, v, val):
        if val:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        else:
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
    
    def find_q3():
        """Returns a Q3 as (A, B) if found, else None."""
        for A, B, A_mask, B_mask in disjoint_pairs:
            ok = True
            for a in A:
                if popcnt[adj_mask[a] & B_mask] != 3:
                    ok = False
                    break
            if not ok:
                continue
            for b in B:
                if popcnt[adj_mask[b] & A_mask] != 3:
                    ok = False
                    break
            if ok:
                return (A, B)
        return None
    
    def has_q3():
        return find_q3() is not None
    
    def get_edges():
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj_mask[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    def count_edges():
        return sum(popcnt[adj_mask[i]] for i in range(n)) // 2
    
    def make_q3_free():
        while True:
            if time.time() > deadline - 5:
                return
            q3 = find_q3()
            if q3 is None:
                return
            A, B = q3
            # Remove a random edge from the Q3
            edges_in_q3 = []
            for a in A:
                for b in B:
                    if adj_mask[a] & (1 << b):
                        edges_in_q3.append((a, b))
            e = random.choice(edges_in_q3)
            set_edge(e[0], e[1], False)
    
    def greedy_add():
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj_mask[i] & (1 << j))]
        random.shuffle(missing)
        for (u, v) in missing:
            if time.time() > deadline - 3:
                return
            set_edge(u, v, True)
            if has_q3():
                set_edge(u, v, False)
    
    # Initial: K(3,13)
    adj_mask = [0] * n
    for a in range(3):
        for b in range(3, 16):
            set_edge(a, b, True)
    greedy_add()
    best_edges = get_edges()
    best_count = len(best_edges)
    
    # Random restarts
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        
        adj_mask = [0] * n
        p = random.uniform(0.5, 0.8)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    set_edge(i, j, True)
        
        make_q3_free()
        greedy_add()
        
        current_count = count_edges()
        if current_count > best_count:
            best_count = current_count
            best_edges = get_edges()
    
    return best_edges
