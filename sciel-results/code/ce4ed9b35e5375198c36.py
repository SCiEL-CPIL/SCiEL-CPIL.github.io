import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    
    # Precompute Q3 masks for vertices 0..7
    # Edge index mapping for 8 vertices (28 edges)
    def get_bit_index(u, v):
        if u > v: u, v = v, u
        return 7*u - u*(u-1)//2 + (v - u - 1)
        
    def idx_to_uv(idx):
        if idx < 7:
            return 0, idx + 1
        elif idx < 13:
            return 1, idx - 7 + 2
        elif idx < 18:
            return 2, idx - 13 + 3
        elif idx < 22:
            return 3, idx - 18 + 4
        elif idx < 25:
            return 4, idx - 22 + 5
        elif idx < 27:
            return 5, idx - 25 + 6
        else:
            return 6, 7

    q3_base_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    
    q3_masks = set()
    for p in permutations(range(8)):
        mask = 0
        for u, v in q3_base_edges:
            gu, gv = p[u], p[v]
            if gu > gv: gu, gv = gv, gu
            idx = get_bit_index(gu, gv)
            mask |= (1 << idx)
        q3_masks.add(mask)
        
    q3_masks_list = list(q3_masks)
    
    all_8sets = list(combinations(range(16), 8))
    
    def get_induced_mask(vs, edge_set):
        mask = 0
        for i in range(8):
            for j in range(i+1, 8):
                u, v = vs[i], vs[j]
                if (min(u, v), max(u, v)) in edge_set:
                    idx = get_bit_index(i, j)
                    mask |= (1 << idx)
        return mask

    def find_q3_violation(edge_set):
        for vs in all_8sets:
            mask = get_induced_mask(vs, edge_set)
            for qm in q3_masks_list:
                if (mask & qm) == qm:
                    return vs, qm
        return None

    def is_q3_free(edge_set):
        return find_q3_violation(edge_set) is None

    random.seed(123)
    
    # Start with a dense graph
    p_start = 0.7
    current_edges = set()
    for i in range(16):
        for j in range(i+1, 16):
            if random.random() < p_start:
                current_edges.add((i, j))
                
    best_edges = set(current_edges)
    best_count = len(current_edges)
    
    # Phase 1: Remove edges to break Q3s
    while time.time() < deadline - 30:
        viol = find_q3_violation(current_edges)
        if not viol:
            break
        vs, qm = viol
        set_bits = [i for i in range(28) if qm & (1 << i)]
        if not set_bits:
            continue
        local_idx = random.choice(set_bits)
        lu, lv = idx_to_uv(local_idx)
        gu, gv = vs[lu], vs[lv]
        edge = (min(gu, gv), max(gu, gv))
        if edge in current_edges:
            current_edges.remove(edge)
            
    if is_q3_free(current_edges):
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
    else:
        # Fallback cleanup
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            viol = find_q3_violation(current_edges)
            if not viol:
                break
            vs, qm = viol
            set_bits = [i for i in range(28) if qm & (1 << i)]
            if not set_bits: continue
            local_idx = random.choice(set_bits)
            lu, lv = idx_to_uv(local_idx)
            gu, gv = vs[lu], vs[lv]
            edge = (min(gu, gv), max(gu, gv))
            if edge in current_edges:
                current_edges.remove(edge)
        if is_q3_free(current_edges):
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = set(current_edges)

    # Phase 2: Greedy Add
    missing_edges = [(i, j) for i in range(16) for j in range(i+1, 16) if (i, j) not in current_edges]
    random.shuffle(missing_edges)
    
    for edge in missing_edges:
        if time.time() > deadline - 5:
            break
        current_edges.add(edge)
        if is_q3_free(current_edges):
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = set(current_edges)
        else:
            current_edges.remove(edge)
            
    return [[u, v] for u, v in best_edges]
