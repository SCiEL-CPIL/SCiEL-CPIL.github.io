import time
import random

def _has_c5_fast(n, adj_mat):
    """Check if graph has a C5. adj_mat is a list of bitmasks (ints).
    Returns True if C5 exists."""
    # For each vertex a, look at pairs of neighbors (b,c) of a.
    # A C5 through a-b-w-x-c-a requires:
    #   w adjacent to b and c (w != a)
    #   x adjacent to w and c (x != a,b,w,c)
    #   and a adjacent to b and c (already true by construction)
    
    deg_order = sorted(range(n), key=lambda v: adj_mat[v].bit_count())
    
    for a in deg_order:
        na = adj_mat[a]
        b_mask = na
        while b_mask:
            b_bit = b_mask & -b_mask
            b = b_bit.bit_length() - 1
            b_mask ^= b_bit
            c_mask = na & ~((1 << (b+1)) - 1)
            while c_mask:
                c_bit = c_mask & -c_mask
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                common_bc = adj_mat[b] & adj_mat[c]
                common_bc &= ~(1 << a)
                while common_bc:
                    w_bit = common_bc & -common_bc
                    w = w_bit.bit_length() - 1
                    common_bc ^= w_bit
                    common_wc = adj_mat[w] & adj_mat[c]
                    mask_forbid = (1 << a) | (1 << b) | (1 << w) | (1 << c)
                    if common_wc & ~mask_forbid:
                        return True
    return False

def _has_independent_set_k(n, adj_mat, k=9):
    """Check if graph has independent set of size k. adj_mat is list of bitmasks.
    Returns True if such a set exists."""
    found = [False]
    
    def search(chosen_count, candidates):
        if found[0]:
            return
        if chosen_count >= k:
            found[0] = True
            return
        if chosen_count + candidates.bit_count() < k:
            return
        if candidates == 0:
            return
        
        best_v = -1
        best_deg = -1
        temp = candidates
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            temp &= temp - 1
            deg = (adj_mat[v] & candidates).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        v = best_v
        bit = 1 << v
        
        search(chosen_count, candidates & ~bit)
        if found[0]:
            return
        
        new_candidates = candidates & ~(adj_mat[v] | bit)
        search(chosen_count + 1, new_candidates)
    
    search(0, (1 << n) - 1)
    return found[0]

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def make_8k4_edges():
        edges = []
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i+1, 4):
                    edges.append([base+i, base+j])
        return edges

    best_n = 32
    best_edges = make_8k4_edges()

    def verify(n, edges):
        adj_mat = [0] * n
        for u, v in edges:
            adj_mat[u] |= (1 << v)
            adj_mat[v] |= (1 << u)
        if _has_c5_fast(n, adj_mat):
            return False, True
        if _has_independent_set_k(n, adj_mat, 9):
            return True, False
        return True, True

    # Structured constructions: 8 K4s + extra vertices
    def try_n33_structured():
        edges = make_8k4_edges()
        for v in range(4):
            edges.append([32, v])
        for b in range(1, 8):
            edges.append([32, 4*b])
        return edges

    def try_n34_structured():
        edges = make_8k4_edges()
        for v in range(4):
            edges.append([32, v])
        for b in range(1, 8):
            edges.append([32, 4*b])
        for v in range(4, 8):
            edges.append([33, v])
        edges.append([33, 0])
        for b in range(2, 8):
            edges.append([33, 4*b])
        edges.append([32, 33])
        return edges

    def try_n35_structured():
        edges = make_8k4_edges()
        for v in range(4):
            edges.append([32, v])
        for b in range(1, 8):
            edges.append([32, 4*b])
        for v in range(4, 8):
            edges.append([33, v])
        edges.append([33, 0])
        for b in range(2, 8):
            edges.append([33, 4*b])
        for v in range(8, 12):
            edges.append([34, v])
        edges.append([34, 0])
        edges.append([34, 4])
        for b in range(3, 8):
            edges.append([34, 4*b])
        edges.append([32, 33])
        edges.append([32, 34])
        edges.append([33, 34])
        return edges

    for target_n, builder in [(35, try_n35_structured), (34, try_n34_structured), (33, try_n33_structured)]:
        if time.time() >= deadline - 5:
            break
        if target_n <= best_n:
            continue
        edges = builder()
        c5_free, alpha_ok = verify(target_n, edges)
        if c5_free and alpha_ok:
            best_n = target_n
            best_edges = edges
            break

    # Randomized search with more trials
    if best_n < 35 and time.time() < deadline - 10:
        for target_n in [35, 34, 33]:
            if target_n <= best_n:
                continue
            if time.time() >= deadline - 5:
                break
            
            n_extra = target_n - 32
            for trial in range(15000):
                if time.time() >= deadline - 5:
                    break
                
                random.seed(trial * 7919 + target_n * 104729)
                edges = make_8k4_edges()
                
                for k in range(n_extra):
                    v = 32 + k
                    chosen_k4 = random.randint(0, 7)
                    base = 4 * chosen_k4
                    for i in range(4):
                        edges.append([v, base + i])
                    for b in range(8):
                        if b != chosen_k4:
                            edges.append([v, 4*b + random.randint(0, 3)])
                    for w in range(32, 32+k):
                        if random.random() < 0.6:
                            edges.append([v, w])
                
                c5_free, alpha_ok = verify(target_n, edges)
                if c5_free and alpha_ok:
                    best_n = target_n
                    best_edges = edges
                    break
            if best_n >= target_n:
                break

    if best_n < 32:
        best_n = 32
        best_edges = make_8k4_edges()

    return {'n': best_n, 'edges': best_edges}
