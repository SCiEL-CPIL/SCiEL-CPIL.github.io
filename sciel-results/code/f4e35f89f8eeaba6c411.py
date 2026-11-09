import time
import random
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 145.0

    def time_left():
        return DEADLINE - time.time()

    # ---------- deterministic C4-free fallback (10 triangles, alpha=10) ----------
    def fallback():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    def bit_count(x):
        return bin(x).count('1')

    # ---------- C4-free graph construction using bitsets for speed ----------
    def build_c4free(n, seed):
        rng = random.Random(seed)
        adj_bits = [0] * n
        # Try multiple random orderings
        best_adj_bits = None
        best_density = -1
        
        for attempt in range(5): # Increased attempts due to speedup
            adj_try = [0] * n
            order = list(range(n))
            rng.shuffle(order)
            
            for idx, i in enumerate(order):
                for j in order[idx+1:]:
                    # Check C4: common neighbors of i and j
                    # adj_try[i] contains neighbors of i
                    # adj_try[j] contains neighbors of j
                    common = adj_try[i] & adj_try[j]
                    if bit_count(common) <= 1:
                        # Add edge i-j
                        adj_try[i] |= (1 << j)
                        adj_try[j] |= (1 << i)
            
            density = sum(bit_count(b) for b in adj_try)
            if density > best_density:
                best_density = density
                best_adj_bits = adj_try
        
        return best_adj_bits

    def densify(adj_bits, n, seed):
        rng = random.Random(seed)
        changed = True
        while changed and time.time() < DEADLINE - 2.0:
            changed = False
            pairs = list(combinations(range(n), 2))
            rng.shuffle(pairs)
            for i, j in pairs:
                if (adj_bits[i] >> j) & 1:
                    continue
                common = adj_bits[i] & adj_bits[j]
                if bit_count(common) <= 1:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
                    changed = True
        return adj_bits

    # ---------- Fast Exact Alpha via Bitset Branch and Bound ----------
    def exact_alpha(adj_bits, n):
        if n == 0:
            return 0
        
        best_alpha = 0
        
        def solve(candidates_mask, current_size):
            nonlocal best_alpha
            if candidates_mask == 0:
                if current_size > best_alpha:
                    best_alpha = current_size
                return
            
            # Pruning
            if current_size + bit_count(candidates_mask) <= best_alpha:
                return
            
            # Heuristic: Pick vertex with max degree in induced subgraph
            v = -1
            max_deg = -1
            temp_mask = candidates_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                idx = lsb.bit_length() - 1
                deg = bit_count(adj_bits[idx] & candidates_mask)
                if deg > max_deg:
                    max_deg = deg
                    v = idx
                temp_mask ^= lsb
            
            # Branch 1: Include v
            # Neighbors of v in candidates
            neighbors_in_candidates = adj_bits[v] & candidates_mask
            # New candidates: remove v and its neighbors
            new_mask_include = candidates_mask & ~(neighbors_in_candidates | (1 << v))
            solve(new_mask_include, current_size + 1)
            
            # Branch 2: Exclude v
            new_mask_exclude = candidates_mask & ~(1 << v)
            solve(new_mask_exclude, current_size)
            
        solve((1 << n) - 1, 0)
        return best_alpha

    # ---------- Candidate validation ----------
    def fast_ok(adj_bits, n):
        # C4 check
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_bits[i] >> j) & 1:
                    if bit_count(adj_bits[i] & adj_bits[j]) > 1:
                        return False
        # Greedy MIS check (fast filter)
        rng = random.Random(999)
        for _ in range(10):
            order = list(range(n))
            rng.shuffle(order)
            indep = []
            in_set = 0
            for v in order:
                if (in_set >> v) & 1:
                    continue
                # Check if v is connected to any in indep
                # indep is a list, but we can maintain a mask of indep
                # Actually, just checking against the list is fine for n<=43
                if all(((adj_bits[v] >> u) & 1) == 0 for u in indep):
                    indep.append(v)
                    in_set |= (1 << v)
            if len(indep) >= 11:
                return False
        return True

    # ---------- Main Search ----------
    best = None
    target_ns = [43, 42, 41, 40, 39, 38, 37]
    trial = 0
    
    for n in target_ns:
        if time.time() >= DEADLINE - 3.0:
            break
        
        # Budget per n
        max_trials = 100 if n <= 40 else 50
        
        for trial_n in range(max_trials):
            if time.time() >= DEADLINE - 3.0:
                break
            trial += 1
            adj_bits = build_c4free(n, trial)
            adj_bits = densify(adj_bits, n, trial)
            
            if fast_ok(adj_bits, n):
                # Verify with exact alpha
                alpha = exact_alpha(adj_bits, n)
                if alpha < 11:
                    best = (n, adj_bits)
                    break
        
        if best is not None:
            break

    if best is None:
        return fallback()

    n, adj_bits = best
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}
