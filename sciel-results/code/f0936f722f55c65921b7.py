"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    max_n = 128
    N = max_n
    
    # Precompute lcm table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute conflict structure
    # conflict_pairs[x] = list of (a, b) such that {a, b, x} is a bad triple
    conflict_pairs = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            ab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lcm_table[a][c] == ab and lcm_table[b][c] == ab:
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
    
    # Precompute bitmasks for conflict pairs for faster checking
    conflict_pairs_bits = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a, b in conflict_pairs[x]:
            conflict_pairs_bits[x].append((1 << (a-1), 1 << (b-1)))
    
    # Build weighted conflict graph for spectral approach
    edge_weight = [[0] * (N + 1) for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a, b in conflict_pairs[x]:
            edge_weight[a][b] += 1
            edge_weight[b][a] += 1
    
    # Compute node "conflict degree"
    node_degree = [0] * (N + 1)
    for x in range(1, N + 1):
        node_degree[x] = len(conflict_pairs[x])
    
    # Spectral relaxation
    def spectral_embed():
        rng_spec = random.Random(123)
        v = [rng_spec.uniform(-1, 1) for _ in range(N + 1)]
        v[0] = 0
        for _ in range(100):
            new_v = [0.0] * (N + 1)
            for i in range(1, N + 1):
                s = 0.0
                for j in range(1, N + 1):
                    if edge_weight[i][j] > 0:
                        s += edge_weight[i][j] * v[j]
                new_v[i] = s
            norm = math.sqrt(sum(x*x for x in new_v[1:]))
            if norm < 1e-10:
                break
            for i in range(1, N + 1):
                new_v[i] /= norm
            v = new_v
        return v
    
    def greedy(order, mask=0):
        for x in order:
            ok = True
            for bit_a, bit_b in conflict_pairs_bits[x]:
                if (mask & bit_a) and (mask & bit_b):
                    ok = False
                    break
            if ok:
                mask |= (1 << (x-1))
        return mask
    
    def mask_to_set(mask):
        return [i+1 for i in range(N) if (mask >> i) & 1]
    
    def mask_size(mask):
        return bin(mask).count("1")
    
    def check_conflict(mask, x):
        for bit_a, bit_b in conflict_pairs_bits[x]:
            if (mask & bit_a) and (mask & bit_b):
                return False
        return True
    
    # Branch and bound for maximum independent set in the conflict hypergraph
    # Upper bound: for remaining vertices, count how many can still be added
    # given that each excluded vertex conflicts with some included vertices
    def branch_and_bound(remaining, current_mask, current_size, best_size_ref, deadline):
        """Branch and bound to find larger independent set.
        remaining: sorted list of vertices not yet decided
        current_mask: bitmask of vertices currently included
        Returns (best_mask, best_size) found in this branch.
        """
        if time.time() > deadline:
            return current_mask, current_size
        
        if not remaining:
            return current_mask, current_size
        
        # Upper bound: current_size + len(remaining)
        # But we can tighten: for each vertex in remaining, if it conflicts with 
        # current_mask, it can't be added. Count feasible vertices.
        feasible = []
        for x in remaining:
            if check_conflict(current_mask, x):
                feasible.append(x)
        
        if not feasible:
            return current_mask, current_size
        
        # Upper bound: current_size + len(feasible)
        if current_size + len(feasible) <= best_size_ref[0]:
            return current_mask, best_size_ref[0]
        
        # Choose branching vertex: pick the one with most conflicts among feasible
        # (branch on excluding it first to prune more)
        branch_x = max(feasible, key=lambda x: node_degree[x])
        
        # Branch 1: Include branch_x
        new_mask = current_mask | (1 << (branch_x - 1))
        new_remaining = [x for x in feasible if x != branch_x]
        # Filter remaining that are still compatible with new_mask
        new_remaining = [x for x in new_remaining if check_conflict(new_mask, x)]
        m1, s1 = branch_and_bound(new_remaining, new_mask, current_size + 1, best_size_ref, deadline)
        if s1 > best_size_ref[0]:
            best_size_ref[0] = s1
        
        # Branch 2: Exclude branch_x
        new_remaining2 = [x for x in feasible if x != branch_x]
        m2, s2 = branch_and_bound(new_remaining2, current_mask, current_size, best_size_ref, deadline)
        if s2 > best_size_ref[0]:
            best_size_ref[0] = s2
            if s2 > current_size:
                current_mask = m2
                current_size = s2
        
        return current_mask, current_size
    
    best_mask = 0
    best_size = 0
    
    rng = random.Random(42)
    
    # Phase 1: Spectral-guided initialization
    eigvec = spectral_embed()
    spec_order = sorted(range(1, N + 1), key=lambda x: eigvec[x])
    m = greedy(spec_order)
    sz = mask_size(m)
    if sz > best_size:
        best_size = sz
        best_mask = m
    
    spec_order_rev = sorted(range(1, N + 1), key=lambda x: -eigvec[x])
    m = greedy(spec_order_rev)
    sz = mask_size(m)
    if sz > best_size:
        best_size = sz
        best_mask = m
    
    # Phase 2: Standard greedy with various orderings
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: node_degree[x]),
        sorted(range(1, N + 1), key=lambda x: -node_degree[x]),
    ]
    for order in orders:
        m = greedy(order)
        sz = mask_size(m)
        if sz > best_size:
            best_size = sz
            best_mask = m
    
    # Phase 3: Random restarts with greedy
    for trial in range(50000):
        if time.time() > deadline - 20.0:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        m = greedy(order)
        sz = mask_size(m)
        if sz > best_size:
            best_size = sz
            best_mask = m
    
    # Phase 4: Local search with 1-for-2 and 2-for-2 swaps
    if time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            set_bits = [i+1 for i in range(N) if (best_mask >> i) & 1]
            not_in = [x for x in range(1, N + 1) if not (best_mask & (1 << (x-1)))]
            
            if not not_in:
                break
            
            # 1-for-2 swaps
            for rem_x in set_bits:
                if time.time() > deadline - 10.0:
                    break
                trial_mask = best_mask & ~(1 << (rem_x - 1))
                added = 0
                candidates_sorted = sorted(not_in, key=lambda x: eigvec[x])
                for add_x in candidates_sorted:
                    if time.time() > deadline - 10.0 or added >= 2:
                        break
                    if check_conflict(trial_mask, add_x):
                        trial_mask |= (1 << (add_x - 1))
                        added += 1
                if added >= 1:
                    sz = mask_size(trial_mask)
                    if sz > best_size:
                        best_size = sz
                        best_mask = trial_mask
                        improved = True
                        break
            
            # 2-for-2 swaps with more trials
            if len(set_bits) >= 2 and len(not_in) >= 2:
                for _ in range(200):
                    if time.time() > deadline - 10.0:
                        break
                    rem1, rem2 = rng.sample(set_bits, 2)
                    trial_mask = best_mask & ~(1 << (rem1 - 1)) & ~(1 << (rem2 - 1))
                    added = 0
                    candidates = not_in[:]
                    rng.shuffle(candidates)
                    for add_x in candidates:
                        if added >= 2:
                            break
                        if check_conflict(trial_mask, add_x):
                            trial_mask |= (1 << (add_x - 1))
                            added += 1
                    sz = mask_size(trial_mask)
                    if sz > best_size:
                        best_size = sz
                        best_mask = trial_mask
                        improved = True
                        break
    
    # Phase 5: Branch and bound on the complement (vertices not in best_mask)
    # Try to add more vertices using B&B with the current set as base
    if time.time() < deadline - 5.0:
        not_in = [x for x in range(1, N + 1) if not (best_mask & (1 << (x-1)))]
        if not_in:
            # Sort by ascending conflict degree (easier to add)
            not_in_sorted = sorted(not_in, key=lambda x: node_degree[x])
            best_size_ref = [best_size]
            m_bb, s_bb = branch_and_bound(not_in_sorted, best_mask, best_size, best_size_ref, deadline - 5.0)
            if s_bb > best_size:
                best_size = s_bb
                best_mask = m_bb
    
    # Phase 6: Final polish - try all single additions
    if time.time() < deadline - 1.0:
        not_in = [x for x in range(1, N + 1) if not (best_mask & (1 << (x-1)))]
        for x in not_in:
            if time.time() > deadline - 1.0:
                break
            if check_conflict(best_mask, x):
                best_mask |= (1 << (x-1))
                best_size += 1
    
    # Final validation
    result = mask_to_set(best_mask)
    if not result:
        return [1]
    
    from itertools import combinations
    valid = True
    for a, b, c in combinations(result, 3):
        if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
            valid = False
            break
    
    if not valid:
        m = greedy(list(range(1, N + 1)))
        result = mask_to_set(m)
        if not result:
            return [1]
    
    return result
# EVOLVE-BLOCK-END
