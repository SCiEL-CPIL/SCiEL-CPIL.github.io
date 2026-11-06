"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Candidate moduli in [20, 200]
    candidates = list(range(20, 201))
    
    # Build all candidate nodes (m, r)
    nodes = []
    node_index = {}  # (m, r) -> index
    for m in candidates:
        for r in range(m):
            idx = len(nodes)
            nodes.append((m, r))
            node_index[(m, r)] = idx
    
    n_nodes = len(nodes)
    
    # Precompute weights (1/m)
    weights = [1.0 / nodes[i][0] for i in range(n_nodes)]
    
    # Precompute gcd for all pairs of moduli
    gcd_table = {}
    for i, m1 in enumerate(candidates):
        for j, m2 in enumerate(candidates):
            if (m1, m2) not in gcd_table:
                gcd_table[(m1, m2)] = gcd(m1, m2)
    
    # Build conflict graph using bitsets (Python ints as bitsets)
    # conflict[i] is a bitset of nodes that conflict with node i
    # Two nodes (m1, r1) and (m2, r2) conflict if:
    #   m1 == m2 (same modulus), OR
    #   r1 % gcd(m1, m2) == r2 % gcd(m1, m2)
    
    # For efficiency, group nodes by modulus
    nodes_by_mod = {}
    for idx, (m, r) in enumerate(nodes):
        if m not in nodes_by_mod:
            nodes_by_mod[m] = []
        nodes_by_mod[m].append(idx)
    
    # Build conflict bitsets
    # For each node, conflicts with all nodes of same modulus
    # Plus nodes of other moduli where r1 % g == r2 % g
    conflicts = [0] * n_nodes
    
    # First: same modulus conflicts
    for m, idxs in nodes_by_mod.items():
        mask = 0
        for idx in idxs:
            mask |= (1 << idx)
        for idx in idxs:
            conflicts[idx] |= mask  # includes self, we'll handle that
    
    # Second: cross-modulus conflicts
    # For each pair of moduli, group residues by their value mod gcd
    for i, m1 in enumerate(candidates):
        for j, m2 in enumerate(candidates):
            if i >= j:
                continue
            g = gcd_table[(m1, m2)]
            # Group residues of m1 by r % g
            groups1 = {}
            for idx in nodes_by_mod[m1]:
                r = nodes[idx][1]
                key = r % g
                if key not in groups1:
                    groups1[key] = []
                groups1[key].append(idx)
            
            # Group residues of m2 by r % g
            groups2 = {}
            for idx in nodes_by_mod[m2]:
                r = nodes[idx][1]
                key = r % g
                if key not in groups2:
                    groups2[key] = []
                groups2[key].append(idx)
            
            # For each common residue class, add conflicts
            for key in groups1:
                if key in groups2:
                    for idx1 in groups1[key]:
                        mask = 0
                        for idx2 in groups2[key]:
                            mask |= (1 << idx2)
                        conflicts[idx1] |= mask
                    for idx2 in groups2[key]:
                        mask = 0
                        for idx1 in groups1[key]:
                            mask |= (1 << idx1)
                        conflicts[idx2] |= mask
    
    # Now solve maximum weight independent set using branch and bound
    # We use a recursive B&B that selects nodes one at a time
    # Upper bound: sum of weights of all candidate nodes that don't conflict with current selection
    
    # Sort nodes by weight descending for better pruning
    order = sorted(range(n_nodes), key=lambda i: -weights[i])
    
    best_score = 0.0
    best_set = set()
    
    # For B&B, we process nodes in order. At each step, we decide to include or exclude.
    # To make it efficient, we use the fact that if we include a node, all its conflicts are excluded.
    
    # Use a different approach: maximum weight independent set via branch and bound
    # with candidate set and upper bound
    
    def max_wis_bnb(candidate_mask, current_score, current_set):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
        
        if candidate_mask == 0:
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            return
        
        # Upper bound: current_score + sum of weights of all nodes in candidate_mask
        # (loose bound, but fast to compute with precomputed prefix sums if sorted)
        # For tighter bound, use: for each node in candidate, it can be included only if
        # it doesn't conflict with already selected. Since we build incrementally,
        # candidate_mask already excludes conflicts with selected nodes.
        
        # Compute upper bound as sum of weights in candidate_mask
        # This is expensive for large masks. Use a simpler approach:
        # Just check if current_score + (n_remaining * max_weight) > best_score
        # But that's too loose.
        
        # Better: iterate through candidate nodes, pick the one with highest weight,
        # branch on include/exclude.
        
        # Find the node in candidate_mask with highest weight
        best_idx = -1
        best_w = -1
        temp = candidate_mask
        while temp:
            # Get lowest set bit
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            if weights[idx] > best_w:
                best_w = weights[idx]
                best_idx = idx
            temp ^= lsb
        
        if best_idx == -1:
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            return
        
        # Upper bound: current_score + best_w + sum of other weights (very loose)
        # For a tighter bound, note that if we include best_idx, we remove its conflicts.
        # If we exclude best_idx, we just remove it.
        
        # Branch 1: Include best_idx
        new_mask = candidate_mask & ~(1 << best_idx)
        # Remove all conflicts of best_idx
        new_mask &= ~conflicts[best_idx]
        new_score = current_score + weights[best_idx]
        
        # Upper bound for this branch: new_score + sum of weights in new_mask
        # Approximate: just check if new_score could beat best_score
        if new_score > best_score or True:  # always explore, bound is loose
            max_wis_bnb(new_mask, new_score, current_set | {best_idx})
        
        if time.time() > deadline:
            return
        
        # Branch 2: Exclude best_idx
        new_mask2 = candidate_mask & ~(1 << best_idx)
        max_wis_bnb(new_mask2, current_score, current_set)
    
    # The above B&B is too slow for ~20910 nodes. Need a different approach.
    # 
    # Alternative: Since the problem structure is specific (modular constraints),
    # use a greedy + local search approach but with a better initialization
    # and more aggressive improvement.
    #
    # Actually, let me try a different structural approach:
    # Use the fact that for coprime moduli, any residues work.
    # The constraints only bind moduli that share factors.
    #
    # New approach: Greedy by modulus with smart residue assignment,
    # then use 2-opt style swaps to improve.
    
    def is_disjoint(m_new, r_new, existing):
        """Check if (m_new, r_new) is disjoint from all in existing list of (m, r)."""
        for (m_e, r_e) in existing:
            g = gcd(m_new, m_e)
            if r_new % g == r_e % g:
                return False
        return True
    
    def find_valid_residue(m, existing):
        """Find any valid residue for m given existing assignments."""
        for r in range(m):
            if is_disjoint(m, r, existing):
                return r
        return None
    
    # Phase 1: Greedy construction with multiple orderings
    def greedy_init(order):
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append((m, r))
        return selected
    
    # Try different orderings
    best = []
    best_score = 0.0
    
    # Ordering 1: ascending moduli
    state1 = greedy_init(sorted(candidates))
    score1 = sum(1.0 / m for m, r in state1)
    if score1 > best_score:
        best, best_score = state1, score1
    
    # Ordering 2: descending moduli (try large moduli first)
    state2 = greedy_init(sorted(candidates, reverse=True))
    score2 = sum(1.0 / m for m, r in state2)
    if score2 > best_score:
        best, best_score = state2, score2
    
    # Ordering 3: by number of prime factors (fewer factors = less constrained)
    def count_factors(n):
        count = 0
        d = 2
        temp = n
        while d * d <= temp:
            while temp % d == 0:
                count += 1
                temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count
    
    order3 = sorted(candidates, key=lambda m: count_factors(m))
    state3 = greedy_init(order3)
    score3 = sum(1.0 / m for m, r in state3)
    if score3 > best_score:
        best, best_score = state3, score3
    
    # Phase 2: Aggressive local search
    random.seed(42)
    
    def compute_score(state):
        return sum(1.0 / m for m, r in state)
    
    def try_add(state, m, r):
        used_mods = set(x[0] for x in state)
        if m in used_mods:
            return None
        for (m_e, r_e) in state:
            g = gcd(m, m_e)
            if r % g == r_e % g:
                return None
        return state + [(m, r)]
    
    def try_remove(state, idx):
        return state[:idx] + state[idx+1:]
    
    import math
    
    # Simulated annealing with multiple restarts
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        # Initialize from current best with some perturbation
        state = list(best)
        score = compute_score(state)
        
        # Perturb: remove k random elements
        k = random.randint(3, min(10, len(state) // 2))
        if len(state) > k:
            to_remove = random.sample(range(len(state)), k)
            to_remove.sort(reverse=True)
            for idx in to_remove:
                state = try_remove(state, idx)
            score = compute_score(state)
        
        T = 0.3
        T_min = 0.0001
        cooling = 0.999
        
        while T > T_min and time.time() < deadline - 30:
            for _ in range(300):
                if time.time() > deadline - 30:
                    break
                
                move_type = random.choice(['add', 'add', 'remove', 'change_r', 'swap'])
                
                if move_type == 'add':
                    used_mods = set(m for m, r in state)
                    available = [m for m in candidates if m not in used_mods]
                    if not available:
                        continue
                    # Prefer small moduli
                    available.sort()
                    m_new = random.choice(available[:min(40, len(available))])
                    for attempt in range(30):
                        r_new = random.randint(0, m_new - 1)
                        new_state = try_add(state, m_new, r_new)
                        if new_state:
                            new_score = compute_score(new_state)
                            delta = new_score - score
                            if delta > 0 or random.random() < math.exp(delta / T):
                                state, score = new_state, new_score
                            break
                elif move_type == 'remove':
                    if len(state) <= 1:
                        continue
                    idx = random.randint(0, len(state) - 1)
                    new_state = try_remove(state, idx)
                    new_score = compute_score(new_state)
                    delta = new_score - score
                    if delta > 0 or random.random() < math.exp(delta / T):
                        state, score = new_state, new_score
                elif move_type == 'change_r':
                    if not state:
                        continue
                    idx = random.randint(0, len(state) - 1)
                    m_old, r_old = state[idx]
                    for attempt in range(30):
                        r_new = random.randint(0, m_old - 1)
                        if r_new == r_old:
                            continue
                        valid = True
                        for j, (m_e, r_e) in enumerate(state):
                            if j == idx:
                                continue
                            g = gcd(m_old, m_e)
                            if r_new % g == r_e % g:
                                valid = False
                                break
                        if valid:
                            new_state = state[:idx] + [(m_old, r_new)] + state[idx+1:]
                            new_score = compute_score(new_state)
                            delta = new_score - score
                            if delta > 0 or random.random() < math.exp(delta / T):
                                state, score = new_state, new_score
                            break
                elif move_type == 'swap':
                    if len(state) <= 1:
                        continue
                    idx = random.randint(0, len(state) - 1)
                    m_old, r_old = state[idx]
                    new_state_without = try_remove(state, idx)
                    used_mods = set(m for m, r in new_state_without)
                    available = [m for m in candidates if m not in used_mods]
                    if not available:
                        continue
                    available.sort()
                    for m_new in available[:min(30, len(available))]:
                        for r_new in range(min(m_new, 15)):
                            new_state = try_add(new_state_without, m_new, r_new)
                            if new_state:
                                new_score = compute_score(new_state)
                                delta = new_score - score
                                if delta > 0 or random.random() < math.exp(delta / T):
                                    state, score = new_state, new_score
                                    break
                        if state != new_state_without:
                            break
            
            T *= cooling
        
        if score > best_score:
            best, best_score = state, score
    
    # Phase 3: Final greedy fill - try to add any remaining moduli
    used_mods = set(m for m, r in best)
    remaining = sorted([m for m in candidates if m not in used_mods])
    
    for m in remaining:
        if time.time() > deadline - 5:
            break
        r = find_valid_residue(m, best)
        if r is not None:
            new_state = best + [(m, r)]
            new_score = best_score + 1.0 / m
            if new_score > best_score:
                best, best_score = new_state, new_score
    
    # Convert to required format
    result = [[m, r] for m, r in best]
    return result
# EVOLVE-BLOCK-END
