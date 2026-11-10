import time
import random
import math
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    n = 14
    m = 35

    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    P = len(all_pairs)  # 91

    # Candidate value sets - focus on collision-rich sets
    candidate_values = [
        # Small numbers with high multiplicative structure
        [1, 2, 3, 4, 6, 8, 9, 12, 15, 18, 20, 24, 30, 36],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 40],
        [3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30],
        [1, 2, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30],
        [2, 3, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 27, 30],
        # Powers of 2
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        # Arithmetic Progression
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        # Mixed
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 144, 192],
        # New: highly composite numbers
        [1, 2, 3, 4, 6, 12, 24, 36, 48, 60, 72, 84, 96, 120],
        [1, 2, 4, 6, 12, 18, 24, 36, 48, 60, 72, 84, 96, 108],
        [2, 3, 6, 12, 18, 24, 36, 48, 60, 72, 84, 96, 108, 120],
        # New: small primes and composites
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 25, 27, 29, 31, 33],
        # New: geometric-like with small base
        [1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72, 96, 144],
    ]

    def evaluate_edges(values, edge_indices):
        """Calculate raw score for a given set of edge indices."""
        sums = set()
        prods = set()
        for idx in edge_indices:
            i, j = all_pairs[idx]
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def greedy_init(values, seed):
        """Greedy initialization: pick edges that minimize new distinct labels."""
        rng = random.Random(seed)
        ps = [values[i] + values[j] for i, j in all_pairs]
        pp = [values[i] * values[j] for i, j in all_pairs]
        
        used_s = set()
        used_p = set()
        selected = []
        remaining = list(range(P))
        rng.shuffle(remaining)
        
        for _ in range(m):
            best_idx = -1
            best_score = float('inf')
            best_tie = -1
            for k in remaining:
                s = ps[k]
                p = pp[k]
                # If we add this edge, how many new distinct sums/prods do we get?
                new_s = 1 if s not in used_s else 0
                new_p = 1 if p not in used_p else 0
                # We want to minimize the resulting max count
                ns = len(used_s) + new_s
                np_ = len(used_p) + new_p
                score = max(ns, np_)
                # Tie breaker: prefer edges that add to existing sets (new_s=0, new_p=0)
                tie = new_s + new_p
                
                if score < best_score or (score == best_score and tie < best_tie):
                    best_score = score
                    best_tie = tie
                    best_idx = k
            
            selected.append(best_idx)
            used_s.add(ps[best_idx])
            used_p.add(pp[best_idx])
            remaining.remove(best_idx)
        
        return selected

    def simulated_annealing(values, initial_edges, max_iters, seed):
        """Optimize edges using Simulated Annealing with improved moves."""
        rng = random.Random(seed)
        ps = [values[i] + values[j] for i, j in all_pairs]
        pp = [values[i] * values[j] for i, j in all_pairs]

        edge_set = set(initial_edges)
        unselected = set(range(P)) - edge_set
        
        sum_counts = Counter()
        prod_counts = Counter()
        for idx in edge_set:
            sum_counts[ps[idx]] += 1
            prod_counts[pp[idx]] += 1

        def get_raw():
            return max(len(sum_counts), len(prod_counts))

        current_raw = get_raw()
        best_raw = current_raw
        best_edge_set = set(edge_set)

        T = 5.0
        T_min = 0.01
        cooling = 0.99995

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Choose move type: 1-swap (70%), 2-swap (20%), 3-swap (10%)
            move_type = rng.random()
            
            if move_type < 0.7:
                # 1-swap
                if not edge_set or not unselected: continue
                remove_idx = rng.choice(list(edge_set))
                add_idx = rng.choice(list(unselected))
                if remove_idx == add_idx: continue

                rs, rp = ps[remove_idx], pp[remove_idx]
                as_, ap = ps[add_idx], pp[add_idx]

                # Apply
                sum_counts[rs] -= 1
                if sum_counts[rs] == 0: del sum_counts[rs]
                prod_counts[rp] -= 1
                if prod_counts[rp] == 0: del prod_counts[rp]
                
                sum_counts[as_] += 1
                prod_counts[ap] += 1

                new_raw = get_raw()
                delta = new_raw - current_raw
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    edge_set.remove(remove_idx)
                    edge_set.add(add_idx)
                    unselected.add(remove_idx)
                    unselected.remove(add_idx)
                    current_raw = new_raw
                    if current_raw < best_raw:
                        best_raw = current_raw
                        best_edge_set = set(edge_set)
                else:
                    # Revert
                    sum_counts[as_] -= 1
                    if sum_counts[as_] == 0: del sum_counts[as_]
                    prod_counts[ap] -= 1
                    if prod_counts[ap] == 0: del prod_counts[ap]
                    
                    sum_counts[rs] += 1
                    prod_counts[rp] += 1
            elif move_type < 0.9:
                # 2-swap
                if len(edge_set) < 2 or len(unselected) < 2: continue
                rem1, rem2 = rng.sample(list(edge_set), 2)
                add1, add2 = rng.sample(list(unselected), 2)
                
                s1, p1 = ps[rem1], pp[rem1]
                s2, p2 = ps[rem2], pp[rem2]
                sa1, pa1 = ps[add1], pp[add1]
                sa2, pa2 = ps[add2], pp[add2]

                # Apply
                sum_counts[s1] -= 1
                if sum_counts[s1] == 0: del sum_counts[s1]
                prod_counts[p1] -= 1
                if prod_counts[p1] == 0: del prod_counts[p1]
                
                sum_counts[s2] -= 1
                if sum_counts[s2] == 0: del sum_counts[s2]
                prod_counts[p2] -= 1
                if prod_counts[p2] == 0: del prod_counts[p2]
                
                sum_counts[sa1] += 1
                prod_counts[pa1] += 1
                sum_counts[sa2] += 1
                prod_counts[pa2] += 1

                new_raw = get_raw()
                delta = new_raw - current_raw
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    edge_set.discard(rem1)
                    edge_set.discard(rem2)
                    edge_set.add(add1)
                    edge_set.add(add2)
                    unselected.add(rem1)
                    unselected.add(rem2)
                    unselected.discard(add1)
                    unselected.discard(add2)
                    current_raw = new_raw
                    if current_raw < best_raw:
                        best_raw = current_raw
                        best_edge_set = set(edge_set)
                else:
                    # Revert
                    sum_counts[sa1] -= 1
                    if sum_counts[sa1] == 0: del sum_counts[sa1]
                    prod_counts[pa1] -= 1
                    if prod_counts[pa1] == 0: del prod_counts[pa1]
                    
                    sum_counts[sa2] -= 1
                    if sum_counts[sa2] == 0: del sum_counts[sa2]
                    prod_counts[pa2] -= 1
                    if prod_counts[pa2] == 0: del prod_counts[pa2]
                    
                    sum_counts[s1] += 1
                    prod_counts[p1] += 1
                    sum_counts[s2] += 1
                    prod_counts[p2] += 1
            else:
                # 3-swap: remove 3, add 3
                if len(edge_set) < 3 or len(unselected) < 3: continue
                rem1, rem2, rem3 = rng.sample(list(edge_set), 3)
                add1, add2, add3 = rng.sample(list(unselected), 3)
                
                # Apply all removals
                for r in [rem1, rem2, rem3]:
                    sum_counts[ps[r]] -= 1
                    if sum_counts[ps[r]] == 0: del sum_counts[ps[r]]
                    prod_counts[pp[r]] -= 1
                    if prod_counts[pp[r]] == 0: del prod_counts[pp[r]]
                
                # Apply all additions
                for a in [add1, add2, add3]:
                    sum_counts[ps[a]] += 1
                    prod_counts[pp[a]] += 1

                new_raw = get_raw()
                delta = new_raw - current_raw
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    for r in [rem1, rem2, rem3]:
                        edge_set.discard(r)
                        unselected.add(r)
                    for a in [add1, add2, add3]:
                        edge_set.add(a)
                        unselected.discard(a)
                    current_raw = new_raw
                    if current_raw < best_raw:
                        best_raw = current_raw
                        best_edge_set = set(edge_set)
                else:
                    # Revert
                    for a in [add1, add2, add3]:
                        sum_counts[ps[a]] -= 1
                        if sum_counts[ps[a]] == 0: del sum_counts[ps[a]]
                        prod_counts[pp[a]] -= 1
                        if prod_counts[pp[a]] == 0: del prod_counts[pp[a]]
                    
                    for r in [rem1, rem2, rem3]:
                        sum_counts[ps[r]] += 1
                        prod_counts[pp[r]] += 1

            T *= cooling
            if T < T_min:
                T = T_min

        return list(best_edge_set), best_raw

    def polish_pass(values, edge_indices, max_iters, seed):
        """Final high-precision polish: try all single-edge swaps to find improvements."""
        rng = random.Random(seed)
        ps = [values[i] + values[j] for i, j in all_pairs]
        pp = [values[i] * values[j] for i, j in all_pairs]

        edge_set = set(edge_indices)
        unselected = set(range(P)) - edge_set
        
        sum_counts = Counter()
        prod_counts = Counter()
        for idx in edge_set:
            sum_counts[ps[idx]] += 1
            prod_counts[pp[idx]] += 1

        def get_raw():
            return max(len(sum_counts), len(prod_counts))

        current_raw = get_raw()
        best_raw = current_raw
        best_edge_set = set(edge_set)
        
        # Try all possible single swaps
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            for remove_idx in list(edge_set):
                for add_idx in list(unselected):
                    rs, rp = ps[remove_idx], pp[remove_idx]
                    as_, ap = ps[add_idx], pp[add_idx]

                    # Apply
                    sum_counts[rs] -= 1
                    if sum_counts[rs] == 0: del sum_counts[rs]
                    prod_counts[rp] -= 1
                    if prod_counts[rp] == 0: del prod_counts[rp]
                    
                    sum_counts[as_] += 1
                    prod_counts[ap] += 1

                    new_raw = get_raw()
                    
                    if new_raw < current_raw:
                        edge_set.remove(remove_idx)
                        edge_set.add(add_idx)
                        unselected.add(remove_idx)
                        unselected.remove(add_idx)
                        current_raw = new_raw
                        if current_raw < best_raw:
                            best_raw = current_raw
                            best_edge_set = set(edge_set)
                        improved = True
                    else:
                        # Revert
                        sum_counts[as_] -= 1
                        if sum_counts[as_] == 0: del sum_counts[as_]
                        prod_counts[ap] -= 1
                        if prod_counts[ap] == 0: del prod_counts[ap]
                        
                        sum_counts[rs] += 1
                        prod_counts[rp] += 1
                    
                    if time.time() > deadline:
                        break
                
                if time.time() > deadline:
                    break
            
            if not improved:
                break

        return list(best_edge_set), best_raw

    best_raw = float('inf')
    best_values = None
    best_edges = None

    # Sort candidates by a heuristic lower bound
    def lower_bound(values):
        ps = set()
        pp = set()
        for i, j in all_pairs:
            ps.add(values[i] + values[j])
            pp.add(values[i] * values[j])
        return max(len(ps), len(pp))

    scored_candidates = []
    for vi, values in enumerate(candidate_values):
        if len(set(values)) != n or any(v < 1 or v > 10000 for v in values):
            continue
        lb = lower_bound(values)
        scored_candidates.append((lb, vi, values))
    
    scored_candidates.sort(key=lambda x: x[0])

    for lb, vi, values in scored_candidates:
        if time.time() > deadline:
            break
        
        # More restarts for better candidates
        n_restarts = 8 if lb <= 40 else 4
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Greedy init
            greedy_edges = greedy_init(values, seed=restart * 7919 + vi * 131)
            
            # SA
            max_iters = 80000
            sa_edges, sa_raw = simulated_annealing(values, greedy_edges, max_iters, seed=restart * 104729 + vi * 17)
            
            # Polish pass
            polished_edges, polished_raw = polish_pass(values, sa_edges, 5000, seed=restart * 31337 + vi * 23)
            
            if polished_raw < best_raw:
                best_raw = polished_raw
                best_values = [int(v) for v in values]
                best_edges = [[int(i), int(j)] for i, j in (all_pairs[idx] for idx in polished_edges)]
                
                # Early exit if we hit a very low value
                if best_raw <= 18:
                    break
        if best_raw <= 18:
            break

    if best_values is None:
        # Fallback
        best_values = list(range(1, n + 1))
        best_edges = [[i, j] for i in range(n) for j in range(i + 1, n)][:m]

    return {'values': best_values, 'edges': best_edges}
