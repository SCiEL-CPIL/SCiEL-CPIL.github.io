import time
import random
import math
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    n = 14
    m = 35

    # Precompute all possible edges (indices)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    P = len(all_pairs)  # 91

    # Generate candidate value sets
    # We want values that are distinct, positive, <= 10000
    # And ideally have many collisions in sums and products
    
    candidate_values = []

    # 1. Small integers (high collision in sums, moderate in products)
    candidate_values.append(list(range(1, 15)))
    candidate_values.append(list(range(2, 16)))
    candidate_values.append(list(range(1, 13)) + [15, 18])
    
    # 2. Divisors of highly composite numbers
    # Divisors of 120: 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120
    divs_120 = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40]
    candidate_values.append(divs_120)
    candidate_values.append(divs_120[:13] + [60])
    
    # Divisors of 360: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360
    divs_360 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360]
    candidate_values.append(divs_360[:14])
    candidate_values.append(divs_360[1:15]) # Skip 1 to avoid 1*x = x collisions? No, 1 is good for sums.
    candidate_values.append(divs_360[2:16])
    
    # 3. Arithmetic Progressions
    for start in [1, 2, 3]:
        for diff in [1, 2, 3]:
            vals = [start + i*diff for i in range(n)]
            if max(vals) <= 10000:
                candidate_values.append(vals)

    # 4. Geometric Progressions (Bad for sums, good for products) - maybe mix?
    # Powers of 2
    candidate_values.append([2**i for i in range(14)])
    # Powers of 3
    candidate_values.append([3**i for i in range(14)])
    # Mixed: Powers of 2 and 3
    vals_mixed = []
    for i in range(14):
        if i % 2 == 0:
            vals_mixed.append(2**(i//2))
        else:
            vals_mixed.append(3**(i//2))
    # Ensure distinct
    if len(set(vals_mixed)) == 14 and max(vals_mixed) <= 10000:
        candidate_values.append(vals_mixed)

    # 5. Random smooth numbers
    rng = random.Random(42)
    for _ in range(20):
        vals = set()
        while len(vals) < n:
            # Pick a random smooth number
            # 2^a 3^b 5^c
            a = rng.randint(0, 5)
            b = rng.randint(0, 4)
            c = rng.randint(0, 3)
            v = (2**a) * (3**b) * (5**c)
            if v <= 10000:
                vals.add(v)
        if len(vals) == n:
            candidate_values.append(sorted(list(vals)))

    # Remove duplicates in candidate_values
    unique_candidates = []
    seen = set()
    for vals in candidate_values:
        # Sort for consistency
        s = tuple(sorted(vals))
        if s not in seen:
            seen.add(s)
            unique_candidates.append(list(s))
    
    candidate_values = unique_candidates

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
                new_s = 1 if s not in used_s else 0
                new_p = 1 if p not in used_p else 0
                ns = len(used_s) + new_s
                np_ = len(used_p) + new_p
                score = max(ns, np_)
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
        """Optimize edges using Simulated Annealing."""
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
        cooling = 0.99995 # Slower cooling for better exploration

        for it in range(max_iters):
            if time.time() > deadline:
                break

            if rng.random() < 0.8:
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
            else:
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

            T *= cooling
            if T < T_min:
                T = T_min

        return list(best_edge_set), best_raw

    best_raw = float('inf')
    best_values = None
    best_edges = None

    # Sort candidates by a heuristic lower bound: 
    # Count distinct sums and products for ALL 91 edges.
    # A lower bound for the 35-edge case is not simply the all-edge count, 
    # but sets with low all-edge distinctness are more likely to have low 35-edge distinctness.
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
        # If lb is small, it's a very promising set
        n_restarts = 10 if lb <= 50 else 5 if lb <= 80 else 2
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Greedy init
            greedy_edges = greedy_init(values, seed=restart * 7919 + vi * 131)
            
            # SA
            # Adjust max_iters based on time remaining
            max_iters = 20000
            sa_edges, sa_raw = simulated_annealing(values, greedy_edges, max_iters, seed=restart * 104729 + vi * 17)
            
            if sa_raw < best_raw:
                best_raw = sa_raw
                best_values = [int(v) for v in values]
                best_edges = [[int(i), int(j)] for i, j in (all_pairs[idx] for idx in sa_edges)]
                
                # Early exit if we hit a very low value
                if best_raw <= 15:
                    break
        if best_raw <= 15:
            break

    if best_values is None:
        # Fallback
        best_values = list(range(1, n + 1))
        best_edges = [[i, j] for i in range(n) for j in range(i + 1, n)][:m]

    return {'values': best_values, 'edges': best_edges}
