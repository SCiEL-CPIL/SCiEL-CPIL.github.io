import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin against 90s limit
    
    N = 256
    all_nodes = list(range(2, N + 1))
    n_total = len(all_nodes)
    node_to_idx = {v: i for i, v in enumerate(all_nodes)}
    
    # Precompute conflict masks: conflict if lcm(a,b) <= 256
    conflict_mask = [0] * n_total
    for i in range(n_total):
        a = all_nodes[i]
        mask = 0
        for j in range(n_total):
            if i == j:
                continue
            b = all_nodes[j]
            if a * b <= 256 * gcd(a, b):
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Compatibility mask
    compat_mask = [0] * n_total
    for i in range(n_total):
        compat_mask[i] = ((1 << n_total) - 1) & ~conflict_mask[i]
    
    conflict_count = [bin(conflict_mask[i]).count('1') for i in range(n_total)]
    
    # Number-theoretic reduction:
    # If x is in the solution, then no multiple of x (that is <= 256) can be in the solution.
    # Specifically, for any x, all multiples kx where k>=2 and kx<=256 conflict with x.
    # This means we can identify "forced exclusions" based on structure.
    
    # Key insight: For primes p > 128, they have no multiples in [2,256] other than themselves,
    # so they only conflict with numbers that share a factor... wait, no.
    # lcm(p, q) > 256 for all q if p is prime and p > 256/2 = 128? 
    # Actually lcm(p,q) = p*q/gcd(p,q). If p is prime and p doesn't divide q, gcd=1, so lcm=p*q.
    # If p > 128 and q >= 2, then p*q > 256. So primes > 128 are compatible with everything!
    # Let's verify: p=131, q=2: lcm=262>256. p=131, q=130: gcd=1, lcm=17030>256.
    # So all primes in (128, 256] are compatible with ALL other numbers.
    
    # Similarly, for any number x where x > 128:
    # lcm(x, y) > 256 for all y? Not necessarily. If x=129=3*43, y=3: lcm=129<=256. Conflict.
    # But if x is prime and x>128, then for any y, gcd(x,y) is either 1 or x.
    # If gcd=x, then y is a multiple of x, but x>128 so 2x>256, so no such y in [2,256].
    # So gcd(x,y)=1 always, and lcm(x,y)=x*y >= x*2 > 256.
    
    # So all primes in (128, 256] are "universally compatible" - they can always be added.
    # Let's find these primes.
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    universal_primes = [v for v in all_nodes if v > 128 and is_prime(v)]
    universal_idx = [node_to_idx[v] for v in universal_primes]
    
    # These can always be included. Let's start with them.
    # Now, for the remaining nodes, we need to find a compatible subset.
    # But wait - the universal primes are compatible with everything, so they don't affect
    # the compatibility of other nodes with each other.
    
    # Let's identify which nodes are "forced" in some sense.
    # Actually, the key structural insight is:
    # If we include x, we cannot include any multiple of x (that is <= 256).
    # This creates a partial order.
    
    # For the search, let's use a smarter approach:
    # 1. Always include all universal primes (primes > 128)
    # 2. For the rest, use branch and bound
    
    # Multiple greedy orderings
    scores = [(1.0 / all_nodes[i]) / (1 + conflict_count[i]) for i in range(n_total)]
    order1 = sorted(range(n_total), key=lambda i: scores[i], reverse=True)
    order2 = sorted(range(n_total), key=lambda i: 1.0 / all_nodes[i], reverse=True)
    order3 = sorted(range(n_total), key=lambda i: conflict_count[i])
    scores4 = [(1.0 / all_nodes[i]) / (1 + conflict_count[i])**0.5 for i in range(n_total)]
    order4 = sorted(range(n_total), key=lambda i: scores4[i], reverse=True)
    scores5 = [(1.0 / all_nodes[i]) / max(1, conflict_count[i]) for i in range(n_total)]
    order5 = sorted(range(n_total), key=lambda i: scores5[i], reverse=True)
    
    best_heuristic_subset = []
    best_heuristic_value = 0.0
    
    for order in [order1, order2, order3, order4, order5]:
        sel = []
        used = 0
        for idx in order:
            if used & conflict_mask[idx]:
                continue
            sel.append(all_nodes[idx])
            used |= (1 << idx)
        val = sum(1.0 / v for v in sel)
        if val > best_heuristic_value:
            best_heuristic_value = val
            best_heuristic_subset = sel[:]
    
    # Local search
    def local_search(selected, deadline):
        selected = selected[:]
        used = 0
        for v in selected:
            used |= (1 << node_to_idx[v])
        
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            
            # 1. Try adding nodes
            for i in range(n_total):
                if (used >> i) & 1:
                    continue
                if used & conflict_mask[i]:
                    continue
                selected.append(all_nodes[i])
                used |= (1 << i)
                improved = True
            
            # 2. Try 1-swap
            non_selected = [i for i in range(n_total) if not ((used >> i) & 1)]
            for s_idx, s in enumerate(selected):
                s_i = node_to_idx[s]
                used_without_s = used & ~(1 << s_i)
                best_n = None
                best_gain = 0
                for n_i in non_selected:
                    n_val = all_nodes[n_i]
                    if n_val >= s:
                        continue
                    if used_without_s & conflict_mask[n_i]:
                        continue
                    gain = 1.0 / n_val - 1.0 / s
                    if gain > best_gain:
                        best_gain = gain
                        best_n = n_i
                if best_n is not None:
                    selected[s_idx] = all_nodes[best_n]
                    used = used_without_s | (1 << best_n)
                    improved = True
                    non_selected = [i for i in range(n_total) if not ((used >> i) & 1)]
                    break
            
            # 3. Try 2-swap
            if not improved and len(selected) >= 2:
                n_sel = len(selected)
                for i in range(n_sel):
                    if (time.time() - start_time) > deadline:
                        break
                    for j in range(i + 1, n_sel):
                        if (time.time() - start_time) > deadline:
                            break
                        si = node_to_idx[selected[i]]
                        sj = node_to_idx[selected[j]]
                        used_without = used & ~(1 << si) & ~(1 << sj)
                        removed_sum = 1.0 / selected[i] + 1.0 / selected[j]
                        
                        best_pair = None
                        best_gain = 0
                        non_sel_list = [k for k in range(n_total) if not ((used_without >> k) & 1)]
                        
                        for a_idx in range(len(non_sel_list)):
                            if (time.time() - start_time) > deadline:
                                break
                            na = non_sel_list[a_idx]
                            na_val = all_nodes[na]
                            if used_without & conflict_mask[na]:
                                continue
                            for b_idx in range(a_idx + 1, len(non_sel_list)):
                                nb = non_sel_list[b_idx]
                                nb_val = all_nodes[nb]
                                if (used_without | (1 << na)) & conflict_mask[nb]:
                                    continue
                                gain = (1.0 / na_val + 1.0 / nb_val) - removed_sum
                                if gain > best_gain:
                                    best_gain = gain
                                    best_pair = (na, nb)
                        
                        if best_pair is not None:
                            na, nb = best_pair
                            selected[i] = all_nodes[na]
                            selected[j] = all_nodes[nb]
                            used = used_without | (1 << na) | (1 << nb)
                            improved = True
                            break
                    if improved:
                        break
        
        return selected
    
    # Run local search from heuristic
    best_subset = local_search(best_heuristic_subset, deadline - 25)
    best_value = sum(1.0 / v for v in best_subset)
    
    # Branch and bound for maximum weight clique
    def max_weight_clique(weights, compat, n, deadline, init_clique=None):
        best_clique = init_clique[:] if init_clique else []
        best_weight = sum(weights[i] for i in best_clique)
        
        def weighted_color_bound(candidates, weights):
            if not candidates:
                return 0.0
            
            sorted_cands = sorted(candidates, key=lambda x: weights[x], reverse=True)
            
            color_classes = []
            
            for v in sorted_cands:
                placed = False
                for c, (max_w, cls) in enumerate(color_classes):
                    can_place = True
                    for u in cls:
                        if not (compat[v] >> u) & 1:
                            can_place = False
                            break
                    if can_place:
                        new_max_w = max(max_w, weights[v])
                        color_classes[c] = (new_max_w, cls + [v])
                        placed = True
                        break
                if not placed:
                    color_classes.append((weights[v], [v]))
            
            return sum(max_w for max_w, _ in color_classes)
        
        if init_clique:
            cand_mask = (1 << n) - 1
            for v in init_clique:
                cand_mask &= compat[v]
            for v in init_clique:
                cand_mask &= ~(1 << v)
        else:
            cand_mask = (1 << n) - 1
        
        stack = [(cand_mask, best_clique[:], best_weight)]
        
        weight_order = sorted(range(n), key=lambda i: weights[i], reverse=True)
        
        while stack:
            if (time.time() - start_time) > deadline:
                break
            
            cand_mask, clique, cl_weight = stack.pop()
            
            if cand_mask == 0:
                continue
            
            candidates = []
            for i in weight_order:
                if (cand_mask >> i) & 1:
                    candidates.append(i)
            
            if not candidates:
                continue
            
            ub = cl_weight + weighted_color_bound(candidates, weights)
            if ub <= best_weight:
                continue
            
            for v in candidates:
                if (time.time() - start_time) > deadline:
                    break
                
                new_clique = clique + [v]
                new_weight = cl_weight + weights[v]
                
                if new_weight > best_weight:
                    best_weight = new_weight
                    best_clique = new_clique[:]
                
                new_cand_mask = cand_mask & compat[v] & ~(1 << v)
                
                if new_cand_mask != 0:
                    stack.append((new_cand_mask, new_clique, new_weight))
        
        return best_clique, best_weight
    
    weights = [1.0 / all_nodes[i] for i in range(n_total)]
    
    init_clique_indices = [node_to_idx[v] for v in best_subset]
    
    remaining = deadline - (time.time() - start_time) - 5
    if remaining > 1:
        try:
            clique_result, clique_weight = max_weight_clique(
                weights, compat_mask, n_total, deadline - 3, init_clique_indices
            )
            
            clique_subset = [all_nodes[i] for i in clique_result]
            if clique_subset:
                if clique_weight > best_value:
                    best_subset = clique_subset
                    best_value = clique_weight
        except Exception:
            pass
    
    # Final polish
    if (time.time() - start_time) < deadline - 3:
        polished = local_search(best_subset, deadline - 2)
        val = sum(1.0 / v for v in polished)
        if val > best_value:
            best_subset = polished
            best_value = val
    
    if not best_subset:
        return [256]
    
    return best_subset
