import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    n_max = 180
    FULL = (1 << n_max) - 1  # all 180 bits set

    # Precompute forbidden triples: (a,b,c) with a<b<c and 1/a = 1/b + 1/c
    # Equivalent: (b-a)(c-a) = a^2
    forbidden_triples = []
    for a in range(1, n_max + 1):
        a_sq = a * a
        for u in range(1, int(a_sq**0.5) + 1):
            if a_sq % u == 0:
                v = a_sq // u
                if u >= v:
                    continue
                b = a + u
                c = a + v
                if b <= n_max and c <= n_max:
                    forbidden_triples.append((a, b, c))

    # For each vertex i (1-indexed), store:
    #   pair_mask[i]: bitmask where bit j is set iff there exists k such that {i,j,k} is a forbidden triple
    #   Actually we need: for vertex i, the set of pairs (j,k) such that {i,j,k} is forbidden.
    #   For branch-and-bound, when we consider adding vertex i, we must remove all j,k that form a triple with i.
    #   So: conflict_mask[i] = bitmask of all vertices that appear in any forbidden triple with i.
    #   And: pair_conflicts[i] = list of (j, k) pairs.
    
    conflict_mask = [0] * (n_max + 1)
    pair_conflicts = [[] for _ in range(n_max + 1)]
    
    for a, b, c in forbidden_triples:
        # Convert to 0-indexed for bitmask
        ai, bi, ci = a - 1, b - 1, c - 1
        conflict_mask[a] |= (1 << bi) | (1 << ci)
        conflict_mask[b] |= (1 << ai) | (1 << ci)
        conflict_mask[c] |= (1 << ai) | (1 << bi)
        pair_conflicts[a].append((bi, ci))
        pair_conflicts[b].append((ai, ci))
        pair_conflicts[c].append((ai, bi))

    # Greedy: build a good initial solution
    def greedy_order(order):
        S = 0  # bitmask
        S_set = set()  # for fast membership
        for num in order:
            idx = num - 1
            if S & (1 << idx):
                continue
            # Check if adding num creates a forbidden triple
            ok = True
            for j, k in pair_conflicts[num]:
                if (S >> j) & 1 and (S >> k) & 1:
                    ok = False
                    break
            if ok:
                S |= (1 << idx)
                S_set.add(idx)
        return S, S_set

    # Multiple greedy orders
    triple_count = [0] * (n_max + 1)
    for a, b, c in forbidden_triples:
        triple_count[a] += 1
        triple_count[b] += 1
        triple_count[c] += 1

    order_deg = list(range(1, n_max + 1))
    order_deg.sort(key=lambda x: triple_count[x])
    order_asc = list(range(1, n_max + 1))
    order_desc = list(range(n_max, 0, -1))

    best_S = 0
    best_size = 0

    for order in [order_deg, order_asc, order_desc]:
        S, S_set = greedy_order(order)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_S = S

    # Random greedy orders
    for _ in range(100):
        if time.time() - start_time > deadline:
            break
        order_rand = list(range(1, n_max + 1))
        random.shuffle(order_rand)
        S, S_set = greedy_order(order_rand)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_S = S

    # Local search: try to improve by removing one and adding two, or removing one and adding one
    def try_improve(S, S_set):
        """Try local search improvements. Returns improved (S, S_set) or None."""
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > deadline:
                return S, S_set
            
            # Try adding a vertex not in S
            for num in range(1, n_max + 1):
                idx = num - 1
                if (S >> idx) & 1:
                    continue
                # Check if we can add num
                ok = True
                for j, k in pair_conflicts[num]:
                    if (S >> j) & 1 and (S >> k) & 1:
                        ok = False
                        break
                if ok:
                    S |= (1 << idx)
                    S_set.add(idx)
                    improved = True
                    break  # restart after improvement
        
        # Try swap: remove one vertex, add another
        for remove_num in range(1, n_max + 1):
            ridx = remove_num - 1
            if not ((S >> ridx) & 1):
                continue
            S2 = S & ~(1 << ridx)
            S_set2 = S_set - {ridx}
            
            for add_num in range(1, n_max + 1):
                aidx = add_num - 1
                if (S2 >> aidx) & 1:
                    continue
                ok = True
                for j, k in pair_conflicts[add_num]:
                    if (S2 >> j) & 1 and (S2 >> k) & 1:
                        ok = False
                        break
                if ok:
                    S2 |= (1 << aidx)
                    S_set2.add(aidx)
                    if S2.bit_count() > S.bit_count():
                        return S2, S_set2
                    # Same size but different - accept if it opens up more possibilities
                    # For now, only accept strict improvements
        return S, S_set

    S_cur, S_set_cur = try_improve(best_S, set(i for i in range(n_max) if (best_S >> i) & 1))
    best_size = S_cur.bit_count()
    best_S = S_cur

    # Branch and bound for maximum independent set
    # State: (available_mask, current_set_mask)
    # available_mask: vertices that can still be considered
    # We use a recursive approach with pruning
    
    def branch_and_bound(available, current_mask, current_size, best_ref):
        """
        available: bitmask of vertices that can still be added
        current_mask: bitmask of vertices already in the solution
        current_size: number of vertices in current solution
        best_ref: list containing [best_size, best_mask] for mutation
        """
        if time.time() - start_time > deadline:
            return
        
        if not available:
            if current_size > best_ref[0]:
                best_ref[0] = current_size
                best_ref[1] = current_mask
            return
        
        # Upper bound: current_size + popcount(available)
        upper = current_size + available.bit_count()
        if upper <= best_ref[0]:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick the vertex with highest degree in the available subgraph
        # For simplicity, pick the first available vertex or use a degree heuristic
        # Let's pick the vertex that conflicts with the most other available vertices
        # This is expensive, so let's just pick a vertex with high global conflict count
        
        # Find an available vertex with high conflict count among available vertices
        best_v = -1
        best_conflicts = -1
        # Sample some vertices to avoid full scan
        avail_list = []
        temp = available
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            avail_list.append(idx)
            temp ^= lsb
        
        if not avail_list:
            return
            
        # Pick vertex with most conflicts within available set
        # For performance, limit the scan
        for idx in avail_list[:50]:  # limit to 50 candidates
            cm = conflict_mask[idx + 1] & available
            cnt = cm.bit_count()
            if cnt > best_conflicts:
                best_conflicts = cnt
                best_v = idx
        
        if best_v == -1:
            best_v = avail_list[0]
        
        v_bit = 1 << best_v
        v_num = best_v + 1
        
        # Branch 1: Include best_v
        # Remove all vertices that conflict with best_v from available
        new_available = available & ~(conflict_mask[v_num] | v_bit)
        branch_and_bound(new_available, current_mask | v_bit, current_size + 1, best_ref)
        
        # Branch 2: Exclude best_v
        branch_and_bound(available & ~v_bit, current_mask, current_size, best_ref)

    # Initialize best_ref
    best_ref = [best_size, best_S]
    
    # Start B&B from the full set, but we can also start from a reduced available set
    # To make it effective, we should start with a good lower bound
    # The greedy already gave us best_size
    
    # For B&B to be effective, we need a good upper bound initially
    # Let's run B&B with a time limit
    # Start with all vertices available
    # But we need to be smart: the B&B should explore including high-value vertices first
    
    # Actually, for a 180-vertex hypergraph, pure B&B might be too slow.
    # Let's use a different approach: iterative improvement with random restarts
    # and a more sophisticated local search.
    
    # Alternative: use a simulated annealing or tabu search approach
    # Or: use the B&B but with aggressive pruning and a good initial solution
    
    # Let's try a different strategy: 
    # 1. Get a good initial solution via greedy (done)
    # 2. Use local search with more aggressive moves (remove up to 2, add up to 3)
    # 3. Random restarts with perturbation
    
    def aggressive_local_search(S, S_set, max_iter=500):
        """More aggressive local search: try removing 1-2 vertices and adding 1-3 vertices."""
        best_S = S
        best_set = S_set.copy()
        best_size = S.bit_count()
        
        for it in range(max_iter):
            if time.time() - start_time > deadline:
                break
            
            # Try: remove 1 vertex, add 1 vertex (swap)
            improved = False
            for r in range(n_max):
                if not ((S >> r) & 1):
                    continue
                S2 = S & ~(1 << r)
                for a in range(n_max):
                    if (S2 >> a) & 1:
                        continue
                    # Check validity of adding a to S2
                    ok = True
                    for j, k in pair_conflicts[a + 1]:
                        if (S2 >> j) & 1 and (S2 >> k) & 1:
                            ok = False
                            break
                    if ok:
                        S3 = S2 | (1 << a)
                        if S3.bit_count() > best_size:
                            best_size = S3.bit_count()
                            best_S = S3
                            best_set = set(i for i in range(n_max) if (S3 >> i) & 1)
                            S = S3
                            S_set = best_set
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try: remove 2 vertices, add 2 vertices
            # This is O(n^4) which is too slow. Limit it.
            if it % 10 == 0:  # only do this occasionally
                for r1 in range(0, n_max, 3):  # sample
                    if not ((S >> r1) & 1):
                        continue
                    for r2 in range(r1 + 1, n_max, 3):
                        if not ((S >> r2) & 1):
                            continue
                        S2 = S & ~(1 << r1) & ~(1 << r2)
                        # Try to add 2 vertices
                        added = 0
                        S3 = S2
                        for a in range(n_max):
                            if added >= 2:
                                break
                            if (S3 >> a) & 1:
                                continue
                            ok = True
                            for j, k in pair_conflicts[a + 1]:
                                if (S3 >> j) & 1 and (S3 >> k) & 1:
                                    ok = False
                                    break
                            if ok:
                                S3 |= (1 << a)
                                added += 1
                        if S3.bit_count() > best_size:
                            best_size = S3.bit_count()
                            best_S = S3
                            best_set = set(i for i in range(n_max) if (S3 >> i) & 1)
                            S = S3
                            S_set = best_set
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved and it > 100:
                # No improvement for a while, try perturbation
                # Remove 2-3 random vertices
                S_list = [i for i in range(n_max) if (S >> i) & 1]
                if len(S_list) > best_size + 2:
                    remove_set = random.sample(S_list, min(3, len(S_list)))
                    for r in remove_set:
                        S &= ~(1 << r)
                        S_set.discard(r)
        
        return best_S, best_set, best_size

    S_cur, S_set_cur, best_size = aggressive_local_search(best_S, set(i for i in range(n_max) if (best_S >> i) & 1))
    best_S = S_cur

    # More random restarts with perturbation
    for restart in range(50):
        if time.time() - start_time > deadline:
            break
        
        # Perturb the current best solution
        S_pert = best_S
        S_set_pert = set(i for i in range(n_max) if (best_S >> i) & 1)
        
        # Remove 5-10 random vertices
        S_list = list(S_set_pert)
        num_remove = random.randint(5, 10)
        if len(S_list) > num_remove:
            remove_set = random.sample(S_list, num_remove)
            for r in remove_set:
                S_pert &= ~(1 << r)
                S_set_pert.discard(r)
        
        # Run local search from perturbed solution
        S_new, S_set_new, size_new = aggressive_local_search(S_pert, S_set_pert, max_iter=200)
        
        if size_new > best_size:
            best_size = size_new
            best_S = S_new

    # Final: try to add any remaining vertices
    S_final = best_S
    S_set_final = set(i for i in range(n_max) if (best_S >> i) & 1)
    for num in range(1, n_max + 1):
        idx = num - 1
        if (S_final >> idx) & 1:
            continue
        ok = True
        for j, k in pair_conflicts[num]:
            if (S_final >> j) & 1 and (S_final >> k) & 1:
                ok = False
                break
        if ok:
            S_final |= (1 << idx)
            S_set_final.add(idx)
    
    result = sorted(i + 1 for i in range(n_max) if (S_final >> i) & 1)
    
    # Verify the result
    if not result:
        result = [1]
    
    return result
