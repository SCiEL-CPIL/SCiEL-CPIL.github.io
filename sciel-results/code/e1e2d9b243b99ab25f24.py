import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj):
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection
            if P:
                pivot = max(P, key=lambda u: len(P & adj[u]))
                candidates = P - adj[pivot]
            else:
                candidates = P
                
            for v in list(candidates):
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.remove(v)
                P.remove(v)
                X.add(v)
                
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(adj):
        n = len(adj)
        cliques = bron_kerbosch(adj)
        if not cliques:
            return []
        
        # Map clique to index
        clique_id = {c: i for i, c in enumerate(cliques)}
        num_cliques = len(cliques)
        
        # Bitmasks for vertices
        # mask[v] has bit i set if v is in clique i
        masks = [0] * n
        for c in cliques:
            idx = clique_id[c]
            bit = 1 << idx
            for v in c:
                masks[v] |= bit
        
        all_mask = (1 << num_cliques) - 1
        
        # Precompute clique frequency for each vertex (how many cliques it appears in)
        vertex_clique_count = [0] * n
        for v in range(n):
            vertex_clique_count[v] = masks[v].bit_count()
        
        best_solution = None
        
        for restart in range(200):
            if time.time() > deadline - 3.0:
                break
            
            rng = random.Random(restart * 7919 + 12345)
            uncovered = all_mask
            selected = []
            available = set(range(n))
            
            while uncovered:
                best_v = -1
                max_score = -1
                for v in available:
                    m = masks[v] & uncovered
                    if not m:
                        continue
                    hits = m.bit_count()
                    # Weight by rarity: prefer vertices that cover rare cliques
                    # Compute a "rarity bonus" - sum of 1/freq for each covered clique
                    rarity_bonus = 0.0
                    mm = m
                    while mm:
                        lsb = mm & -mm
                        idx = lsb.bit_length() - 1
                        # Count how many vertices cover this clique
                        # We'll approximate: fewer vertices covering = more valuable
                        rarity_bonus += 1.0
                        mm ^= lsb
                    # Combine: primary is hits, secondary is some noise
                    noise = rng.random() * 0.05
                    score = hits * 10.0 + rarity_bonus + noise
                    if score > max_score:
                        max_score = score
                        best_v = v
                
                if best_v == -1:
                    break
                
                selected.append(best_v)
                available.discard(best_v)
                uncovered &= ~masks[best_v]
            
            # Minimization Phase: remove redundant vertices
            cover_count = [0] * num_cliques
            for v in selected:
                m = masks[v]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] += 1
                    m ^= lsb
            
            # Try to remove vertices in random order
            sel_list = list(selected)
            rng.shuffle(sel_list)
            changed = True
            while changed:
                changed = False
                for v in sel_list:
                    if v not in selected:
                        continue
                    critical = False
                    m = masks[v]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_count[idx] == 1:
                            critical = True
                            break
                        m ^= lsb
                    if not critical:
                        selected.remove(v)
                        m = masks[v]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            m ^= lsb
                        changed = True
            
            # Local search: try swapping one selected vertex with one unselected vertex
            if time.time() < deadline - 2.0:
                sel_set = set(selected)
                unselected = [v for v in range(n) if v not in sel_set]
                improved = True
                while improved and time.time() < deadline - 2.0:
                    improved = False
                    # Recompute cover counts
                    cover_count = [0] * num_cliques
                    for v in selected:
                        m = masks[v]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] += 1
                            m ^= lsb
                    
                    # Try removing each selected vertex and adding each unselected vertex
                    for i, v_out in enumerate(list(selected)):
                        if time.time() > deadline - 2.0:
                            break
                        # Temporarily remove v_out
                        m_out = masks[v_out]
                        # Check if removal leaves any clique uncovered
                        can_remove = True
                        for idx in range(num_cliques):
                            if (m_out >> idx) & 1 and cover_count[idx] == 1:
                                can_remove = False
                                break
                        if not can_remove:
                            continue
                        
                        # Find best replacement
                        best_in = -1
                        best_gain = 0
                        for v_in in unselected:
                            m_in = masks[v_in] & all_mask
                            # Gain: how many currently covered cliques does v_in also cover?
                            # We want to minimize loss: cliques that only v_out covers
                            loss_mask = m_out  # cliques v_out covers
                            # After removing v_out, cliques with cover_count==1 become uncovered
                            # v_in must cover those
                            critical_cliques = 0
                            for idx in range(num_cliques):
                                if (m_out >> idx) & 1 and cover_count[idx] == 1:
                                    if not ((m_in >> idx) & 1):
                                        critical_cliques += 1
                            if critical_cliques == 0:
                                # Valid swap, gain is 0 (same size), but we want to keep going
                                # Actually, we want to try to REDUCE size, so swaps don't help
                                # Unless we can then remove another vertex
                                pass
                        
                        # Actually, for size reduction, we need to find if removing v_out
                        # and adding v_in allows further removals. This is complex.
                        # Let's do a simpler approach: try all pairs (remove one, add one)
                        # and check if the new set is still valid and smaller after minimization.
                        
                        # Simpler: just try to find if there's a vertex v_in that covers
                        # all the critical cliques of v_out, then we can swap and potentially
                        # enable further removals.
                        # For now, skip complex swaps and just do removal-based minimization.
                        pass
                
                # Alternative local search: try adding a vertex then removing two
                # This is more complex. Let's do a simpler 1-swap:
                # For each pair (v_out in selected, v_in in unselected), check if
                # replacing v_out with v_in still covers all cliques. If yes and we can
                # then remove another vertex, we have an improvement.
                # This is O(|selected| * |unselected| * n) which is fine for n=40.
                
                # Let's do a proper 1-exchange local search:
                sel_list2 = list(selected)
                unselected_list = [v for v in range(n) if v not in sel_set]
                for v_out in sel_list2:
                    if time.time() > deadline - 2.0:
                        break
                    m_out = masks[v_out]
                    # Find critical cliques (only covered by v_out)
                    critical_cliques = []
                    for idx in range(num_cliques):
                        if (m_out >> idx) & 1 and cover_count[idx] == 1:
                            critical_cliques.append(idx)
                    
                    if not critical_cliques:
                        # v_out is not critical, already handled by minimization
                        continue
                    
                    # Find v_in that covers all critical cliques
                    for v_in in unselected_list:
                        if v_in in sel_set:
                            continue
                        m_in = masks[v_in]
                        valid = True
                        for idx in critical_cliques:
                            if not ((m_in >> idx) & 1):
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Try the swap
                        new_selected = [v for v in selected if v != v_out] + [v_in]
                        # Check if this new set covers all cliques
                        new_cover = 0
                        for v in new_selected:
                            new_cover |= masks[v]
                        if new_cover == all_mask:
                            # Now try to minimize the new set
                            new_cover_count = [0] * num_cliques
                            for v in new_selected:
                                m = masks[v]
                                while m:
                                    lsb = m & -m
                                    idx = lsb.bit_length() - 1
                                    new_cover_count[idx] += 1
                                    m ^= lsb
                            
                            # Try removing vertices from new_selected
                            new_sel = list(new_selected)
                            new_changed = True
                            while new_changed:
                                new_changed = False
                                for v in list(new_sel):
                                    if v not in new_sel:
                                        continue
                                    crit = False
                                    m = masks[v]
                                    while m:
                                        lsb = m & -m
                                        idx = lsb.bit_length() - 1
                                        if new_cover_count[idx] == 1:
                                            crit = True
                                            break
                                        m ^= lsb
                                    if not crit:
                                        new_sel.remove(v)
                                        m = masks[v]
                                        while m:
                                            lsb = m & -m
                                            idx = lsb.bit_length() - 1
                                            new_cover_count[idx] -= 1
                                            m ^= lsb
                                        new_changed = True
                            
                            if len(new_sel) < len(selected):
                                selected = new_sel
                                sel_set = set(selected)
                                unselected_list = [v for v in range(n) if v not in sel_set]
                                improved = True
                                break
                    if improved:
                        break
            
            if best_solution is None or len(selected) < len(best_solution):
                best_solution = list(selected)
            
            if len(best_solution) <= 3:
                break

        return sorted(best_solution) if best_solution else []

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        adj = random_graph(n, p, seed)
        sol = solve_instance(adj)
        out.append(sol)
        
    # Safety: if we ran out of time, return best available
    return out
