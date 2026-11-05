import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def enumerate_maximal_cliques(adj, n):
        """
        Enumerates all maximal cliques using Bron-Kerbosch with pivot.
        Returns a list of bitmasks, each representing a maximal clique.
        Only includes cliques with size >= 2.
        """
        cliques = []
        
        def bk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                # We only care about cliques with at least 2 vertices
                if R.bit_count() >= 2:
                    cliques.append(R)
                return
            
            # Pivot selection: choose u in P ∪ X that maximizes |P ∩ N(u)|
            PX = P | X
            if PX:
                max_common = -1
                pivot = -1
                temp = PX
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    common = (P & adj[u]).bit_count()
                    if common > max_common:
                        max_common = common
                        pivot = u
                    temp ^= lsb
            else:
                pivot = -1

            if pivot != -1:
                candidates = P & ~(adj[pivot])
            else:
                candidates = P

            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                
                bk(R | lsb, P & adj[v], X & adj[v])
                
                P ^= lsb
                X |= lsb

        bk(0, (1 << n) - 1, 0)
        return cliques

    def solve_instance(adj, n):
        """
        Finds a minimum clique transversal using Multi-Start Greedy + Local Search.
        """
        cliques = enumerate_maximal_cliques(adj, n)
        
        if not cliques:
            return []
        
        num_cliques = len(cliques)
        
        # Map vertex to list of clique indices it belongs to
        # For faster lookup, we can also store a bitmask of cliques for each vertex?
        # num_cliques can be large, but for n=40, it's usually manageable.
        # Let's stick to lists for simplicity, but optimize the coverage check.
        
        vertex_to_cliques = [[] for _ in range(n)]
        for idx, c in enumerate(cliques):
            m = c
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                vertex_to_cliques[v].append(idx)
                m ^= lsb

        # Precompute a bitmask of cliques for each vertex if num_cliques < 64?
        # If num_cliques > 64, we can't use a single int.
        # Let's check num_cliques.
        # For n=40, p=0.2, number of maximal cliques is usually small (< 100).
        # If num_cliques <= 64, we can use bitmasks for coverage.
        use_bitmask_coverage = num_cliques <= 64
        
        if use_bitmask_coverage:
            vertex_clique_mask = [0] * n
            for idx, c in enumerate(cliques):
                mask = 1 << idx
                m = c
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    vertex_clique_mask[v] |= mask
                    m ^= lsb
            full_mask = (1 << num_cliques) - 1
        else:
            # Fallback to sets if too many cliques (unlikely for these params)
            vertex_clique_set = [set(vertex_to_cliques[v]) for v in range(n)]
            full_set = set(range(num_cliques))

        def is_valid_cover(selected):
            if use_bitmask_coverage:
                cover = 0
                for v in selected:
                    cover |= vertex_clique_mask[v]
                return cover == full_mask
            else:
                cover = set()
                for v in selected:
                    cover.update(vertex_clique_set[v])
                return cover == full_set

        def greedy_set_cover(randomize_ties=False):
            uncovered_mask = full_mask if use_bitmask_coverage else full_set
            selected = []
            
            # To speed up, we can maintain a count of how many times each vertex covers an uncovered clique.
            # But recalculating is O(n * num_cliques) which is small.
            
            while uncovered_mask:
                if time.time() > deadline:
                    break
                
                best_v = -1
                max_covered = 0
                candidates = []
                
                for v in range(n):
                    if use_bitmask_coverage:
                        covered_here = (uncovered_mask & vertex_clique_mask[v]).bit_count()
                    else:
                        covered_here = len(uncovered_mask.intersection(vertex_clique_set[v]))
                    
                    if covered_here > max_covered:
                        max_covered = covered_here
                        best_v = v
                        candidates = [v]
                    elif covered_here == max_covered and covered_here > 0:
                        candidates.append(v)
                
                if best_v == -1:
                    break
                
                if randomize_ties and len(candidates) > 1:
                    best_v = random.choice(candidates)
                
                selected.append(best_v)
                
                if use_bitmask_coverage:
                    uncovered_mask &= ~vertex_clique_mask[best_v]
                else:
                    uncovered_mask -= vertex_clique_set[best_v]
            
            return selected

        def local_search(selected):
            # Phase 1: Removal
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for i in range(len(selected)):
                    v = selected[i]
                    temp_selected = selected[:i] + selected[i+1:]
                    if is_valid_cover(temp_selected):
                        selected = temp_selected
                        changed = True
                        break # Restart loop as indices changed
            
            # Phase 2: 1-Swap to enable removal?
            # Actually, a swap that keeps size same is only useful if it allows a removal.
            # Let's try: for each v in selected, for each w not in selected:
            #   if valid(selected - v + w):
            #      new_sel = selected - v + w
            #      try removal on new_sel
            #      if size(new_sel_reduced) < size(selected):
            #          selected = new_sel_reduced
            # This is O(n^2 * cost_check). With n=40, n^2=1600. cost_check is small.
            # We do this only if we have time.
            
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(len(selected)):
                    if time.time() > deadline - 5:
                        break
                    v_out = selected[i]
                    base_without_v = selected[:i] + selected[i+1:]
                    
                    for w_in in range(n):
                        if w_in in base_without_v:
                            continue
                        
                        # Check if base_without_v + w_in is a valid cover
                        # Optimization: 
                        # Uncovered by base_without_v: U1
                        # Covered by w_in: C_w
                        # Valid if U1 is subset of C_w
                        if use_bitmask_coverage:
                            cover_base = 0
                            for v in base_without_v:
                                cover_base |= vertex_clique_mask[v]
                            uncovered_by_base = full_mask & ~cover_base
                            if uncovered_by_base & ~vertex_clique_mask[w_in] == 0:
                                # Valid swap
                                new_selected = base_without_v + [w_in]
                                # Try to remove from new_selected
                                # Simple removal pass
                                changed = True
                                while changed:
                                    changed = False
                                    for j in range(len(new_selected)):
                                        v_rem = new_selected[j]
                                        temp = new_selected[:j] + new_selected[j+1:]
                                        if is_valid_cover(temp):
                                            new_selected = temp
                                            changed = True
                                            break
                                
                                if len(new_selected) < len(selected):
                                    selected = new_selected
                                    improved = True
                                    break
                        else:
                            cover_base = set()
                            for v in base_without_v:
                                cover_base.update(vertex_clique_set[v])
                            uncovered_by_base = full_set - cover_base
                            if uncovered_by_base.issubset(vertex_clique_set[w_in]):
                                new_selected = base_without_v + [w_in]
                                changed = True
                                while changed:
                                    changed = False
                                    for j in range(len(new_selected)):
                                        v_rem = new_selected[j]
                                        temp = new_selected[:j] + new_selected[j+1:]
                                        if is_valid_cover(temp):
                                            new_selected = temp
                                            changed = True
                                            break
                                
                                if len(new_selected) < len(selected):
                                    selected = new_selected
                                    improved = True
                                    break
                    if improved:
                        break
            
            return selected

        best_solution = None
        best_size = float('inf')
        
        # Multi-start
        num_restarts = 10
        # If n is small, we can afford more restarts.
        if n <= 30:
            num_restarts = 20
        elif n <= 35:
            num_restarts = 15
            
        for i in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # First restart is deterministic, others randomized
            sol = greedy_set_cover(randomize_ties=(i > 0))
            sol = local_search(sol)
            
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
                
            # Early stop if we found a size 1? Unlikely for connected graphs with edges.
            # Or if we hit a lower bound? 
            # A simple lower bound: max clique size / max degree? No.
            # For now, just run restarts.

        if best_solution is None:
            # Fallback
            best_solution = greedy_set_cover(False)
            best_solution = local_search(best_solution)
            
        return sorted(best_solution)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 15:
            # Fallback: return empty or heuristic if running out of time
            # But these should be fast.
            pass
            
        adj = random_graph(n, p, seed)
        vc = solve_instance(adj, n)
        results.append(vc)
        
    return results

