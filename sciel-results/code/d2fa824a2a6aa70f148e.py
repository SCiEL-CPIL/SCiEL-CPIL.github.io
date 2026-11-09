"""Structural change: ILP-based hitting set solver with bitmask Bron-Kerbosch."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s kill

    def generate_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def enumerate_maximal_cliques_bitmask(n, edges):
        """Fast Bron-Kerbosch with pivot using bitmasks."""
        # Build adjacency as bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Non-isolated vertices
        non_isolated_mask = 0
        for u in range(n):
            if adj[u]:
                non_isolated_mask |= (1 << u)
        
        if non_isolated_mask == 0:
            return []
        
        cliques = []
        
        def bk(R, P, X):
            if time.time() > DEADLINE:
                return
            if P == 0 and X == 0:
                if R != 0:
                    # R has at least one bit; check if size >= 2
                    if R & (R - 1):  # more than one bit
                        cliques.append(R)
                return
            
            # Pivot: choose u from P|X maximizing popcount(P & adj[u])
            candidates = P | X
            if candidates == 0:
                return
            
            best_u = -1
            best_count = -1
            # Iterate through bits in candidates
            c = candidates
            while c:
                lb = c & (-c)
                u = lb.bit_length() - 1
                cnt = (P & adj[u]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
                c ^= lb
            
            if best_u == -1:
                return
            
            # Iterate over P \ N(best_u)
            to_process = P & ~adj[best_u]
            while to_process:
                lb = to_process & (-to_process)
                v = lb.bit_length() - 1
                bk(R | lb, P & adj[v], X & adj[v])
                P ^= lb
                X |= lb
                to_process ^= lb
        
        bk(0, non_isolated_mask, 0)
        return cliques

    def greedy_hitting_set_fast(clique_masks, n):
        """Greedy hitting set using bitmasks for speed."""
        m = len(clique_masks)
        if m == 0:
            return 0
        
        # uncovered: bitmask of clique indices (use int for small m, else list)
        # For m up to ~1000, use Python int as bitmask
        uncovered = (1 << m) - 1
        result = 0
        
        # vertex_cliques[v] = bitmask of clique indices containing v
        vertex_cliques = [0] * n
        for i, cm in enumerate(clique_masks):
            # For each vertex in clique i, set bit i in vertex_cliques[v]
            c = cm
            while c:
                lb = c & (-c)
                v = lb.bit_length() - 1
                vertex_cliques[v] |= (1 << i)
                c ^= lb
        
        while uncovered:
            best_v = -1
            best_count = 0
            for v in range(n):
                if result & (1 << v):
                    continue
                cnt = (vertex_cliques[v] & uncovered).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
            
            if best_v == -1 or best_count == 0:
                break
            
            result |= (1 << best_v)
            uncovered &= ~vertex_cliques[best_v]
        
        return result

    def ilp_hitting_set(clique_masks, n, time_limit=20.0):
        """Solve hitting set as ILP using scipy.optimize.milp."""
        m = len(clique_masks)
        if m == 0:
            return 0
        
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            from scipy.sparse import csr_matrix
            import scipy.sparse as sp
            
            # Objective: minimize sum x_v
            c = np.ones(n)
            
            # Constraint: for each clique i, sum_{v in clique_i} x_v >= 1
            # A x >= 1, where A[i,v] = 1 if v in clique_i
            rows = []
            cols = []
            data = []
            for i, cm in enumerate(clique_masks):
                c_mask = cm
                while c_mask:
                    lb = c_mask & (-c_mask)
                    v = lb.bit_length() - 1
                    rows.append(i)
                    cols.append(v)
                    data.append(1.0)
                    c_mask ^= lb
            
            A = sp.csr_matrix((data, (rows, cols)), shape=(m, n))
            
            constraint = LinearConstraint(A, 1, np.inf)
            bounds = Bounds(0, 1)
            
            # Set integrality: all variables integer
            integrality = np.ones(n)
            
            res = milp(c, constraints=[constraint], integrality=integrality, bounds=bounds,
                       options={'time_limit': time_limit, 'mip_rel_gap': 0.0})
            
            if res.status == 0 or res.status == 1:  # optimal or near-optimal
                x = res.x
                selected = 0
                for v in range(n):
                    if x[v] > 0.5:
                        selected |= (1 << v)
                return selected
        except Exception:
            pass
        return -1  # signal failure

    def local_search_bitmask(hitting_mask, clique_masks, n):
        """Remove redundant vertices from hitting set."""
        hs = hitting_mask
        if hs == 0:
            return hs
        
        # For each vertex v in hs, check if removing it still covers all cliques
        # A vertex v is redundant if for every clique containing v, there's another vertex in hs
        m = len(clique_masks)
        
        changed = True
        while changed:
            changed = False
            h = hs
            while h:
                lb = h & (-h)
                v = lb.bit_length() - 1
                # Check all cliques containing v
                can_remove = True
                for cm in clique_masks:
                    if cm & lb:  # v is in this clique
                        # Check if any other vertex in cm is in hs (excluding v)
                        others = cm & ~(lb)
                        if (hs & others) == 0:
                            can_remove = False
                            break
                if can_remove:
                    hs ^= lb
                    changed = True
                    break  # restart
                h ^= lb
        return hs

    def find_min_hitting_set(clique_masks, n):
        """Find minimum hitting set using ILP + greedy fallback + local search."""
        if not clique_masks:
            return 0
        
        # 1. Greedy initialization
        sol_greedy = greedy_hitting_set_fast(clique_masks, n)
        
        # 2. Try ILP
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 5.0  # 5s buffer
        ilp_time = max(1.0, min(remaining, 30.0))
        
        sol_ilp = ilp_hitting_set(clique_masks, n, time_limit=ilp_time)
        
        if sol_ilp >= 0:
            # ILP succeeded
            current = sol_ilp
            # Local search to potentially reduce further
            current = local_search_bitmask(current, clique_masks, n)
            # Compare with greedy
            if current.bit_count() <= sol_greedy.bit_count():
                return current
            else:
                return sol_greedy
        else:
            # ILP failed, use greedy
            current = sol_greedy
            current = local_search_bitmask(current, clique_masks, n)
            return current

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            # Fallback: return all non-isolated vertices
            edges = generate_edges(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
        
        edges = generate_edges(n, p, seed)
        cliques = enumerate_maximal_cliques_bitmask(n, edges)
        
        if not cliques:
            results.append([])
            continue
        
        sol_mask = find_min_hitting_set(cliques, n)
        # Convert bitmask to sorted list
        sol_list = []
        sm = sol_mask
        while sm:
            lb = sm & (-sm)
            v = lb.bit_length() - 1
            sol_list.append(v)
            sm ^= lb
        results.append(sorted(sol_list))

    return results
# EVOLVE-BLOCK-END
