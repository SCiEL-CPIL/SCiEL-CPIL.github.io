"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill

    def random_graph(n, p, seed):
        """Build adjacency as list of bitmasks (int)."""
        rng = Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def bron_kerbosch(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot, bitmask-based."""
        cliques = []
        
        def bk(R_mask, P_mask, X_mask):
            if P_mask == 0 and X_mask == 0:
                # Extract vertices from R_mask
                R = [v for v in range(n) if (R_mask >> v) & 1]
                if len(R) >= 2:
                    cliques.append(tuple(sorted(R)))
                return
            
            # Choose pivot u from P ∪ X to minimize branching
            union_mask = P_mask | X_mask
            if union_mask:
                # Pick pivot with max degree in P
                best_u = -1
                best_deg = -1
                # Iterate over bits in union_mask
                um = union_mask
                while um:
                    lb = um & (-um)
                    u = lb.bit_length() - 1
                    deg = bin(adj[u] & P_mask).count('1')
                    if deg > best_deg:
                        best_deg = deg
                        best_u = u
                    um ^= lb
                candidates = P_mask & ~(adj[best_u])
            else:
                candidates = P_mask
            
            # Iterate over candidates
            cm = candidates
            while cm:
                lb = cm & (-cm)
                v = lb.bit_length() - 1
                bk(R_mask | lb, P_mask & adj[v], X_mask & adj[v])
                P_mask &= ~lb
                X_mask |= lb
                cm ^= lb
        
        bk(0, (1 << n) - 1, 0)
        return cliques

    def greedy_hitting_set(cliques, n):
        """Greedy: pick vertex covering most uncovered cliques. Uses bitmask for cliques."""
        m = len(cliques)
        if m == 0:
            return set()
        
        # For each vertex, compute bitmask of cliques it belongs to
        vertex_clique_mask = [0] * n
        for i, c in enumerate(cliques):
            bit = 1 << i
            for v in c:
                vertex_clique_mask[v] |= bit
        
        remaining_mask = (1 << m) - 1  # all cliques uncovered
        selected = set()
        
        while remaining_mask:
            # Find vertex with max coverage of remaining cliques
            best_v = -1
            best_count = 0
            for v in range(n):
                if v in selected:
                    continue
                count = bin(vertex_clique_mask[v] & remaining_mask).count('1')
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1:
                break
            
            selected.add(best_v)
            remaining_mask &= ~vertex_clique_mask[best_v]
        
        return selected

    def local_search(selected, cliques, n, time_limit):
        """Improve via deletions and swaps using precomputed clique membership."""
        selected = set(selected)
        m = len(cliques)
        
        # Precompute: for each clique, which selected vertices are in it
        # And for each vertex, which cliques it covers
        vertex_clique_mask = [0] * n
        for i, c in enumerate(cliques):
            bit = 1 << i
            for v in c:
                vertex_clique_mask[v] |= bit
        
        def is_valid(sel):
            """Check if sel hits all cliques using bitmask."""
            covered = 0
            for v in sel:
                covered |= vertex_clique_mask[v]
            return covered == (1 << m) - 1
        
        def try_delete(sel):
            """Try removing one vertex at a time."""
            for v in list(sel):
                new_sel = sel - {v}
                if is_valid(new_sel):
                    return new_sel, True
            return sel, False
        
        def try_swap(sel):
            """Try swapping one vertex out for another."""
            for v_out in list(sel):
                new_sel_no = sel - {v_out}
                # Find uncovered cliques
                covered = 0
                for v in new_sel_no:
                    covered |= vertex_clique_mask[v]
                uncovered_mask = ((1 << m) - 1) & ~covered
                if uncovered_mask == 0:
                    return new_sel_no, True
                # Find a vertex that covers all uncovered cliques
                for v_in in range(n):
                    if v_in in sel:
                        continue
                    if (vertex_clique_mask[v_in] & uncovered_mask) == uncovered_mask:
                        return new_sel_no | {v_in}, True
            return sel, False

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            new_sel, ok = try_delete(selected)
            if ok:
                selected = new_sel
                improved = True
                continue
            new_sel, ok = try_swap(selected)
            if ok:
                selected = new_sel
                improved = True
                continue
        return selected

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5:
            results.append(list(range(n)))
            continue

        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj, n)

        if not cliques:
            results.append([])
            continue

        # Greedy initial solution
        greedy = greedy_hitting_set(cliques, n)

        # Local search with time budget
        per_instance_time = min(35, max(5, deadline - time.time() - 5))
        improved = local_search(greedy, cliques, n, time.time() + per_instance_time)

        results.append(sorted(improved))

    return results
# EVOLVE-BLOCK-END
