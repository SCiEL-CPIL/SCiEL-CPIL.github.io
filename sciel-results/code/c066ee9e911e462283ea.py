import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute conflict graph: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Identify maximal cliques: weight-4 vectors sharing exactly 3 coordinates
    # Two vectors conflict if their dot product > 2.
    # For weight-4 vectors in {-1,0,1}^11, dot = sum of products of shared nonzero coords.
    # If they share k coords with same sign and (4-k) coords with opposite sign among shared,
    # dot = (#same) - (#opposite) among the intersection of support.
    # Actually: if supports overlap in m positions, and in t of those they have same sign,
    # dot = t - (m - t) = 2t - m.
    # Conflict if 2t - m > 2, i.e., 2t > m + 2.
    # For m=4 (same support): t can be 0,1,2,3,4. Conflict if 2t > 6, so t>=4. Only identical.
    # Wait, that's not right. Let me reconsider.
    # Two vectors with same support of size 4: dot = sum over 4 positions of s1[i]*s2[i].
    # Each term is +1 or -1. dot = (#agreements) - (#disagreements) = 2*(#agreements) - 4.
    # Conflict if 2*(#agreements) - 4 > 2, i.e., #agreements > 3, i.e., #agreements = 4.
    # So same support: only conflict if identical (all signs same).
    # For m=3 (share 3 coords): dot = 2*(#agreements among 3) - 3.
    # Conflict if 2*(#agreements) - 3 > 2, i.e., 2*(#agreements) > 5, i.e., #agreements >= 3.
    # So if they share 3 coords and all 3 have same sign, they conflict.
    # For m=2: dot = 2*(#agreements) - 2. Conflict if 2*(#agreements) > 4, #agreements > 2. Impossible for m=2.
    # For m=1: dot = 2*(#agreements) - 1. Conflict if 2*(#agreements) > 3, #agreements > 1.5. Impossible.
    # For m=0: dot = 0. No conflict.
    #
    # So conflicts occur when:
    # 1. Same support (m=4) and all signs agree (identical vectors) - but we have unique vectors
    # 2. Share exactly 3 coordinates, and all 3 shared coords have the same sign
    #
    # This means maximal cliques are formed by fixing 3 coordinates and their signs,
    # then the 4th coordinate can vary in position and sign.
    # Wait, no. If we fix 3 positions and their signs, the 4th position can be any of the remaining 8 positions,
    # and its sign can be +1 or -1. But two vectors with the same 3 fixed coords but different 4th coords
    # only share those 3 coords, so they conflict (since all 3 shared have same sign).
    # So a clique is: fix 3 positions and their signs. All weight-4 vectors that include those 3 positions
    # with those signs form a clique.
    # The 4th position can be any of the remaining D-3 = 8 positions, with sign +1 or -1.
    # So each such clique has 8*2 = 16 vectors.
    #
    # Number of such cliques: C(11,3) * 2^3 = 165 * 8 = 1320 cliques.
    
    # Precompute for each vector which cliques it belongs to
    # A vector is identified by (support, signs). 
    # For a vector with support S (size 4) and signs s, it belongs to cliques defined by
    # each 3-subset of S with the corresponding signs.
    
    # Map from vector index to list of clique IDs
    # First, enumerate all cliques
    clique_data = []  # list of (3_positions, 3_signs, [vector_indices])
    pos3_sign3_to_clique = {}
    
    for pos3 in itertools.combinations(range(D), 3):
        for sign3 in itertools.product([1, -1], repeat=3):
            key = (pos3, sign3)
            pos3_sign3_to_clique[key] = len(clique_data)
            clique_data.append((pos3, sign3, []))
    
    # For each vector, find which cliques it belongs to
    vec_to_cliques = [[] for _ in range(N4)]
    
    for i, v in enumerate(v4_candidates):
        # Find support positions
        support = [j for j in range(D) if v[j] != 0]
        # For each 3-subset of support, check if the signs match a clique
        for pos3 in itertools.combinations(support, 3):
            sign3 = tuple(v[p] for p in pos3)
            key = (pos3, sign3)
            if key in pos3_sign3_to_clique:
                cid = pos3_sign3_to_clique[key]
                clique_data[cid][2].append(i)
                vec_to_cliques[i].append(cid)
    
    # Compute clique coverage: for each vertex, number of cliques it's in
    # And for each clique, the set of vertices in it
    clique_sizes = np.array([len(c[2]) for c in clique_data], dtype=np.int64)
    
    # Now implement greedy independent set with clique coverage heuristic
    # Strategy: at each step, pick the vertex that "covers" the most cliques
    # (i.e., is in the most cliques), because removing it prunes the most future conflicts.
    # Actually, for independent set, we want to SELECT vertices. The heuristic should be:
    # select the vertex that conflicts with the fewest remaining candidates,
    # OR select the vertex that is in the fewest cliques (less "constrained").
    # 
    # Better: use the dual view. In the conflict graph, we want an independent set.
    # A vertex in many cliques has many conflicts. So we should prefer vertices
    # with fewer cliques (less constrained).
    # 
    # Alternative: "clique coverage" as in the directive: remove the vertex with
    # highest clique coverage to prune large conflict structures. This is for the
    # VERTEX COVER / MAX CLIQUE dual. For independent set, we want the opposite:
    # select vertices with LOW clique coverage first.
    
    def greedy_independent_set(conflicts, N, clique_coverage, rng, deadline):
        """Greedy IS: select vertex with lowest clique coverage among alive vertices."""
        alive = np.ones(N, dtype=bool)
        selected = np.zeros(N, dtype=bool)
        coverage = clique_coverage.copy()
        
        while np.any(alive):
            if time.time() > deadline - 15:
                break
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            # Pick vertex with minimum coverage among alive
            min_cov = np.min(coverage[alive_indices])
            cands = alive_indices[coverage[alive_indices] == min_cov]
            chosen = cands[rng.integers(len(cands))]
            
            selected[chosen] = True
            to_remove = np.where(conflicts[chosen] | (np.arange(N) == chosen))[0]
            alive[to_remove] = False
            # Update coverage: for remaining alive vertices, subtract cliques that are now "satisfied"
            # A clique is satisfied if we've selected one of its vertices.
            # Actually, coverage should reflect remaining conflicts. Let's just use degree.
            # Simpler: use degree as proxy
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                # Recompute degrees for remaining
                pass
        
        return selected
    
    # Simpler approach: use degree-based greedy with multiple restarts,
    # but also try clique-coverage-based ordering
    
    best_selected = None
    best_count = 0
    
    # Strategy 1: Greedy minimum degree with incremental updates
    def greedy_min_degree(conflicts, N, rng, deadline):
        alive = np.ones(N, dtype=bool)
        selected = np.zeros(N, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int64)
        
        while np.any(alive):
            if time.time() > deadline - 15:
                break
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            alive_degrees = degrees[alive_indices]
            min_deg = np.min(alive_degrees)
            cands = alive_indices[alive_degrees == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            selected[chosen] = True
            to_remove = np.where(conflicts[chosen] | (np.arange(N) == chosen))[0]
            alive[to_remove] = False
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                degrees[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
        
        return selected
    
    # Strategy 2: Greedy by clique coverage (select low-coverage vertices first)
    def greedy_clique_coverage(conflicts, N, vec_to_cliques, rng, deadline):
        alive = np.ones(N, dtype=bool)
        selected = np.zeros(N, dtype=bool)
        coverage = np.array([len(c) for c in vec_to_cliques], dtype=np.int64)
        
        while np.any(alive):
            if time.time() > deadline - 15:
                break
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            alive_cov = coverage[alive_indices]
            min_cov = np.min(alive_cov)
            cands = alive_indices[alive_cov == min_cov]
            chosen = cands[rng.integers(len(cands))]
            
            selected[chosen] = True
            to_remove = np.where(conflicts[chosen] | (np.arange(N) == chosen))[0]
            alive[to_remove] = False
            # Update coverage for remaining: a vertex's effective coverage decreases
            # as its clique-mates are removed. Approximate by degree.
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                # Use degree as updated coverage proxy
                coverage[remaining] = conflicts[remaining][:, alive].sum(axis=1)
        
        return selected
    
    # Strategy 3: Spectral initialization + local search
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    
    degree = conflicts.sum(axis=1)
    L = np.diag(degree) - conflicts.astype(np.float64)
    L_sparse = csr_matrix(L)
    
    try:
        eigvals, eigvecs = eigsh(L_sparse, k=3, sigma=0, which='LM')
        idx = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx]
        fiedler = eigvecs[:, 1]
    except Exception:
        rng = np.random.default_rng(42)
        fiedler = rng.standard_normal(N4)
    
    # Strategy 4: Local search refinement
    def local_search(selected, conflicts, N, deadline):
        selected = selected.copy()
        improved = True
        max_iters = 2000
        for _ in range(max_iters):
            if time.time() > deadline - 15:
                break
            improved = False
            non_selected = np.where(~selected)[0]
            if len(non_selected) == 0:
                break
            
            # Try to add any non-selected vertex with no conflicts with selected
            for v in non_selected:
                if time.time() > deadline - 15:
                    break
                sel_neighbors = np.where(conflicts[v] & selected)[0]
                if len(sel_neighbors) == 0:
                    selected[v] = True
                    improved = True
                    break
            
            if not improved:
                # Try 1-swap: find pair (u selected, v not selected) where v conflicts only with u
                # and removing u allows adding v plus possibly others
                # Simpler: try removing a selected vertex that has many conflicts,
                # then see if we can add more
                selected_indices = np.where(selected)[0]
                if len(selected_indices) == 0:
                    break
                
                # Find selected vertex with maximum degree to non-selected
                best_gain = 0
                best_u = None
                for u in selected_indices:
                    if time.time() > deadline - 15:
                        break
                    # If we remove u, which non-selected vertices become addable?
                    non_sel = np.where(~selected)[0]
                    # v is addable if its only conflict with selected is u
                    addable = 0
                    for v in non_sel:
                        if time.time() > deadline - 15:
                            break
                        sn = np.where(conflicts[v] & selected)[0]
                        if len(sn) == 1 and sn[0] == u:
                            addable += 1
                    gain = addable - 1  # net change
                    if gain > best_gain:
                        best_gain = gain
                        best_u = u
                
                if best_u is not None and best_gain > 0:
                    u = best_u
                    selected[u] = False
                    # Now add all non-selected vertices that only conflicted with u
                    non_sel = np.where(~selected)[0]
                    for v in non_sel:
                        if time.time() > deadline - 15:
                            break
                        sn = np.where(conflicts[v] & selected)[0]
                        if len(sn) == 0:
                            selected[v] = True
                    improved = True
        
        return selected
    
    # Run multiple strategies
    rng_main = np.random.default_rng(42)
    
    # Run greedy min degree with several restarts
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        rng_r = np.random.default_rng(1000 + restart)
        sel = greedy_min_degree(conflicts, N4, rng_r, deadline)
        # Local search
        sel_ls = local_search(sel, conflicts, N4, deadline)
        cnt = sel_ls.sum()
        if cnt > best_count:
            best_count = cnt
            best_selected = sel_ls.copy()
    
    # Run greedy clique coverage with several restarts
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        rng_r = np.random.default_rng(2000 + restart)
        sel = greedy_clique_coverage(conflicts, N4, vec_to_cliques, rng_r, deadline)
        sel_ls = local_search(sel, conflicts, N4, deadline)
        cnt = sel_ls.sum()
        if cnt > best_count:
            best_count = cnt
            best_selected = sel_ls.copy()
    
    # Spectral-based ordering
    if time.time() < deadline - 30:
        order = np.argsort(fiedler)
        alive = np.ones(N4, dtype=bool)
        selected = np.zeros(N4, dtype=bool)
        for i in order:
            if not alive[i]:
                continue
            selected[i] = True
            to_remove = np.where(conflicts[i] | (np.arange(N4) == i))[0]
            alive[to_remove] = False
        sel_ls = local_search(selected, conflicts, N4, deadline)
        cnt = sel_ls.sum()
        if cnt > best_count:
            best_count = cnt
            best_selected = sel_ls.copy()
    
    # More random restarts if time permits
    restart_num = 5
    while time.time() < deadline - 20:
        rng_r = np.random.default_rng(3000 + restart_num)
        order = rng_r.permutation(N4)
        alive = np.ones(N4, dtype=bool)
        selected = np.zeros(N4, dtype=bool)
        for i in order:
            if not alive[i]:
                continue
            selected[i] = True
            to_remove = np.where(conflicts[i] | (np.arange(N4) == i))[0]
            alive[to_remove] = False
        sel_ls = local_search(selected, conflicts, N4, deadline)
        cnt = sel_ls.sum()
        if cnt > best_count:
            best_count = cnt
            best_selected = sel_ls.copy()
        restart_num += 1
    
    if best_selected is None:
        best_selected = np.zeros(N4, dtype=bool)
    
    result_v4 = vectors_v4[best_selected]
    
    # Add all V1 vectors (they're all compatible with V4 and each other)
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
