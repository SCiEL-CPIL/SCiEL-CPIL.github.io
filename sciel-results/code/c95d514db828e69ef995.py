"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    # Generate Weight 3 and Weight 4 vectors
    candidates = []
    norms_sq = []

    # Weight 3
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(3)

    # Weight 4
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(4)

    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.array(norms_sq, dtype=np.int64)
    n_candidates = len(candidates)

    # Compute dot products
    dots = np.dot(candidates, candidates.T)

    # Determine conflicts
    # dist_sq[i, j] = norms_sq[i] + norms_sq[j] - 2*dots[i, j]
    # Condition for validity: dist_sq[i, j] >= max(norms_sq[i], norms_sq[j])
    # Conflict if dist_sq[i, j] < max(norms_sq[i], norms_sq[j])
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    max_norms_sq = np.maximum(norms_sq[:, None], norms_sq[None, :])
    conflicts = dist_sq < max_norms_sq
    np.fill_diagonal(conflicts, False)

    # Precompute neighbors for each node
    neighbors = [set() for _ in range(n_candidates)]
    for i in range(n_candidates):
        neigh_indices = np.where(conflicts[i])[0]
        neighbors[i] = set(neigh_indices.tolist())

    best_selected = []

    # Spectral relaxation: compute Laplacian L = D - A
    adj = np.zeros((n_candidates, n_candidates), dtype=np.float64)
    for i in range(n_candidates):
        for j in neighbors[i]:
            adj[i, j] = 1.0

    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj

    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh

        L_sparse = csr_matrix(L)
        k_eig = min(50, n_candidates - 1)
        evals, evecs = eigsh(L_sparse, k=k_eig, which='SM')

        idx = np.argsort(evals)
        evals = evals[idx]
        evecs = evecs[:, idx]

        # Try multiple eigenvector combinations
        for eig_idx in range(1, min(10, k_eig)):
            if time.time() > deadline - 10.0:
                break

            vec = evecs[:, eig_idx]
            selected_idx = np.where(vec > 0)[0]

            available = set(selected_idx.tolist())
            refined = []
            while available:
                if time.time() > deadline - 5.0:
                    break
                if len(available) > 100:
                    sample = random.sample(list(available), 100)
                else:
                    sample = list(available)

                best_v = None
                min_deg = float('inf')
                for v in sample:
                    deg = len(neighbors[v] & available)
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v

                if best_v is None:
                    break

                refined.append(best_v)
                available.discard(best_v)
                for n in neighbors[best_v]:
                    if n in available:
                        available.discard(n)

            if len(refined) > len(best_selected):
                best_selected = refined

        for eig_idx in range(1, min(10, k_eig)):
            if time.time() > deadline - 10.0:
                break

            vec = evecs[:, eig_idx]
            selected_idx = np.where(vec < 0)[0]

            available = set(selected_idx.tolist())
            refined = []
            while available:
                if time.time() > deadline - 5.0:
                    break
                if len(available) > 100:
                    sample = random.sample(list(available), 100)
                else:
                    sample = list(available)

                best_v = None
                min_deg = float('inf')
                for v in sample:
                    deg = len(neighbors[v] & available)
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v

                if best_v is None:
                    break

                refined.append(best_v)
                available.discard(best_v)
                for n in neighbors[best_v]:
                    if n in available:
                        available.discard(n)

            if len(refined) > len(best_selected):
                best_selected = refined

    except Exception:
        best_selected = []

    # WalkSAT-style local search for Max-Clique on conflict graph
    if time.time() < deadline - 10.0:
        # Convert to Max-Clique: find largest set of mutually non-conflicting vertices
        # This is equivalent to Max-Independent-Set in the conflict graph
        
        # Build conflict graph adjacency (conflict = edge)
        # We want largest independent set
        
        # Use WalkSAT-inspired approach:
        # 1. Start with random partial assignment (selected set)
        # 2. Try to add vertices that don't conflict
        # 3. If stuck, remove a random vertex and try different additions
        
        random.seed(42)
        np.random.seed(42)
        
        best_size = len(best_selected)
        best_set = set(best_selected) if best_selected else set()
        
        def greedy_fill(start_set):
            """Fill a starting set greedily with non-conflicting vertices."""
            selected = set(start_set)
            available = set(range(n_candidates)) - selected
            while available:
                if time.time() > deadline - 5.0:
                    break
                # Pick vertex with minimum conflicts in available
                if len(available) > 150:
                    sample = random.sample(list(available), 150)
                else:
                    sample = list(available)
                
                best_v = None
                min_conf = float('inf')
                for v in sample:
                    conf = len(neighbors[v] & available)
                    if conf < min_conf:
                        min_conf = conf
                        best_v = v
                
                if best_v is None:
                    break
                
                selected.add(best_v)
                available.discard(best_v)
                for n in neighbors[best_v]:
                    if n in available:
                        available.discard(n)
            
            return selected
        
        # Multiple random restarts with WalkSAT-style moves
        num_restarts = 30
        for restart in range(num_restarts):
            if time.time() > deadline - 15.0:
                break
            
            # Start with random subset
            random_start_size = random.randint(5, 20)
            random_start = set(random.sample(range(n_candidates), min(random_start_size, n_candidates)))
            
            # Remove conflicts from random start
            valid_start = set()
            for v in random_start:
                if not (neighbors[v] & valid_start):
                    valid_start.add(v)
            
            # Greedy fill from this start
            current_set = greedy_fill(valid_start)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            
            # Local search: try 1-swap moves
            for _ in range(50):
                if time.time() > deadline - 10.0:
                    break
                
                if not current_set:
                    break
                
                # Try removing one vertex and adding up to two others
                v_remove = random.choice(list(current_set))
                trial_set = current_set - {v_remove}
                
                # Find vertices that can be added
                addable = []
                for v in range(n_candidates):
                    if v in trial_set or v == v_remove:
                        continue
                    if not (neighbors[v] & trial_set):
                        addable.append(v)
                
                if not addable:
                    continue
                
                # Try adding one vertex
                v_add1 = random.choice(addable)
                new_set1 = trial_set | {v_add1}
                if len(new_set1) > len(current_set):
                    current_set = new_set1
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                    continue
                
                # Try adding two vertices
                if len(addable) >= 2:
                    v_add2_candidates = [v for v in addable if v != v_add1 and not (neighbors[v] & new_set1)]
                    if v_add2_candidates:
                        v_add2 = random.choice(v_add2_candidates)
                        if v_add1 not in neighbors[v_add2]:
                            new_set2 = new_set1 | {v_add2}
                            if len(new_set2) > len(current_set):
                                current_set = new_set2
                                if len(current_set) > best_size:
                                    best_size = len(current_set)
                                    best_set = set(current_set)
        
        if len(best_set) > len(best_selected):
            best_selected = list(best_set)

    # Local search improvement on best_selected
    if best_selected and time.time() < deadline - 5.0:
        selected_set = set(best_selected)

        # Try to add more vertices
        added = True
        while added and time.time() < deadline - 5.0:
            added = False
            for v in range(n_candidates):
                if v in selected_set:
                    continue
                if not (neighbors[v] & selected_set):
                    selected_set.add(v)
                    added = True

    # Sanitize: Ensure the final set is strictly valid
    # This guards against any logical errors in the search process
    if best_selected:
        final_valid = []
        for vi in best_selected:
            vec_i = candidates[vi]
            n_sq_i = norms_sq[vi]
            ok = True
            for vj in final_valid:
                vec_j = candidates[vj]
                n_sq_j = norms_sq[vj]
                dot = np.dot(vec_i, vec_j)
                d_sq = n_sq_i + n_sq_j - 2 * dot
                if d_sq < max(n_sq_i, n_sq_j):
                    ok = False
                    break
            if ok:
                final_valid.append(vi)
        best_selected = final_valid

    if not best_selected:
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)

    return candidates[best_selected]
# EVOLVE-BLOCK-END
