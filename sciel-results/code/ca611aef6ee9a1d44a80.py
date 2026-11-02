"""Program for kissing number in dimension 11: MIS on conflict graph with SA."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate all squared-norm-4 integer vectors in R^11
    # Type 1: weight 4, entries in {-1,0,1}
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)

    # Type 2: weight 2, entries in {-2,0,2} (norm^2 = 4+4=8? No, that's 8)
    # Actually weight 1 with entry ±2: norm^2 = 4
    v2_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v2_list.append(v)

    # Type 3: weight 2, entries ±2? norm^2 = 8, no.
    # What about entries ±1, ±2 mixed? e.g., one 2 and one 0... no.
    # Let's just use the two types above.
    all_vectors = np.array(v4_list + v2_list, dtype=np.int64)
    N = len(all_vectors)

    if N == 0:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    # Build conflict graph: two vectors conflict if their dot product > 2
    # (since ||a-b||^2 = 8 - 2*a.b >= 4 requires a.b <= 2)
    dots = all_vectors @ all_vectors.T
    conflict = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)

    # Precompute adjacency as sparse-like structure
    adj = [np.where(conflict[i] == 1)[0] for i in range(N)]
    deg = np.array([len(a) for a in adj], dtype=np.int32)

    rng = np.random.default_rng(42)

    best_size = 0
    best_set = np.zeros(N, dtype=bool)

    def init_greedy():
        """Greedy initialization: repeatedly pick vertex with minimum degree among alive."""
        selected = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        cur_deg = deg.copy()
        
        while np.any(alive) and time.time() < deadline - 10:
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0:
                break
            min_d = np.min(cur_deg[alive_idx])
            cands = alive_idx[cur_deg[alive_idx] == min_d]
            v = cands[rng.integers(len(cands))]
            selected[v] = True
            to_kill = np.unique(np.concatenate(([v], adj[v])))
            alive[to_kill] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                cur_deg[rem] -= conflict[rem][:, to_kill].sum(axis=1)
        
        return selected

    def sa_search(init_set, time_budget):
        """
        Simulated annealing on the conflict graph.
        State: set of selected vertices (may be invalid, i.e., contain conflicts).
        Energy: -size + penalty * (number of conflicting pairs in set)
        Moves: add a vertex, remove a vertex, or swap.
        """
        in_set = init_set.copy()
        # Track conflict count: number of pairs (u,v) in in_set with conflict[u,v]=1
        conflict_count = 0
        for i in np.where(in_set)[0]:
            nbrs_in_set = adj[i][in_set[adj[i]]]
            conflict_count += len(nbrs_in_set)
        conflict_count //= 2  # each pair counted twice

        size = in_set.sum()
        penalty = 10.0  # strong penalty to prefer valid sets
        energy = -size + penalty * conflict_count

        T = 1.0
        cooling = 0.995
        min_T = 0.05
        sa_end = time.time() + time_budget

        best_local_size = size
        best_local_set = in_set.copy()
        best_local_energy = energy

        # Precompute: for each vertex, its neighbors
        # For fast conflict count updates

        while time.time() < sa_end:
            # Choose a move type
            move_type = rng.integers(3)

            if move_type == 0:
                # Add a vertex not in set
                not_in = np.where(~in_set)[0]
                if len(not_in) == 0:
                    continue
                v = not_in[rng.integers(len(not_in))]
                
                # New conflict count: add conflicts between v and existing in_set
                new_conflicts = in_set[adj[v]].sum()
                delta_energy = -1 + penalty * new_conflicts
                
                if rng.random() < np.exp(-delta_energy / max(T, 1e-10)):
                    in_set[v] = True
                    size += 1
                    conflict_count += new_conflicts
                    energy += delta_energy

            elif move_type == 1:
                # Remove a vertex from set
                in_idx = np.where(in_set)[0]
                if len(in_idx) == 0:
                    continue
                v = in_idx[rng.integers(len(in_idx))]
                
                # Remove conflicts between v and other in_set vertices
                removed_conflicts = in_set[adj[v]].sum()
                delta_energy = 1 - penalty * removed_conflicts
                
                if rng.random() < np.exp(-delta_energy / max(T, 1e-10)):
                    in_set[v] = False
                    size -= 1
                    conflict_count -= removed_conflicts
                    energy += delta_energy

            else:
                # Swap: remove one, add another
                in_idx = np.where(in_set)[0]
                not_in = np.where(~in_set)[0]
                if len(in_idx) == 0 or len(not_in) == 0:
                    continue
                v_out = in_idx[rng.integers(len(in_idx))]
                v_in = not_in[rng.integers(len(not_in))]
                
                if v_in == v_out:
                    continue
                
                # Compute delta
                # Remove v_out: lose its conflicts with other in_set (excluding v_in if it was adjacent)
                conflicts_out = in_set[adj[v_out]].sum()
                # But v_in is not yet in set, so just count conflicts with current in_set
                # Add v_in: gain its conflicts with current in_set (after removing v_out)
                # Let's compute more carefully
                in_set_after_remove = in_set.copy()
                in_set_after_remove[v_out] = False
                conflicts_in = in_set_after_remove[adj[v_in]].sum()
                
                delta_conflict = conflicts_in - conflicts_out
                delta_energy = penalty * delta_conflict  # size unchanged
                
                if rng.random() < np.exp(-delta_energy / max(T, 1e-10)):
                    in_set[v_out] = False
                    in_set[v_in] = True
                    conflict_count += delta_conflict
                    energy += delta_energy

            # Track best
            if conflict_count == 0 and size > best_local_size:
                best_local_size = size
                best_local_set = in_set.copy()
                best_local_energy = energy
            elif size > best_local_size and conflict_count == 0:
                best_local_size = size
                best_local_set = in_set.copy()
                best_local_energy = energy

            T *= cooling
            if T < min_T:
                T = min_T

        # At the end, if we have conflicts, try to fix by removing conflicting vertices
        if conflict_count > 0:
            # Greedily remove vertices to eliminate conflicts
            while conflict_count > 0 and time.time() < deadline - 5:
                # Find all vertices involved in conflicts
                conflict_vertices = set()
                for i in np.where(in_set)[0]:
                    nbrs_in = adj[i][in_set[adj[i]]]
                    if len(nbrs_in) > 0:
                        conflict_vertices.add(i)
                        conflict_vertices.update(nbrs_in.tolist())
                
                if not conflict_vertices:
                    break
                    
                # Remove the vertex with most conflicts
                max_conf = -1
                v_to_remove = -1
                for v in conflict_vertices:
                    if in_set[v]:
                        c = in_set[adj[v]].sum()
                        if c > max_conf:
                            max_conf = c
                            v_to_remove = v
                
                if v_to_remove == -1:
                    break
                    
                # Remove v_to_remove
                in_set[v_to_remove] = False
                removed_conflicts = in_set[adj[v_to_remove]].sum() + (1 if conflict_count > 0 else 0)
                # Recompute
                new_cc = 0
                for i in np.where(in_set)[0]:
                    new_cc += in_set[adj[i]].sum()
                new_cc //= 2
                conflict_count = new_cc

        return best_local_set, best_local_size

    def hill_climb_from_valid(valid_set):
        """From a valid independent set, try to add vertices one by one."""
        in_set = valid_set.copy()
        size = in_set.sum()
        hc_end = time.time() + 5.0
        
        while time.time() < hc_end:
            not_in = np.where(~in_set)[0]
            if len(not_in) == 0:
                break
            # Find vertices with no conflicts with in_set
            safe = []
            for v in not_in:
                if in_set[adj[v]].sum() == 0:
                    safe.append(v)
            if not safe:
                break
            v = safe[rng.integers(len(safe))]
            in_set[v] = True
            size += 1
        
        return in_set, size

    # Main loop: multiple SA restarts
    n_restarts = 0
    max_restarts = 50

    while time.time() < deadline - 15 and n_restarts < max_restarts:
        n_restarts += 1
        remaining = deadline - 15 - time.time()
        if remaining < 3:
            break

        # Greedy init
        init = init_greedy()
        init_size = init.sum()

        # Check if init is valid (should be, since greedy ensures independence)
        # SA search
        sa_time = min(remaining * 0.5, 10.0)
        sa_set, sa_size = sa_search(init, sa_time)

        if sa_size > best_size:
            best_size = sa_size
            best_set = sa_set.copy()

        # Hill climb from the SA result if it's valid
        remaining2 = deadline - 10 - time.time()
        if remaining2 > 2 and sa_size >= init_size:
            hc_set, hc_size = hill_climb_from_valid(sa_set)
            if hc_size > best_size:
                best_size = hc_size
                best_set = hc_set.copy()

    # Final: ensure we have a valid set
    # Verify best_set is independent
    if best_size == 0:
        # Fallback: use greedy
        best_set = init_greedy()
        best_size = best_set.sum()

    final_vectors = all_vectors[best_set]
    return final_vectors

# EVOLVE-BLOCK-END
