import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    num_triples = len(triples)

    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Precompute coverage structure
    block_triple_indices = []
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i, b in enumerate(all_blocks):
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris.append(idx)
            triple_to_blocks[idx].append(i)
        block_triple_indices.append(tris)

    # Vectorized representations
    max_tris = 35  # C(7,3)
    block_triple_matrix = np.full((num_blocks, max_tris), -1, dtype=np.int32)
    block_triple_lens = np.zeros(num_blocks, dtype=np.int32)
    block_triple_arrays = []
    for i, tris in enumerate(block_triple_indices):
        block_triple_matrix[i, :len(tris)] = tris
        block_triple_lens[i] = len(tris)
        block_triple_arrays.append(np.array(tris, dtype=np.int32))

    triple_to_blocks_arr = [np.array(b, dtype=np.int32) for b in triple_to_blocks]

    best_sol = None
    best_energy = float('inf')
    seed_counter = 0

    while time.time() < deadline - 12:
        seed_counter += 1
        rng = random.Random(seed_counter)

        # 1. Fast incremental greedy with randomized tie-breaking
        sol = _fast_greedy(rng, num_blocks, block_triple_arrays, triple_to_blocks_arr,
                           num_triples, deadline - 10)

        # 2. Remove redundant blocks
        sol = _remove_redundant(sol, block_triple_arrays, num_triples)

        # 3. Local search (SA) with adaptive temperature
        time_left_for_sa = deadline - 10 - time.time()
        if time_left_for_sa < 1:
            break

        sol = _local_search_v2(sol, num_blocks, block_triple_arrays,
                               triple_to_blocks_arr, num_triples, time_left_for_sa, rng)

        # 4. High-precision iterative polish
        sol = _polish(sol, num_blocks, block_triple_arrays, triple_to_blocks_arr, num_triples)

        # Evaluate
        if sol:
            covered = np.zeros(num_triples, dtype=bool)
            for b_idx in sol:
                tris = block_triple_arrays[b_idx]
                if len(tris) > 0:
                    covered[tris] = True
            if not np.any(~covered):
                energy = len(sol)
                if energy < best_energy:
                    best_energy = energy
                    best_sol = list(sol)

    if best_sol is None:
        return []
    return [list(all_blocks[i]) for i in best_sol]


def _fast_greedy(rng, num_blocks, block_triple_arrays, triple_to_blocks_arr, num_triples, deadline):
    """Incremental greedy: maintain gain per block, update only affected blocks."""
    gain = np.array([len(ts) for ts in block_triple_arrays], dtype=np.int32)
    is_uncovered = np.ones(num_triples, dtype=bool)
    selected = []
    selected_set = set()

    while np.any(is_uncovered) and time.time() < deadline:
        max_gain = np.max(gain)
        if max_gain <= 0:
            break
        candidates = np.where(gain == max_gain)[0]
        # Exclude already-selected
        if selected_set:
            cand_list = candidates.tolist()
            cand_list = [c for c in cand_list if c not in selected_set]
            if not cand_list:
                break
            chosen = int(rng.choice(cand_list))
        else:
            chosen = int(rng.choice(candidates))

        selected.append(chosen)
        selected_set.add(chosen)

        tris = block_triple_arrays[chosen]
        for t_idx in tris:
            if is_uncovered[t_idx]:
                is_uncovered[t_idx] = False
                # Decrease gain for all blocks covering this triple
                gain[triple_to_blocks_arr[t_idx]] -= 1

    return selected


def _remove_redundant(sol, block_triple_arrays, num_triples):
    if not sol:
        return sol
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_arrays[idx]
        if len(tris) > 0:
            counts[tris] += 1
    changed = True
    while changed:
        changed = False
        for idx in list(sol):
            tris = block_triple_arrays[idx]
            if len(tris) == 0:
                sol.remove(idx)
                changed = True
                continue
            if np.all(counts[tris] > 1):
                counts[tris] -= 1
                sol.remove(idx)
                changed = True
    return sol


def _local_search_v2(initial_sol, num_blocks, block_triple_arrays,
                     triple_to_blocks_arr, num_triples, time_limit, rng):
    sol = list(initial_sol)
    sol_set = set(sol)
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_arrays[idx]
        if len(tris) > 0:
            counts[tris] += 1
    current_uncovered = int(np.sum(counts == 0))

    def energy(n, unc):
        w = 50 + 50 * unc if unc > 0 else 1
        return n + w * unc

    current_energy = energy(len(sol), current_uncovered)
    best_sol = list(sol)
    best_energy = current_energy

    T = 8.0
    cooling = 0.997
    min_T = 0.05

    while time.time() < time_limit and T > min_T:
        move_type = rng.random()

        if move_type < 0.5 and sol:
            to_remove = rng.choice(sol)
            tris = block_triple_arrays[to_remove]
            if len(tris) == 0:
                continue
            min_count = int(np.min(counts[tris]))
            if min_count > 1:
                counts[tris] -= 1
                sol.remove(to_remove)
                sol_set.discard(to_remove)
                current_uncovered = int(np.sum(counts == 0))
                current_energy = energy(len(sol), current_uncovered)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_sol = list(sol)
            else:
                delta_unc = int(np.sum(counts[tris] == 1))
                new_unc = current_uncovered + delta_unc
                new_energy = energy(len(sol) - 1, new_unc)
                delta_e = new_energy - current_energy
                if delta_e <= 0 or rng.random() < np.exp(-delta_e / T):
                    counts[tris] -= 1
                    sol.remove(to_remove)
                    sol_set.discard(to_remove)
                    current_uncovered = new_unc
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)

        elif move_type < 0.85 and sol:
            to_remove = rng.choice(sol)
            tris_remove = block_triple_arrays[to_remove]
            if len(tris_remove) == 0:
                continue
            uncovered_mask = (counts[tris_remove] == 1)
            uncovered_tris = tris_remove[uncovered_mask]
            if len(uncovered_tris) == 0:
                continue
            candidate_blocks = set()
            for t_idx in uncovered_tris:
                candidate_blocks.update(triple_to_blocks_arr[t_idx].tolist())
            candidate_blocks.difference_update(sol_set)
            if not candidate_blocks:
                continue
            candidates_list = list(candidate_blocks)
            if len(candidates_list) > 50:
                candidates_list = rng.sample(candidates_list, 50)
            best_candidate = None
            best_score = -1
            for cand in candidates_list:
                tris_add = block_triple_arrays[cand]
                score = int(np.sum(np.isin(tris_add, uncovered_tris)))
                if score > best_score:
                    best_score = score
                    best_candidate = cand
            if best_candidate is None or best_score == 0:
                continue
            to_add = best_candidate
            tris_add = block_triple_arrays[to_add]
            new_counts = counts.copy()
            new_counts[tris_remove] -= 1
            new_counts[tris_add] += 1
            new_unc = int(np.sum(new_counts == 0))
            new_energy = energy(len(sol), new_unc)
            delta_e = new_energy - current_energy
            if delta_e <= 0 or rng.random() < np.exp(-delta_e / T):
                counts[tris_remove] -= 1
                counts[tris_add] += 1
                sol.remove(to_remove)
                sol_set.discard(to_remove)
                sol.append(to_add)
                sol_set.add(to_add)
                current_uncovered = new_unc
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_sol = list(sol)

        elif move_type < 0.95 and len(sol) >= 2:
            idx1, idx2 = rng.sample(range(len(sol)), 2)
            to_remove1 = sol[idx1]
            to_remove2 = sol[idx2]
            tris1 = block_triple_arrays[to_remove1]
            tris2 = block_triple_arrays[to_remove2]
            if len(tris1) == 0 or len(tris2) == 0:
                continue
            sim_counts = counts.copy()
            sim_counts[tris1] -= 1
            sim_counts[tris2] -= 1
            uncovered_tris = np.where(sim_counts == 0)[0]
            if len(uncovered_tris) == 0:
                new_energy = energy(len(sol) - 2, 0)
                delta_e = new_energy - current_energy
                if delta_e <= 0 or rng.random() < np.exp(-delta_e / T):
                    counts[tris1] -= 1
                    counts[tris2] -= 1
                    sol.remove(to_remove1)
                    sol.remove(to_remove2)
                    sol_set.discard(to_remove1)
                    sol_set.discard(to_remove2)
                    current_uncovered = 0
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)
            else:
                candidate_blocks = set()
                for t_idx in uncovered_tris[:10]:
                    candidate_blocks.update(triple_to_blocks_arr[t_idx].tolist())
                candidate_blocks.difference_update(sol_set)
                if not candidate_blocks:
                    continue
                candidates_list = list(candidate_blocks)
                if len(candidates_list) > 30:
                    candidates_list = rng.sample(candidates_list, 30)
                best_candidate = None
                best_score = -1
                for cand in candidates_list:
                    tris_cand = block_triple_arrays[cand]
                    score = int(np.sum(np.isin(tris_cand, uncovered_tris)))
                    if score > best_score:
                        best_score = score
                        best_candidate = cand
                if best_candidate is None or best_score == 0:
                    continue
                to_add = best_candidate
                tris_add = block_triple_arrays[to_add]
                sim_counts[tris_add] += 1
                new_unc = int(np.sum(sim_counts == 0))
                new_energy = energy(len(sol) - 1, new_unc)
                delta_e = new_energy - current_energy
                if delta_e <= 0 or rng.random() < np.exp(-delta_e / T):
                    counts[tris1] -= 1
                    counts[tris2] -= 1
                    counts[tris_add] += 1
                    sol.remove(to_remove1)
                    sol.remove(to_remove2)
                    sol_set.discard(to_remove1)
                    sol_set.discard(to_remove2)
                    sol.append(to_add)
                    sol_set.add(to_add)
                    current_uncovered = new_unc
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)

        T *= cooling
        if time.time() > time_limit:
            break

    return best_sol


def _polish(sol, num_blocks, block_triple_arrays, triple_to_blocks_arr, num_triples):
    if not sol:
        return sol
    sol_set = set(sol)
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_arrays[idx]
        if len(tris) > 0:
            counts[tris] += 1

    # Iterative redundant removal to fixpoint
    changed = True
    while changed:
        changed = False
        for idx in list(sol):
            tris = block_triple_arrays[idx]
            if len(tris) == 0:
                sol.remove(idx)
                sol_set.discard(idx)
                changed = True
                continue
            if np.all(counts[tris] > 1):
                counts[tris] -= 1
                sol.remove(idx)
                sol_set.discard(idx)
                changed = True

    # Iterative pair-replacement: remove two blocks, add one covering all lost triples
    improved = True
    while improved:
        improved = False
        for i in range(len(sol)):
            for j in range(i + 1, len(sol)):
                b1 = sol[i]
                b2 = sol[j]
                tris1 = block_triple_arrays[b1]
                tris2 = block_triple_arrays[b2]
                if len(tris1) == 0 or len(tris2) == 0:
                    continue
                sim_counts = counts.copy()
                sim_counts[tris1] -= 1
                sim_counts[tris2] -= 1
                uncovered_tris = np.where(sim_counts == 0)[0]
                if len(uncovered_tris) == 0:
                    counts[tris1] -= 1
                    counts[tris2] -= 1
                    sol.remove(b1)
                    sol.remove(b2)
                    sol_set.discard(b1)
                    sol_set.discard(b2)
                    improved = True
                    break
                elif len(uncovered_tris) <= 5:
                    candidate_blocks = set()
                    for t_idx in uncovered_tris:
                        candidate_blocks.update(triple_to_blocks_arr[t_idx].tolist())
                    candidate_blocks.difference_update(sol_set)
                    found = False
                    if candidate_blocks:
                        for cand in candidate_blocks:
                            tris_cand = block_triple_arrays[cand]
                            if np.all(np.isin(uncovered_tris, tris_cand)):
                                counts[tris1] -= 1
                                counts[tris2] -= 1
                                counts[tris_cand] += 1
                                sol.remove(b1)
                                sol.remove(b2)
                                sol_set.discard(b1)
                                sol_set.discard(b2)
                                sol.append(cand)
                                sol_set.add(cand)
                                improved = True
                                found = True
                                break
                    if found:
                        break
        # Recompute counts after structural changes to stay consistent
        if improved:
            counts = np.zeros(num_triples, dtype=np.int32)
            for idx in sol:
                tris = block_triple_arrays[idx]
                if len(tris) > 0:
                    counts[tris] += 1
            sol_set = set(sol)

    return sol
