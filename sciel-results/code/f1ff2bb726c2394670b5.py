"""Combined: Lagrangian-guided WalkSAT + vectorized greedy + pair/triple-replacement refinement.
Uses smooth log-sum-exp surrogate for coverage deficit with dynamic penalty weights,
adaptive penalty momentum, and a final high-precision systematic polish pass."""
import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

_RNG = random.Random(20250101)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    # Precompute all t-subsets
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {ts: i for i, ts in enumerate(t_subs)}
    n_t = len(t_subs)

    # Precompute all k-subsets and their covered t-subset indices
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    block_covers = []
    for ks in k_subs:
        covered = []
        for ts in itertools.combinations(ks, t):
            covered.append(t_sub_index[ts])
        block_covers.append(covered)

    # Inverse: t-subset -> list of blocks covering it
    t_to_blocks = [[] for _ in range(n_t)]
    for bi, cov in enumerate(block_covers):
        for ti in cov:
            t_to_blocks[ti].append(bi)

    # Convert to numpy for speed
    block_covers_np = [np.array(c, dtype=np.int32) for c in block_covers]
    t_to_blocks_np = [np.array(b, dtype=np.int32) for b in t_to_blocks]

    # Precompute: for each block, the number of t-subsets it covers
    block_sizes = np.array([len(c) for c in block_covers], dtype=np.int32)

    # ---------------- Vectorized Greedy with frequency-aware tie-breaking ----------------
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        gain = np.zeros(n_k, dtype=np.int32)
        for bi in range(n_k):
            gain[bi] = np.sum(uncovered[block_covers_np[bi]])
        blocks_used = []
        while np.any(uncovered):
            if time.time() > deadline - 35.0:
                break
            max_gain = gain.max()
            if max_gain == 0:
                break
            candidates = np.where(gain == max_gain)[0]
            # Frequency-aware: prefer blocks that cover more currently-uncovered subsets
            # with less overlap with already-selected blocks
            if len(candidates) > 1:
                # Weight by block size (more coverage per block)
                cands_gains = gain[candidates]
                cands_sizes = block_sizes[candidates]
                # Score: gain * size (prefer larger blocks among equal gain)
                scores = cands_gains.astype(np.float64) * cands_sizes.astype(np.float64)
                max_score = scores.max()
                top_cands = candidates[scores == max_score]
                chosen = top_cands[rng.randint(0, len(top_cands) - 1)]
            else:
                chosen = candidates[0]
            blocks_used.append(chosen)
            for ts_idx in block_covers_np[chosen]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    for bi2 in t_to_blocks_np[ts_idx]:
                        gain[bi2] -= 1
            gain[chosen] = -1
        return blocks_used

    # Run multiple greedy restarts, keep best
    best_greedy = None
    best_greedy_size = n_k + 1
    for seed in range(10):
        if time.time() > deadline - 55.0:
            break
        blocks = greedy_cover(seed)
        if len(blocks) < best_greedy_size:
            best_greedy_size = len(blocks)
            best_greedy = blocks
        if len(blocks) <= 135:
            break

    if best_greedy is None:
        best_greedy = greedy_cover(0)

    selected = set(best_greedy)

    # Build cover_count
    cover_count = np.zeros(n_t, dtype=np.int32)
    for bi in selected:
        cover_count[block_covers_np[bi]] += 1

    # ---------------- Deletion reduction ----------------
    def reduce_blocks(selected_set, cc):
        changed = True
        while changed:
            changed = False
            for bi in list(selected_set):
                if time.time() > deadline - 25.0:
                    return selected_set, cc
                cc[block_covers_np[bi]] -= 1
                if np.any(cc[block_covers_np[bi]] == 0):
                    cc[block_covers_np[bi]] += 1
                else:
                    selected_set.discard(bi)
                    changed = True
        return selected_set, cc

    selected, cover_count = reduce_blocks(selected, cover_count)

    # ---------------- Pair replacement: try to replace 2 blocks with 1 ----------------
    def pair_replace(selected_set, cc, max_trials=800):
        rng = _RNG
        n_sel = len(selected_set)
        if n_sel < 2:
            return selected_set, cc
        sel_list = list(selected_set)
        for trial in range(max_trials):
            if time.time() > deadline - 35.0:
                break
            i, j = rng.sample(range(n_sel), 2)
            bi, bj = sel_list[i], sel_list[j]
            # Union of t-subsets covered by bi and bj
            union_t = set(block_covers_np[bi].tolist()) | set(block_covers_np[bj].tolist())
            if len(union_t) > 35:
                continue
            # Check if any single block covers all t-subsets in union_t
            common = None
            for ti in union_t:
                blocks_covering = set(t_to_blocks[ti])
                if common is None:
                    common = blocks_covering
                else:
                    common &= blocks_covering
                if not common:
                    break
            if not common:
                continue
            for cand in common:
                if cand == bi or cand == bj:
                    continue
                cc[block_covers_np[bi]] -= 1
                cc[block_covers_np[bj]] -= 1
                cc[block_covers_np[cand]] += 1
                if np.all(cc >= 1):
                    selected_set.discard(bi)
                    selected_set.discard(bj)
                    selected_set.add(cand)
                    n_sel = len(selected_set)
                    sel_list = list(selected_set)
                    selected_set, cc = reduce_blocks(selected_set, cc)
                    n_sel = len(selected_set)
                    sel_list = list(selected_set)
                    break
                else:
                    cc[block_covers_np[bi]] += 1
                    cc[block_covers_np[bj]] += 1
                    cc[block_covers_np[cand]] -= 1
        return selected_set, cc

    selected, cover_count = pair_replace(selected, cover_count)

    # ---------------- Triple replacement: try to replace 3 blocks with 1 or 2 ----------------
    def triple_replace(selected_set, cc, max_trials=400):
        rng = _RNG
        n_sel = len(selected_set)
        if n_sel < 3:
            return selected_set, cc
        sel_list = list(selected_set)
        for trial in range(max_trials):
            if time.time() > deadline - 35.0:
                break
            i, j, m = rng.sample(range(n_sel), 3)
            bi, bj, bm = sel_list[i], sel_list[j], sel_list[m]
            # Union of t-subsets covered by all three
            union_t = (set(block_covers_np[bi].tolist()) |
                       set(block_covers_np[bj].tolist()) |
                       set(block_covers_np[bm].tolist()))
            if len(union_t) > 50:
                continue
            # Try replacing with 1 block
            common = None
            for ti in union_t:
                blocks_covering = set(t_to_blocks[ti])
                if common is None:
                    common = blocks_covering
                else:
                    common &= blocks_covering
                if not common:
                    break
            if common:
                for cand in common:
                    if cand in (bi, bj, bm):
                        continue
                    cc[block_covers_np[bi]] -= 1
                    cc[block_covers_np[bj]] -= 1
                    cc[block_covers_np[bm]] -= 1
                    cc[block_covers_np[cand]] += 1
                    if np.all(cc >= 1):
                        selected_set.discard(bi)
                        selected_set.discard(bj)
                        selected_set.discard(bm)
                        selected_set.add(cand)
                        n_sel = len(selected_set)
                        sel_list = list(selected_set)
                        selected_set, cc = reduce_blocks(selected_set, cc)
                        n_sel = len(selected_set)
                        sel_list = list(selected_set)
                        break
                    else:
                        cc[block_covers_np[bi]] += 1
                        cc[block_covers_np[bj]] += 1
                        cc[block_covers_np[bm]] += 1
                        cc[block_covers_np[cand]] -= 1
        return selected_set, cc

    selected, cover_count = triple_replace(selected, cover_count)

    # ---------------- Lagrangian-guided WalkSAT with smooth log-sum-exp surrogate ----------------
    x = np.array([bi in selected for bi in range(n_k)], dtype=bool)
    n_true = x.sum()

    clause_sat = np.zeros(n_t, dtype=np.int32)
    for bi in range(n_k):
        if x[bi]:
            clause_sat[block_covers_np[bi]] += 1

    # Penalty weights with momentum
    penalty = np.ones(n_t, dtype=np.float64)
    penalty_momentum = np.zeros(n_t, dtype=np.float64)

    def smooth_cost(x_arr, clause_sat_arr, penalty_arr):
        """Smooth cost using log-sum-exp surrogate for coverage deficit.
        
        For each clause i:
        - If sat_i == 0: deficit = 1.0 (fully uncovered)
        - If sat_i >= 1: deficit = 1/(sat_i + 1) (smooth decay)
        
        We use a soft formulation: deficit_i = exp(-alpha * max(sat_i, 0))
        which gives ~1.0 for sat=0, ~0.6 for sat=1, ~0.37 for sat=2, etc.
        Combined with the penalty weight, this creates a smooth landscape.
        """
        n_true_val = float(x_arr.sum())
        # Smooth deficit via exponential decay
        # deficit_i = exp(-alpha * sat_i) for sat_i >= 0
        # For sat_i = 0: deficit = 1.0
        # For sat_i = 1: deficit = exp(-alpha)
        # For sat_i = 2: deficit = exp(-2*alpha)
        alpha = 0.5
        safe_sat = np.maximum(clause_sat_arr.astype(np.float64), 0.0)
        deficit = np.exp(-alpha * safe_sat)
        # Scale: uncovered (sat=0) gives deficit=1, covered (sat>=1) gives < 1
        # We want the penalty to be 0 when sat >= 1 ideally, so use:
        # soft_deficit = 1.0 if sat == 0, else exp(-alpha * (sat - 1))
        # This gives: sat=0 -> 1.0, sat=1 -> 1.0, sat=2 -> exp(-alpha), ...
        # Actually let's use: soft_deficit = 1.0 / (1.0 + sat_i)
        # sat=0 -> 1.0, sat=1 -> 0.5, sat=2 -> 0.333, sat=3 -> 0.25
        soft_deficit = 1.0 / (1.0 + safe_sat)
        pen_val = np.sum(penalty_arr * soft_deficit)
        return n_true_val + 30.0 * pen_val

    best_x = x.copy()
    best_cost = smooth_cost(x, clause_sat, penalty)
    best_n_true = int(n_true)

    # WalkSAT with Lagrangian guidance and smooth cost
    remaining = deadline - time.time() - 25.0
    if remaining > 10.0:
        max_steps = int(min(remaining * 15000, 800000))
        max_steps = max(max_steps, 8000)

        unsat_count = int((clause_sat == 0).sum())
        true_blocks = np.where(x)[0].tolist()

        # Track penalty history for momentum
        penalty_history = penalty.copy()

        for step in range(max_steps):
            if time.time() > deadline - 25.0:
                break

            if unsat_count > 0:
                # Pick an unsatisfied clause, weighted by penalty
                unsat_mask = clause_sat == 0
                unsat_indices = np.where(unsat_mask)[0]
                weights = penalty[unsat_indices]
                total_w = weights.sum()
                if total_w == 0:
                    ti = unsat_indices[_RNG.randint(0, len(unsat_indices) - 1)]
                else:
                    probs = weights / total_w
                    ti = unsat_indices[_RNG.choice(len(unsat_indices), p=probs)]

                # Pick a block covering this clause
                cands = t_to_blocks[ti]
                bi = cands[_RNG.randint(0, len(cands) - 1)]

                # Flip
                old = x[bi]
                x[bi] = not old
                delta = 1 if x[bi] else -1
                affected = block_covers_np[bi]
                clause_sat[affected] += delta

                if delta == 1:
                    newly_sat = (clause_sat[affected] == 1)
                    unsat_count -= int(newly_sat.sum())
                else:
                    newly_unsat = (clause_sat[affected] == 0)
                    unsat_count += int(newly_unsat.sum())

                # Update penalty with momentum: increase for unsat, decrease for sat
                unsat_mask = clause_sat == 0
                # Momentum: penalty follows a moving average of unsat status
                penalty_momentum[unsat_mask] = 0.7 * penalty_momentum[unsat_mask] + 0.3 * 1.0
                penalty_momentum[~unsat_mask] = 0.7 * penalty_momentum[~unsat_mask] + 0.3 * 0.0
                penalty[unsat_mask] = np.minimum(penalty[unsat_mask] * 1.2, 100.0)
                penalty[~unsat_mask] = np.maximum(penalty[~unsat_mask] * 0.95, 0.05)

                c = smooth_cost(x, clause_sat, penalty)
                if c < best_cost:
                    best_cost = c
                    best_x = x.copy()
                    best_n_true = int(x.sum())
            else:
                # Satisfied: try to remove a block
                if not true_blocks:
                    break
                bi = true_blocks[_RNG.randint(0, len(true_blocks) - 1)]
                affected = block_covers_np[bi]
                clause_sat[affected] -= 1
                failed = np.any(clause_sat[affected] == 0)
                if failed:
                    x[bi] = True
                    clause_sat[affected] += 1
                else:
                    x[bi] = False
                    true_blocks.remove(bi)
                    c = smooth_cost(x, clause_sat, penalty)
                    if c < best_cost:
                        best_cost = c
                        best_x = x.copy()
                        best_n_true = int(x.sum())

                # Try 2-block swap: remove one true block, add one false block
                if _RNG.random() < 0.4 and true_blocks:
                    bi_remove = true_blocks[_RNG.randint(0, len(true_blocks) - 1)]
                    false_blocks = np.where(~x)[0]
                    if len(false_blocks) > 0:
                        bi_add = false_blocks[_RNG.randint(0, len(false_blocks) - 1)]
                        aff_rm = block_covers_np[bi_remove]
                        aff_add = block_covers_np[bi_add]
                        clause_sat[aff_rm] -= 1
                        clause_sat[aff_add] += 1
                        if np.all(clause_sat >= 1):
                            x[bi_remove] = False
                            x[bi_add] = True
                            true_blocks.remove(bi_remove)
                            true_blocks.append(bi_add)
                            c = smooth_cost(x, clause_sat, penalty)
                            if c < best_cost:
                                best_cost = c
                                best_x = x.copy()
                                best_n_true = int(x.sum())
                        else:
                            clause_sat[aff_rm] += 1
                            clause_sat[aff_add] -= 1

        # Use best found
        x = best_x
        clause_sat = np.zeros(n_t, dtype=np.int32)
        for bi in range(n_k):
            if x[bi]:
                clause_sat[block_covers_np[bi]] += 1
        selected = set(int(bi) for bi in range(n_k) if x[bi])
        cover_count = clause_sat

    # ---------------- Final high-precision polish: systematic local search ----------------
    # Try all single-block removals and additions systematically
    def polish_pass(selected_set, cc, max_rounds=5):
        for rnd in range(max_rounds):
            if time.time() > deadline - 15.0:
                break
            # Try removing each block
            improved = False
            for bi in list(selected_set):
                if time.time() > deadline - 15.0:
                    return selected_set, cc
                cc[block_covers_np[bi]] -= 1
                if np.any(cc[block_covers_np[bi]] == 0):
                    cc[block_covers_np[bi]] += 1
                else:
                    selected_set.discard(bi)
                    improved = True
            if not improved:
                break

        # Try adding blocks that cover the most uncovered (if any)
        # This is mainly for recovery if something went wrong
        if np.any(cc == 0):
            uncovered = np.where(cc == 0)[0]
            # Greedily add blocks to cover uncovered
            for ti in uncovered:
                if time.time() > deadline - 15.0:
                    break
                best_bi = None
                best_gain = -1
                for bi in t_to_blocks[ti]:
                    if bi in selected_set:
                        continue
                    gain = np.sum(cc[block_covers_np[bi]] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_bi = bi
                if best_bi is not None:
                    selected_set.add(best_bi)
                    cc[block_covers_np[best_bi]] += 1
        return selected_set, cc

    selected, cover_count = polish_pass(selected, cover_count)

    # ---------------- MILP polish (if time permits) ----------------
    if time.time() < deadline - 45.0:
        try:
            A_rows, A_cols, A_data = [], [], []
            for ti in range(n_t):
                for bi in t_to_blocks[ti]:
                    A_rows.append(ti)
                    A_cols.append(bi)
                    A_data.append(1)
            A = csr_matrix((A_data, (A_rows, A_cols)), shape=(n_t, n_k))
            c_obj = np.ones(n_k)
            lb = np.ones(n_t)
            ub = np.full(n_t, np.inf)
            integrality = np.ones(n_k)
            bounds = Bounds(lb=np.zeros(n_k), ub=np.ones(n_k))
            constraints = LinearConstraint(A, lb, ub)
            remaining_for_milp = max(15, deadline - 45.0 - time.time())
            res = milp(c_obj, constraints=constraints, integrality=integrality,
                       bounds=bounds, options={"time_limit": remaining_for_milp})
            if res.status == 0 and res.x is not None:
                milp_sel = set(int(bi) for bi in range(n_k) if res.x[bi] > 0.5)
                if len(milp_sel) < len(selected):
                    cc = np.zeros(n_t, dtype=np.int32)
                    for bi in milp_sel:
                        cc[block_covers_np[bi]] += 1
                    if np.all(cc >= 1):
                        selected = milp_sel
                        selected, cover_count = reduce_blocks(selected, cc)
                        selected, cover_count = polish_pass(selected, cover_count)
        except Exception:
            pass

    # Final reduction
    if selected:
        cc = np.zeros(n_t, dtype=np.int32)
        for bi in selected:
            cc[block_covers_np[bi]] += 1
        selected, cover_count = reduce_blocks(selected, cc)

    # Build output
    blocks = [list(k_subs[bi]) for bi in sorted(selected)]

    # Final validation
    if blocks:
        cc_final = np.zeros(n_t, dtype=np.int32)
        for bi in selected:
            cc_final[block_covers_np[bi]] += 1
        if np.any(cc_final == 0):
            # Fallback: rebuild from best_greedy
            selected = set(best_greedy)
            cc_fb = np.zeros(n_t, dtype=np.int32)
            for bi in selected:
                cc_fb[block_covers_np[bi]] += 1
            selected, cc_fb = reduce_blocks(selected, cc_fb)
            blocks = [list(k_subs[bi]) for bi in sorted(selected)]

    return blocks
