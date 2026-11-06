import random
import time


def construct():
    start = time.time()
    deadline = start + 118.0  # safety margin before 150s kill

    d = 7
    n = 1 << d  # 128 vertices

    # Precompute all edges with indices
    all_edges = []
    edge_index = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(all_edges)
                all_edges.append((u, v))
                edge_index[(u, v)] = idx

    E = len(all_edges)  # 448

    # For each edge, precompute all C4s it participates in as bitmask integers.
    # A C4 is identified by its 4 edge indices; we store the other 3 as a mask.
    blocking_masks = [[] for _ in range(E)]
    for idx, (u, v) in enumerate(all_edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            bit_j = 1 << j
            u_j = u ^ bit_j
            v_j = v ^ bit_j
            e1 = (min(u, u_j), max(u, u_j))
            e2 = (min(v, v_j), max(v, v_j))
            e3 = (min(u_j, v_j), max(u_j, v_j))
            idx1 = edge_index[e1]
            idx2 = edge_index[e2]
            idx3 = edge_index[e3]
            mask = (1 << idx1) | (1 << idx2) | (1 << idx3)
            blocking_masks[idx].append(mask)

    def is_blocked(edge_idx, chosen_mask):
        """Check if adding edge_idx would complete any C4."""
        for mask in blocking_masks[edge_idx]:
            if (mask & chosen_mask) == mask:
                return True
        return False

    def greedy_fill(chosen_mask, order=None, rng=None):
        """Add any unblocked edges; if order given, only consider those in order."""
        added = 0
        if order is not None:
            for idx in order:
                if not (chosen_mask >> idx) & 1:
                    if not is_blocked(idx, chosen_mask):
                        chosen_mask |= (1 << idx)
                        added += 1
        else:
            for idx in range(E):
                if not (chosen_mask >> idx) & 1:
                    if not is_blocked(idx, chosen_mask):
                        chosen_mask |= (1 << idx)
                        added += 1
        return chosen_mask, added

    def greedy_random(rng):
        """Random greedy: shuffle edges, add if not blocked."""
        chosen_mask = 0
        order = list(range(E))
        rng.shuffle(order)
        for idx in order:
            if not is_blocked(idx, chosen_mask):
                chosen_mask |= (1 << idx)
        return chosen_mask

    def popcount(x):
        return bin(x).count("1")

    def simulated_annealing(chosen_mask, rng, time_limit):
        """
        SA over the chosen set: repeatedly try to add unblocked edges,
        and perform 1-for-1 swaps with occasional non-improving moves.
        """
        best_mask = chosen_mask
        best_size = popcount(chosen_mask)
        current_size = best_size

        # Temperature schedule
        t = 0.8
        t_min = 0.02
        cooling = 0.995

        while time.time() < time_limit:
            # Phase 1: greedy add any unblocked edges (deterministic scan)
            for idx in range(E):
                if not (chosen_mask >> idx) & 1:
                    if not is_blocked(idx, chosen_mask):
                        chosen_mask |= (1 << idx)
                        current_size += 1

            if current_size > best_size:
                best_size = current_size
                best_mask = chosen_mask

            if time.time() >= time_limit:
                break

            # Phase 2: try a swap
            # Build lists of chosen and unchosen edges (only occasionally to save time)
            if rng.random() < 0.5 or current_size == best_size:
                chosen_list = [i for i in range(E) if (chosen_mask >> i) & 1]
                unchosen_list = [i for i in range(E) if not ((chosen_mask >> i) & 1)]
                if not unchosen_list:
                    break
                rng.shuffle(chosen_list)
                rng.shuffle(unchosen_list)

                # Try a few swaps
                found_improve = False
                for _ in range(20):
                    if time.time() >= time_limit:
                        break
                    if not chosen_list or not unchosen_list:
                        break

                    rem = chosen_list[rng.randrange(len(chosen_list))]
                    add = unchosen_list[rng.randrange(len(unchosen_list))]

                    trial = chosen_mask
                    trial &= ~(1 << rem)
                    if not is_blocked(add, trial):
                        trial |= (1 << add)
                        # Greedy refill on the trial
                        trial, added = greedy_fill(trial, order=None, rng=None)
                        new_size = popcount(trial)
                        delta = new_size - current_size

                        if delta >= 0:
                            # Accept
                            chosen_mask = trial
                            current_size = new_size
                            if new_size > best_size:
                                best_size = new_size
                                best_mask = chosen_mask
                            found_improve = True
                            # Rebuild lists lazily
                            chosen_list = [i for i in range(E) if (chosen_mask >> i) & 1]
                            unchosen_list = [i for i in range(E) if not ((chosen_mask >> i) & 1)]
                            break
                        else:
                            # Metropolis accept
                            if rng.random() < pow(2.718281828, delta / max(t, 1e-9)):
                                chosen_mask = trial
                                current_size = new_size
                                chosen_list = [i for i in range(E) if (chosen_mask >> i) & 1]
                                unchosen_list = [i for i in range(E) if not ((chosen_mask >> i) & 1)]
                                break

                if found_improve:
                    t = max(t * cooling, t_min)
                    continue

            # Cool down
            t = max(t * cooling, t_min)

            # If stuck for a while, do a random perturbation
            if rng.random() < 0.05:
                # Remove a few random edges and refill
                chosen_list = [i for i in range(E) if (chosen_mask >> i) & 1]
                if len(chosen_list) > 10:
                    to_remove = rng.sample(chosen_list, min(3, len(chosen_list) // 10))
                    for r in to_remove:
                        chosen_mask &= ~(1 << r)
                        current_size -= 1
                    chosen_mask, _ = greedy_fill(chosen_mask)
                    current_size = popcount(chosen_mask)

        return best_mask, best_size

    def two_opt_search(chosen_mask, rng, time_limit):
        """
        Two-opt: remove two edges, add two different edges.
        """
        best_mask = chosen_mask
        best_size = popcount(chosen_mask)
        current_size = best_size

        while time.time() < time_limit:
            # Greedy fill
            for idx in range(E):
                if not (chosen_mask >> idx) & 1:
                    if not is_blocked(idx, chosen_mask):
                        chosen_mask |= (1 << idx)
                        current_size += 1

            if current_size > best_size:
                best_size = current_size
                best_mask = chosen_mask

            if time.time() >= time_limit:
                break

            unchosen_list = [i for i in range(E) if not ((chosen_mask >> i) & 1)]
            if len(unchosen_list) < 2:
                break
            rng.shuffle(unchosen_list)

            improved = False
            for e1_idx in unchosen_list[:80]:
                if time.time() >= time_limit:
                    break

                blocking1 = []
                for mask in blocking_masks[e1_idx]:
                    if (mask & chosen_mask) == mask:
                        blocking1.append(mask)

                if not blocking1:
                    chosen_mask |= (1 << e1_idx)
                    current_size += 1
                    improved = True
                    continue

                for e2_idx in unchosen_list[:80]:
                    if e2_idx == e1_idx:
                        continue
                    if time.time() >= time_limit:
                        break

                    blocking2 = []
                    for mask in blocking_masks[e2_idx]:
                        if (mask & chosen_mask) == mask:
                            blocking2.append(mask)

                    if not blocking2:
                        chosen_mask |= (1 << e2_idx)
                        current_size += 1
                        improved = True
                        continue

                    # Try removing one edge from blocking1 and one from blocking2
                    if len(blocking1) == 1 and len(blocking2) == 1:
                        t1 = blocking1[0]
                        t2 = blocking2[0]
                        # Get candidate removals
                        rem1_candidates = [i for i in range(E) if (t1 >> i) & 1]
                        rem2_candidates = [i for i in range(E) if (t2 >> i) & 1]
                        for rem1 in rem1_candidates:
                            for rem2 in rem2_candidates:
                                if rem1 == rem2:
                                    continue
                                trial = chosen_mask
                                trial &= ~(1 << rem1)
                                trial &= ~(1 << rem2)
                                # Check if both e1 and e2 are now unblocked
                                ok1 = True
                                for mask in blocking_masks[e1_idx]:
                                    if (mask & trial) == mask:
                                        ok1 = False
                                        break
                                ok2 = True
                                for mask in blocking_masks[e2_idx]:
                                    if (mask & trial) == mask:
                                        ok2 = False
                                        break
                                if ok1 and ok2:
                                    trial |= (1 << e1_idx)
                                    trial |= (1 << e2_idx)
                                    # Greedy refill
                                    trial, _ = greedy_fill(trial)
                                    if popcount(trial) > current_size:
                                        chosen_mask = trial
                                        current_size = popcount(trial)
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break

            if current_size > best_size:
                best_size = current_size
                best_mask = chosen_mask

            if not improved:
                # Random perturbation
                if time.time() < time_limit - 5:
                    chosen_list = [i for i in range(E) if (chosen_mask >> i) & 1]
                    if len(chosen_list) > 10:
                        to_remove = rng.sample(chosen_list, min(3, len(chosen_list) // 10))
                        for r in to_remove:
                            chosen_mask &= ~(1 << r)
                        chosen_mask, _ = greedy_fill(chosen_mask)
                        current_size = popcount(chosen_mask)

        return best_mask, best_size

    best_mask = 0
    best_size = 0
    rng = random.Random(42)

    # Multiple random greedy starts + SA + two-opt
    num_starts = 12
    for s in range(num_starts):
        if time.time() > deadline - 50:
            break

        chosen_mask = greedy_random(rng)
        size = popcount(chosen_mask)
        if size > best_size:
            best_mask = chosen_mask
            best_size = size

        # SA from this start
        time_limit = deadline - 40
        chosen_mask, size = simulated_annealing(chosen_mask, rng, time_limit)
        if size > best_size:
            best_mask = chosen_mask
            best_size = size

        # Two-opt search
        if time.time() < deadline - 30:
            chosen_mask, size = two_opt_search(chosen_mask, rng, deadline - 25)
            if size > best_size:
                best_mask = chosen_mask
                best_size = size

        if time.time() > deadline - 25:
            break

    # Final greedy fill on best
    if time.time() < deadline - 5:
        for idx in range(E):
            if not (best_mask >> idx) & 1:
                if not is_blocked(idx, best_mask):
                    best_mask |= (1 << idx)

    # Convert to list of [u, v]
    out = []
    for idx in range(E):
        if (best_mask >> idx) & 1:
            u, v = all_edges[idx]
            out.append([u, v])
    return out
