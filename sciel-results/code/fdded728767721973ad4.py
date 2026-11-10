import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute all t-subsets and map to index
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute all k-blocks and their covered t-subset masks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, list of t-subset indices it covers (for fast counting)
    block_subsets = []
    for b in all_blocks:
        subs = [t_sub_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)

    # Greedy initialization (fast, deterministic-ish with random tie-break)
    def greedy_init(seed=None):
        if seed is not None:
            random.seed(seed)
        uncovered_count = [0] * n_t_subsets  # not needed, use bitmask
        uncovered_mask = (1 << n_t_subsets) - 1
        selected = []
        selected_set = set()

        while uncovered_mask:
            best_idx = -1
            best_gain = 0
            for i in range(n_blocks):
                if i in selected_set:
                    continue
                gain = 0
                subs = block_subsets[i]
                # count how many of this block's t-subsets are still uncovered
                # Use bit operations on a small per-block mask? We didn't precompute masks.
                # For speed in init, just count via loop (init is fast enough)
                for s in subs:
                    if (uncovered_mask >> s) & 1:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                elif gain == best_gain and gain > 0:
                    if random.random() < 0.2:
                        best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            selected_set.add(best_idx)
            for s in block_subsets[best_idx]:
                uncovered_mask &= ~(1 << s)
            if time.time() > deadline:
                return None
        return selected

    def compute_energy(blocks):
        """Return (n_uncovered, len(blocks)) for a list of block indices."""
        if not blocks:
            return n_t_subsets, 0
        counts = [0] * n_t_subsets
        for b in blocks:
            for s in block_subsets[b]:
                counts[s] += 1
        n_uncov = sum(1 for c in counts if c == 0)
        return n_uncov, len(blocks)

    def simulated_annealing(initial_blocks, seed=None):
        if seed is not None:
            random.seed(seed)

        blocks = list(initial_blocks)
        if not blocks:
            return blocks

        block_set = set(blocks)
        # coverage counts per t-subset
        counts = [0] * n_t_subsets
        for b in blocks:
            for s in block_subsets[b]:
                counts[s] += 1
        n_uncov = sum(1 for c in counts if c == 0)
        energy = n_uncov + len(blocks)
        best_energy = energy
        best_blocks = list(blocks)

        T = 50.0
        T_min = 0.01
        cooling = 0.997

        while T > T_min and time.time() < deadline:
            move_type = random.random()
            if move_type < 0.35:
                # Remove a block
                if not blocks:
                    continue
                i = random.randrange(len(blocks))
                b = blocks[i]
                # delta energy: removing b
                # n_uncov change: for each s in block_subsets[b], if counts[s]==1 then becomes uncovered (+1 each)
                delta_uncov = 0
                for s in block_subsets[b]:
                    if counts[s] == 1:
                        delta_uncov += 1
                delta_len = -1
                delta_e = delta_uncov + delta_len
                if delta_e <= 0 or random.random() < np.exp(-delta_e / T):
                    # accept
                    for s in block_subsets[b]:
                        counts[s] -= 1
                    blocks.pop(i)
                    block_set.discard(b)
                    n_uncov += delta_uncov
                    energy += delta_e
                    if energy < best_energy:
                        best_energy = energy
                        best_blocks = list(blocks)
                T *= cooling

            elif move_type < 0.65:
                # Add a block
                # pick a random block not in set
                b = random.randrange(n_blocks)
                if b in block_set:
                    # try a few more
                    for _ in range(5):
                        b = random.randrange(n_blocks)
                        if b not in block_set:
                            break
                    else:
                        continue
                # delta energy: adding b
                delta_uncov = 0
                for s in block_subsets[b]:
                    if counts[s] == 0:
                        delta_uncov -= 1  # becomes covered
                delta_len = 1
                delta_e = delta_uncov + delta_len
                if delta_e <= 0 or random.random() < np.exp(-delta_e / T):
                    for s in block_subsets[b]:
                        counts[s] += 1
                    blocks.append(b)
                    block_set.add(b)
                    n_uncov += delta_uncov
                    energy += delta_e
                    if energy < best_energy:
                        best_energy = energy
                        best_blocks = list(blocks)
                T *= cooling

            else:
                # Swap: remove one, add one
                if not blocks:
                    continue
                i = random.randrange(len(blocks))
                b_out = blocks[i]
                b_in = random.randrange(n_blocks)
                if b_in in block_set and b_in != b_out:
                    for _ in range(5):
                        b_in = random.randrange(n_blocks)
                        if b_in not in block_set or b_in == b_out:
                            break
                    else:
                        continue
                # compute delta for removing b_out and adding b_in
                delta_uncov = 0
                # remove b_out first
                for s in block_subsets[b_out]:
                    if counts[s] == 1:
                        delta_uncov += 1  # becomes uncovered
                    else:
                        pass
                # add b_in
                for s in block_subsets[b_in]:
                    # after removing b_out, counts[s] is adjusted
                    # We need to simulate: counts' = counts - (1 if s in b_out) + (1 if s in b_in)
                    # For simplicity, compute net effect:
                    pass
                # Let's compute properly:
                # For each s, new_count = counts[s] - (1 if s in b_out else 0) + (1 if s in b_in else 0)
                # delta_uncov = (# s where old_count==0 and new_count>0) - (# s where old_count>0 and new_count==0)
                # But we need to be careful with overlaps.
                # Efficient way: iterate over union of subsets
                subs_out = set(block_subsets[b_out])
                subs_in = set(block_subsets[b_in])
                all_subs = subs_out | subs_in
                delta_uncov = 0
                for s in all_subs:
                    old = counts[s]
                    new = old - (1 if s in subs_out else 0) + (1 if s in subs_in else 0)
                    if old == 0 and new > 0:
                        delta_uncov -= 1
                    elif old > 0 and new == 0:
                        delta_uncov += 1
                delta_len = 0  # swap keeps same count
                delta_e = delta_uncov + delta_len
                if delta_e <= 0 or random.random() < np.exp(-delta_e / T):
                    # accept
                    for s in block_subsets[b_out]:
                        counts[s] -= 1
                    for s in block_subsets[b_in]:
                        counts[s] += 1
                    blocks[i] = b_in
                    block_set.discard(b_out)
                    block_set.add(b_in)
                    n_uncov += delta_uncov
                    energy += delta_e
                    if energy < best_energy:
                        best_energy = energy
                        best_blocks = list(blocks)
                T *= cooling

        return best_blocks

    best_blocks = None
    best_energy = float('inf')

    num_restarts = 6
    for i in range(num_restarts):
        if time.time() > deadline:
            break

        init = greedy_init(seed=i)
        if init is None:
            continue

        sa_blocks = simulated_annealing(init, seed=1000 + i)
        if sa_blocks is None:
            continue

        e_uncov, e_len = compute_energy(sa_blocks)
        e_total = e_uncov + e_len
        if e_total < best_energy:
            best_energy = e_total
            best_blocks = sa_blocks

        if best_energy <= 335:
            break

    if best_blocks is None:
        init = greedy_init(seed=None)
        if init:
            return [list(all_blocks[i]) for i in init]
        return []

    return [list(all_blocks[i]) for i in best_blocks]
