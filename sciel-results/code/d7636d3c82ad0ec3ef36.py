import time
import itertools
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct(n):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "121321"
    if n == 4:
        return "12131214121312141213121"

    def build_classic(n_sym):
        """Build classic superpermutation for n_sym symbols."""
        if n_sym == 1:
            return [1]
        word = [1]
        for m in range(2, n_sym + 1):
            perms = []
            seen = set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new_word = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_overlap = min(len(new_word), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o > 0 and new_word[len(new_word) - o:] == block[:o]:
                        overlap = o
                        break
                    elif o == 0:
                        overlap = 0
                        break
                new_word.extend(block[overlap:])
            word = new_word
        return word

    # For n <= 5, use classic (fast enough and optimal/near-optimal)
    if n <= 5:
        word = build_classic(n)
        return "".join(map(str, word))

    # For n >= 6, use optimized construction
    base_word = build_classic(n - 1)

    # Extract the sequence of (n-1)-permutations from base_word
    base_perms = []
    seen = set()
    for i in range(len(base_word) - n + 2):
        window = tuple(base_word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            base_perms.append(window)

    num_perms = len(base_perms)
    m = n - 1  # length of each (n-1)-perm

    # Build overlap matrix using bitmask approach
    # For each perm, compute bitmask of all its suffixes (as integer keys)
    # and bitmask of all its prefixes
    # overlap[i][j] = max k such that suffix_k(perm_i) == prefix_k(perm_j)

    # Encode each (n-1)-perm as a tuple of ints
    # For overlap computation, use the fact that max overlap is at most m-1 = n-2

    # Fast approach: for each k from m-1 down to 1, build dict mapping suffix->list of indices
    # and prefix->list of indices, then for each pair check match

    # Actually, the fastest approach for 720 perms:
    # Precompute for each perm: all its suffixes (as tuples) and prefixes (as tuples)
    # Then for each k, create a dict: suffix_tuple -> set of indices having that suffix
    # And prefix_tuple -> set of indices having that prefix
    # For each k from high to low, for each suffix, find matching prefixes

    # Let's use numpy with a different strategy:
    # Convert perms to numpy array
    perms_arr = np.array(base_perms, dtype=np.int32)  # shape (num_perms, m)

    # overlap[i][j] = max k in 1..m-1 such that perms_arr[i][m-k:] == perms_arr[j][:k]
    # For k=0, overlap is 0

    # Build the overlap matrix
    overlap_mat = np.zeros((num_perms, num_perms), dtype=np.int8)

    # For each k from m-1 down to 1:
    #   For each i, suffix = perms_arr[i][m-k:]
    #   Find all j where perms_arr[j][:k] == suffix
    #   Set overlap_mat[i][j] = k (only if not already set, i.e., still 0)

    # Use hashing approach for speed
    # For each k, create a dict mapping prefix_tuple -> list of indices
    for k in range(m - 1, 0, -1):
        # Build prefix dictionary for this k
        prefix_dict = {}
        for j in range(num_perms):
            key = tuple(perms_arr[j][:k])
            if key not in prefix_dict:
                prefix_dict[key] = []
            prefix_dict[key].append(j)

        # For each i, look up its suffix in prefix_dict
        for i in range(num_perms):
            suffix_key = tuple(perms_arr[i][m - k:])
            if suffix_key in prefix_dict:
                for j in prefix_dict[suffix_key]:
                    if overlap_mat[i][j] == 0:
                        overlap_mat[i][j] = k

    t1 = time.time()
    print(f"Overlap matrix computed in {t1 - start_time:.2f}s", flush=True)

    # Now solve TSP-like problem: find ordering of all num_perms perms
    # maximizing total overlap sum
    # Total length = 2*m + 1 + sum_{j=1}^{num_perms-1} (n - overlap[order[j-1]][order[j]])
    # = 2*m + 1 + (num_perms-1)*n - sum(overlap)
    # Minimize length <=> maximize sum(overlap)

    # Use multiple random restarts with greedy + 2-opt + or-opt
    best_order = None
    best_total_overlap = -1

    def compute_path_overlap(order, overlap_mat, num_perms):
        """Compute total overlap for a path."""
        total = 0
        for i in range(num_perms - 1):
            total += overlap_mat[order[i]][order[i + 1]]
        return total

    def greedy_order(overlap_mat, num_perms, start=None, randomize=False):
        """Greedy nearest neighbor with optional randomization."""
        if start is None:
            start = random.randint(0, num_perms - 1)
        visited = [False] * num_perms
        visited[start] = True
        order = [start]
        current = start
        total_overlap = 0

        for _ in range(num_perms - 1):
            if randomize:
                # Pick from top candidates with some randomness
                candidates = [(overlap_mat[current][j], j) for j in range(num_perms) if not visited[j]]
                candidates.sort(reverse=True)
                # Pick from top 5 with weighted probability
                top_k = min(5, len(candidates))
                weights = [max(candidates[i][0] + 1, 1) for i in range(top_k)]
                total_w = sum(weights)
                r = random.random() * total_w
                cum = 0
                chosen_idx = 0
                for i in range(top_k):
                    cum += weights[i]
                    if r <= cum:
                        chosen_idx = i
                        break
                best_next = candidates[chosen_idx][1]
                best_ov = candidates[chosen_idx][0]
            else:
                best_next = -1
                best_ov = -1
                for j in range(num_perms):
                    if not visited[j] and overlap_mat[current][j] > best_ov:
                        best_ov = overlap_mat[current][j]
                        best_next = j
            if best_next == -1:
                break
            visited[best_next] = True
            order.append(best_next)
            total_overlap += best_ov
            current = best_next
        return order, total_overlap

    def two_opt(order, overlap_mat, num_perms, deadline, max_passes=10):
        """2-opt improvement."""
        order = order[:]
        current_overlap = compute_path_overlap(order, overlap_mat, num_perms)
        improved = True
        passes = 0

        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline - 10:
                break

            for i in range(num_perms - 2):
                if time.time() > deadline - 10:
                    return order, current_overlap

                for j in range(i + 2, num_perms):
                    if time.time() > deadline - 10:
                        return order, current_overlap

                    # Reverse segment [i+1, j]
                    # Old edges: (order[i]->order[i+1]) and (order[j]->order[j+1]) if j < num_perms-1
                    # New edges: (order[i]->order[j]) and (order[i+1]->order[j+1]) if j < num_perms-1

                    old_cost = overlap_mat[order[i]][order[i + 1]]
                    new_cost = overlap_mat[order[i]][order[j]]

                    if j + 1 < num_perms:
                        old_cost += overlap_mat[order[j]][order[j + 1]]
                        new_cost += overlap_mat[order[i + 1]][order[j + 1]]

                    if new_cost > old_cost:
                        # Reverse segment
                        order[i + 1:j + 1] = order[i + 1:j + 1][::-1]
                        current_overlap += (new_cost - old_cost)
                        improved = True
                        break
                if improved:
                    break

        return order, current_overlap

    def or_opt(order, overlap_mat, num_perms, deadline, max_passes=5):
        """Or-opt: relocate a segment of length 1, 2, or 3."""
        order = order[:]
        current_overlap = compute_path_overlap(order, overlap_mat, num_perms)
        improved = True
        passes = 0

        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline - 10:
                break

            for seg_len in [1, 2, 3]:
                for i in range(num_perms - seg_len):
                    if time.time() > deadline - 10:
                        return order, current_overlap

                    # Segment to move: order[i:i+seg_len]
                    # Remove it and try inserting at all other positions
                    seg = order[i:i + seg_len]

                    # Compute old overlap contribution
                    old_ov = 0
                    # Edge before segment (if exists)
                    if i > 0:
                        old_ov += overlap_mat[order[i - 1]][order[i]]
                    # Edge after segment (if exists)
                    if i + seg_len < num_perms:
                        old_ov += overlap_mat[order[i + seg_len - 1]][order[i + seg_len]]
                    # Internal edges of segment
                    for s in range(seg_len - 1):
                        old_ov += overlap_mat[seg[s]][seg[s + 1]]

                    # Try inserting at each position j (j != i, j != i+1, ..., j != i+seg_len)
                    # After removal, the list has num_perms - seg_len elements
                    # Insert position j means before element j in the shortened list
                    # But we need to be careful about indices

                    # Actually, let's just try all insertion positions in the original list
                    # excluding the segment's own positions
                    for j in range(num_perms + 1):
                        if j >= i and j <= i + seg_len:
                            continue

                        # Build new order with segment removed and reinserted at j
                        # This is O(n) per move, which is fine for 720 elements
                        new_order = order[:i] + order[i + seg_len:]
                        # Adjust j for the removal
                        if j > i:
                            j -= seg_len
                        new_order = new_order[:j] + seg + new_order[j:]

                        # Compute new overlap
                        new_ov = 0
                        # Edge before segment in new position
                        if j > 0:
                            new_ov += overlap_mat[new_order[j - 1]][seg[0]]
                        # Edge after segment in new position
                        if j + seg_len < num_perms:
                            new_ov += overlap_mat[seg[seg_len - 1]][new_order[j + seg_len]]
                        # Internal edges of segment (same as before)
                        for s in range(seg_len - 1):
                            new_ov += overlap_mat[seg[s]][seg[s + 1]]
                        # Edge connecting the gap where segment was removed
                        if i > 0 and i + seg_len < num_perms:
                            new_ov += overlap_mat[order[i - 1]][order[i + seg_len]]

                        # But we also need to account for the edges that were broken/created
                        # Let me recompute more carefully
                        # Old total overlap in affected region:
                        # New total overlap in affected region:
                        # The difference is:
                        # Old: (order[i-1]->order[i]) + (order[i+seg_len-1]->order[i+seg_len]) [if both exist]
                        # New: (new_order[j-1]->seg[0]) + (seg[seg_len-1]->new_order[j+seg_len]) [if both exist]
                        #      + (order[i-1]->order[i+seg_len]) [if the gap is closed]

                        # This is getting complex. Let me just compute the full path overlap
                        # for small segments - it's O(n) which is acceptable
                        new_full_ov = compute_path_overlap(new_order, overlap_mat, num_perms)

                        if new_full_ov > current_overlap:
                            order = new_order
                            current_overlap = new_full_ov
                            improved = True
                            break

                    if improved:
                        break
                if improved:
                    break

        return order, current_overlap

    # Try multiple strategies
    strategies = [
        ("greedy", lambda: greedy_order(overlap_mat, num_perms, randomize=False)),
        ("greedy_rand", lambda: greedy_order(overlap_mat, num_perms, randomize=True)),
        ("random", lambda: (list(range(num_perms)), 0)),  # Will be improved by 2-opt
    ]

    # Also try starting from the classic order (identity)
    classic_order = list(range(num_perms))
    strategies.append(("classic", lambda: (classic_order[:], compute_path_overlap(classic_order, overlap_mat, num_perms))))

    num_restarts = 8  # Total number of restarts across strategies

    for restart in range(num_restarts):
        if time.time() > deadline - 25:
            break

        # Choose strategy
        strat_name, strat_fn = strategies[restart % len(strategies)]

        if time.time() > deadline - 25:
            break

        order, init_overlap = strat_fn()

        # Apply 2-opt
        order, ov2 = two_opt(order, overlap_mat, num_perms, deadline, max_passes=5)

        # Apply or-opt
        if time.time() < deadline - 15:
            order, ov3 = or_opt(order, overlap_mat, num_perms, deadline, max_passes=2)
        else:
            ov3 = ov2

        if ov3 > best_total_overlap:
            best_total_overlap = ov3
            best_order = order[:]
            print(f"Restart {restart} ({strat_name}): overlap={ov3}, length={2*m+1+(num_perms-1)*n-ov3}", flush=True)

    if best_order is None:
        best_order = list(range(num_perms))
        best_total_overlap = compute_path_overlap(best_order, overlap_mat, num_perms)

    print(f"Best overlap: {best_total_overlap}, word length: {2*m+1+(num_perms-1)*n-best_total_overlap}", flush=True)

    # Build the final word
    word = []
    for idx, perm_idx in enumerate(best_order):
        p = list(base_perms[perm_idx])
        if idx == 0:
            word.extend(p)
            word.append(n)
            word.extend(p)
        else:
            prev_p = list(base_perms[best_order[idx - 1]])
            # Find max overlap using precomputed matrix
            overlap = overlap_mat[best_order[idx - 1]][perm_idx]
            word.append(n)
            word.extend(p[overlap:])

    # Verify it's a valid superpermutation
    if time.time() < deadline - 5:
        all_perms = set(itertools.permutations(range(1, n + 1)))
        found = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i + n])
            if len(set(window)) == n:
                found.add(window)
        if len(found) < math.factorial(n):
            print(f"Validation failed: found {len(found)} of {math.factorial(n)} perms, using classic", flush=True)
            word = build_classic(n)

    return "".join(map(str, word))

# EVOLVE-BLOCK-END
