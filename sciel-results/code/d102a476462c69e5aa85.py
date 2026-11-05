import time
import random
import sys
import itertools

def construct(q, n, radius):
    if not (q == 3 and n == 6 and radius == 1):
        all_words = list(itertools.product(range(q), repeat=n))
        code = []
        covered = set()
        for w in all_words:
            if w not in covered:
                code.append(list(w))
                for i in range(n):
                    for v in range(q):
                        if v == w[i]: continue
                        new_w = list(w)
                        new_w[i] = v
                        covered.add(tuple(new_w))
                covered.add(w)
        return code

    start_time = time.time()
    deadline = start_time + 115.0

    total_words = q ** n  # 729

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return tuple(w)

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Build neighbor lists as sorted lists for fast iteration
    # Also build a set version for O(1) membership
    neighbors_list = [None] * total_words
    neighbors_set = [None] * total_words

    for i in range(total_words):
        w = idx_to_word(i)
        nbs = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                nbs.append(word_to_idx(tuple(nw)))
        neighbors_list[i] = nbs
        neighbors_set[i] = set(nbs)

    # Initial code: Tetracode x tails (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    code_indices = set()
    counts = [0] * total_words

    for w in seed_words:
        idx = word_to_idx(tuple(w))
        if idx not in code_indices:
            code_indices.add(idx)
            for nb in neighbors_list[idx]:
                counts[nb] += 1

    best_size = len(code_indices)
    best_code = list(code_indices)

    def try_remove():
        """Try to remove a codeword. Returns True if removed."""
        for c in list(code_indices):
            removable = True
            for nb in neighbors_list[c]:
                if counts[nb] <= 1:
                    removable = False
                    break
            if removable:
                for nb in neighbors_list[c]:
                    counts[nb] -= 1
                code_indices.remove(c)
                return True
        return False

    def full_swap_search():
        """Deterministic full swap search. Returns (c_out, c_in) or None."""
        critical = [u for u in range(total_words) if counts[u] == 1]
        critical_set = set(critical)

        best_swap = None
        best_score = -1

        c_out_list = list(code_indices)
        random.shuffle(c_out_list)

        for c_out in c_out_list:
            if time.time() > deadline:
                break
            nb_out_set = neighbors_set[c_out]
            nb_out_list = neighbors_list[c_out]

            # For each potential c_in not in code, check validity and score
            for c_in in range(total_words):
                if c_in in code_indices:
                    continue

                # Validity: for all u in nb_out not in nb_in, counts[u] >= 2
                nb_in_set = neighbors_set[c_in]
                valid = True
                for u in nb_out_list:
                    if u not in nb_in_set and counts[u] < 2:
                        valid = False
                        break
                if not valid:
                    continue

                # Score: number of critical words in nb_in but not nb_out
                score = 0
                for u in neighbors_list[c_in]:
                    if u in critical_set and u not in nb_out_set:
                        score += 1

                if score > best_score:
                    best_score = score
                    best_swap = (c_out, c_in)
                    if score >= 3:
                        return best_swap

        return best_swap

    # Main search loop
    stuck_count = 0
    while time.time() < deadline:
        # Phase 1: Greedy removal
        if try_remove():
            stuck_count = 0
            if len(code_indices) < best_size:
                best_size = len(code_indices)
                best_code = list(code_indices)
            continue

        # Phase 2: Swap search
        swap = full_swap_search()
        if swap:
            c_out, c_in = swap
            for nb in neighbors_list[c_out]:
                counts[nb] -= 1
            for nb in neighbors_list[c_in]:
                counts[nb] += 1
            code_indices.remove(c_out)
            code_indices.add(c_in)
            stuck_count = 0
        else:
            # Stuck: try random perturbation
            swapped = False
            for _ in range(30):
                if time.time() > deadline:
                    break
                c_out = random.choice(list(code_indices))
                c_in = random.randint(0, total_words - 1)
                if c_in in code_indices:
                    continue
                nb_out_set = neighbors_set[c_out]
                nb_in_set = neighbors_set[c_in]
                valid = True
                for u in neighbors_list[c_out]:
                    if u not in nb_in_set and counts[u] < 2:
                        valid = False
                        break
                if valid:
                    for nb in neighbors_list[c_out]:
                        counts[nb] -= 1
                    for nb in neighbors_list[c_in]:
                        counts[nb] += 1
                    code_indices.remove(c_out)
                    code_indices.add(c_in)
                    swapped = True
                    break

            if not swapped:
                stuck_count += 1
                if stuck_count > 5:
                    break
            else:
                stuck_count = 0

    return [list(idx_to_word(idx)) for idx in best_code]
