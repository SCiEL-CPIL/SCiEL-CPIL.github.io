<<<<<<< SEARCH
_DEADLINE = 145.0  # safety margin vs 150s wall-clock kill
=======
_DEADLINE = 145.0  # safety margin vs 150s wall-clock kill
import random
>>>>>>> REPLACE
<<<<<<< SEARCH
    # ---- Local search improvement (time-bounded) ----
    # Use coverage-count array: covcnt[w] = number of chosen codewords whose ball contains w.
    # A codeword i is removable iff for every w in ball[i], covcnt[w] >= 2.
    # A swap (remove i, add j) is valid iff for every w in ball[i] \ ball[j], covcnt[w] >= 2.
    covcnt = np.zeros(total, dtype=np.int32)
    for i in chosen:
        covcnt[balls[i]] += 1

    def try_remove(i):
        """Check if codeword i can be removed (all its ball words still covered)."""
        b = balls[i]
        return (covcnt[b] >= 2).all()

    def do_remove(i):
        b = balls[i]
        covcnt[b] -= 1
        in_code[i] = False
        chosen.remove(i)

    def do_add(j):
        b = balls[j]
        covcnt[b] += 1
        in_code[j] = True
        chosen.append(j)

    def try_swap(i, j):
        """Check if we can remove i and add j (i.e., all words in ball[i] not in ball[j] still covered by others)."""
        b_i = balls[i]
        b_j = balls[j]
        # Words in ball[i] that are NOT in ball[j] must have covcnt >= 2
        # Build a mask: for each w in b_i, is w in b_j?
        in_j = np.zeros(total, dtype=bool)
        in_j[b_j] = True
        critical = b_i[~in_j[b_i]]
        if critical.size == 0:
            return True
        return (covcnt[critical] >= 2).all()

    while time_left() > 2.0:
        improved = False
        # Pass 1: redundancy removal
        for idx in range(len(chosen)):
            if time_left() < 2.0:
                break
            i = chosen[idx]
            if try_remove(i):
                do_remove(i)
                improved = True
                # After removal, indices shifted; restart scan
                break
        if improved:
            continue  # restart after a removal to catch cascades

        # Pass 2: 1-for-1 swaps
        if time_left() > 3.0:
            # For each chosen i, find if there's a j not in code such that swap(i,j) is valid
            # and j covers more "critical" words than i does (net improvement is just size -1)
            # Actually any valid swap reduces size by 1, so we just need ANY valid swap.
            # To be efficient, try random j's first, then scan.
            import random
            n_chosen = len(chosen)
            for idx in range(n_chosen):
                if time_left() < 3.0:
                    break
                i = chosen[idx]
                b_i = balls[i]
                # Try a few random candidates
                found = False
                for _ in range(20):
                    j = random.randint(0, total - 1)
                    if in_code[j]:
                        continue
                    if try_swap(i, j):
                        do_remove(i)
                        do_add(j)
                        improved = True
                        found = True
                        break
                if found:
                    break
                # If no random hit, do a limited scan (expensive, so only for small lists)
                if not found and n_chosen < 500:
                    # Scan all j not in code
                    for j in range(total):
                        if time_left() < 3.0:
                            break
                        if in_code[j]:
                            continue
                        if try_swap(i, j):
                            do_remove(i)
                            do_add(j)
                            improved = True
                            found = True
                            break
                if found:
                    break
        if not improved:
            break
=======
    # ---- Local search improvement (time-bounded) ----
    # Use coverage-count array: covcnt[w] = number of chosen codewords whose ball contains w.
    # A codeword i is removable iff for every w in ball[i], covcnt[w] >= 2.
    # A swap (remove i, add j) is valid iff for every w in ball[i] \ ball[j], covcnt[w] >= 2.
    covcnt = np.zeros(total, dtype=np.int32)
    for i in chosen:
        covcnt[balls[i]] += 1

    def try_remove(i):
        """Check if codeword i can be removed (all its ball words still covered)."""
        b = balls[i]
        return (covcnt[b] >= 2).all()

    def do_remove(i):
        b = balls[i]
        covcnt[b] -= 1
        in_code[i] = False
        chosen.remove(i)

    def do_add(j):
        b = balls[j]
        covcnt[b] += 1
        in_code[j] = True
        chosen.append(j)

    def try_swap(i, j):
        """Check if we can remove i and add j (i.e., all words in ball[i] not in ball[j] still covered by others)."""
        b_i = balls[i]
        b_j = balls[j]
        # Words in ball[i] that are NOT in ball[j] must have covcnt >= 2
        # Build a mask: for each w in b_i, is w in b_j?
        in_j = np.zeros(total, dtype=bool)
        in_j[b_j] = True
        critical = b_i[~in_j[b_i]]
        if critical.size == 0:
            return True
        return (covcnt[critical] >= 2).all()

    def greedy_remove():
        """Remove any redundant codeword. Returns True if removed one."""
        for idx in range(len(chosen)):
            i = chosen[idx]
            if try_remove(i):
                do_remove(i)
                return True
        return False

    def greedy_swap():
        """Find and execute one valid 1-for-1 swap. Returns True if swapped."""
        n_chosen = len(chosen)
        # Try random candidates first for speed
        for idx in range(n_chosen):
            i = chosen[idx]
            for _ in range(30):
                j = random.randint(0, total - 1)
                if in_code[j]:
                    continue
                if try_swap(i, j):
                    do_remove(i)
                    do_add(j)
                    return True
        # If random didn't find, do a limited systematic scan
        if n_chosen < 300:
            for idx in range(n_chosen):
                i = chosen[idx]
                for j in range(total):
                    if in_code[j]:
                        continue
                    if try_swap(i, j):
                        do_remove(i)
                        do_add(j)
                        return True
        return False

    # Main local search loop with random restarts to escape local minima
    best_size = len(chosen)
    best_code = chosen.copy()
    best_covcnt = covcnt.copy()

    while time_left() > 3.0:
        # Phase 1: Greedy removal of redundant codewords
        while time_left() > 3.0 and greedy_remove():
            pass

        # Phase 2: Greedy swaps
        if time_left() > 4.0:
            swap_count = 0
            while time_left() > 4.0 and swap_count < 20:
                if greedy_swap():
                    swap_count += 1
                else:
                    break

        # Track best solution found
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_code = chosen.copy()
            best_covcnt = covcnt.copy()

        # Phase 3: Random perturbation to escape local minima
        if time_left() > 5.0:
            # Randomly remove 1-3 non-redundant codewords and re-add greedily
            n_perturb = random.randint(1, 3)
            removed = []
            for _ in range(n_perturb):
                if len(chosen) <= 1:
                    break
                idx = random.randint(0, len(chosen) - 1)
                i = chosen[idx]
                do_remove(i)
                removed.append(i)
            
            # Re-cover greedily
            for i in removed:
                do_add(i)
            
            # Then try to remove any newly redundant ones
            while time_left() > 3.0 and greedy_remove():
                pass

            # Try some more swaps after perturbation
            if time_left() > 4.0:
                for _ in range(10):
                    if not greedy_swap():
                        break

            if len(chosen) < best_size:
                best_size = len(chosen)
                best_code = chosen.copy()
                best_covcnt = covcnt.copy()

    # Restore best solution found
    if best_size < len(chosen):
        chosen = best_code
        covcnt = best_covcnt
        in_code = np.zeros(total, dtype=bool)
        for i in chosen:
            in_code[i] = True
>>>>>>> REPLACE

