"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    n = 256

    def _compute_sums(elems):
        s = set()
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                s.add(elems[i] + elems[j])
        return s

    def _compute_diffs(elems):
        d = set()
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                diff = abs(elems[i] - elems[j])
                if diff > 0:
                    d.add(diff)
        return d

    def _is_valid_sidon(elems):
        sums = _compute_sums(elems)
        return len(sums) == len(elems) * (len(elems) + 1) // 2

    def _can_add_to_sidon(x, elems, sums):
        for a in elems:
            if a + x in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def _greedy_balanced(order, deadline, max_size_each=32):
        """
        Interleaved greedy: alternately add to A and B, tracking diffs.
        Returns (setA, setB).
        """
        elemsA = []
        sumsA = set()
        diffsA = set()
        elemsB = []
        sumsB = set()
        diffsB = set()
        all_diffs = set()  # union of diffsA and diffsB for cross-check

        for x in order:
            if time.time() > deadline:
                break
            if len(elemsA) >= max_size_each and len(elemsB) >= max_size_each:
                break

            # Try adding to A if it's smaller or both are small
            target = None
            if len(elemsA) < len(elemsB) or (len(elemsA) == len(elemsB) and random.random() < 0.5):
                target = 'A'
            else:
                target = 'B'

            # Try preferred target
            for tgt in [target, 'B' if target == 'A' else 'A']:
                if tgt == 'A':
                    elems_t = elemsA
                    sums_t = sumsA
                    diffs_t = diffsA
                    other_diffs = diffsB
                    max_sz = max_size_each
                else:
                    elems_t = elemsB
                    sums_t = sumsB
                    diffs_t = diffsB
                    other_diffs = diffsA
                    max_sz = max_size_each

                if len(elems_t) >= max_sz:
                    continue

                # Check Sidon
                if not _can_add_to_sidon(x, elems_t, sums_t):
                    continue

                # Check diff conflict with other set
                conflict = False
                for a in elems_t:
                    d = abs(a - x)
                    if d > 0 and d in other_diffs:
                        conflict = True
                        break
                if conflict:
                    continue

                # Add x
                for a in elems_t:
                    sums_t.add(a + x)
                    d = abs(a - x)
                    if d > 0:
                        diffs_t.add(d)
                sums_t.add(2 * x)
                elems_t.append(x)
                break

        return sorted(elemsA), sorted(elemsB)

    def _local_search_two_sets(setA, setB, deadline, max_iters=2000):
        """
        Efficient local search with incremental validation.
        """
        setA = sorted(list(setA))
        setB = sorted(list(setB))

        # Precompute state
        sumsA = _compute_sums(setA)
        sumsB = _compute_sums(setB)
        diffsA = _compute_diffs(setA)
        diffsB = _compute_diffs(setB)

        def _validate_full(A, B):
            if not A or not B:
                return False
            if not _is_valid_sidon(A):
                return False
            if not _is_valid_sidon(B):
                return False
            dA = _compute_diffs(A)
            dB = _compute_diffs(B)
            return len(dA & dB) == 0

        def _size(A, B):
            return len(A) + len(B)

        best_A, best_B = list(setA), list(setB)
        best_size = _size(best_A, best_B)

        # Build candidate pool: elements near existing ones or random
        all_elems = set(setA) | set(setB)
        candidates = list(range(1, n + 1))

        for it in range(max_iters):
            if time.time() > deadline:
                break

            move_type = random.random()

            if move_type < 0.35:
                # Add to A
                x = random.choice(candidates)
                if x in setA:
                    continue
                # Quick check
                if not _can_add_to_sidon(x, setA, sumsA):
                    continue
                conflict = False
                for a in setA:
                    d = abs(a - x)
                    if d > 0 and d in diffsB:
                        conflict = True
                        break
                if conflict:
                    continue
                # Accept
                for a in setA:
                    sumsA.add(a + x)
                    d = abs(a - x)
                    if d > 0:
                        diffsA.add(d)
                sumsA.add(2 * x)
                setA.append(x)
                setA.sort()
                if _size(setA, setB) > best_size:
                    best_A, best_B = list(setA), list(setB)
                    best_size = _size(best_A, best_B)

            elif move_type < 0.70:
                # Add to B
                x = random.choice(candidates)
                if x in setB:
                    continue
                if not _can_add_to_sidon(x, setB, sumsB):
                    continue
                conflict = False
                for a in setB:
                    d = abs(a - x)
                    if d > 0 and d in diffsA:
                        conflict = True
                        break
                if conflict:
                    continue
                for a in setB:
                    sumsB.add(a + x)
                    d = abs(a - x)
                    if d > 0:
                        diffsB.add(d)
                sumsB.add(2 * x)
                setB.append(x)
                setB.sort()
                if _size(setA, setB) > best_size:
                    best_A, best_B = list(setA), list(setB)
                    best_size = _size(best_A, best_B)

            elif move_type < 0.85:
                # Remove from A (frees diffs for B)
                if len(setA) <= 1:
                    continue
                idx = random.randint(0, len(setA) - 1)
                removed = setA[idx]
                # Rebuild sumsA and diffsA without removed
                new_setA = setA[:idx] + setA[idx+1:]
                new_sumsA = _compute_sums(new_setA)
                new_diffsA = _compute_diffs(new_setA)
                setA = new_setA
                sumsA = new_sumsA
                diffsA = new_diffsA

            else:
                # Remove from B
                if len(setB) <= 1:
                    continue
                idx = random.randint(0, len(setB) - 1)
                removed = setB[idx]
                new_setB = setB[:idx] + setB[idx+1:]
                new_sumsB = _compute_sums(new_setB)
                new_diffsB = _compute_diffs(new_setB)
                setB = new_setB
                sumsB = new_sumsB
                diffsB = new_diffsB

        # Final validation
        if _validate_full(setA, setB):
            if _size(setA, setB) > best_size:
                best_A, best_B = list(setA), list(setB)
                best_size = _size(best_A, best_B)

        return best_A, best_B

    # Strategy: multiple restarts with balanced greedy + local search
    best_sets = ([1], [2])
    best_size = 2

    rng = random.Random(42)

    for restart in range(500):
        if time.time() > deadline:
            break

        # Shuffled order
        order = list(range(1, n + 1))
        rng.shuffle(order)

        # Balanced greedy
        setA, setB = _greedy_balanced(order, deadline, max_size_each=32)

        if not setA or not setB:
            continue

        current_size = len(setA) + len(setB)

        if current_size > best_size:
            best_sets = (list(setA), list(setB))
            best_size = current_size

        # Local search on promising solutions
        if current_size >= best_size - 2:
            setA_ls, setB_ls = _local_search_two_sets(setA, setB, deadline, max_iters=800)
            ls_size = len(setA_ls) + len(setB_ls)
            if ls_size > best_size:
                best_sets = (setA_ls, setB_ls)
                best_size = ls_size

    return [best_sets[0], best_sets[1]]
# EVOLVE-BLOCK-END
