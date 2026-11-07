import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8s safety margin for 90s limit

    # Precompute for each x: all (u, v, w) triples where u=x+b, v=x+c, w=b+c,
    # b<c, u<=128, v<=128, and (w==x or w<=128). If u,v in S and (w in S or w==x), invalid.
    triples = [[] for _ in range(129)]
    for x in range(1, 129):
        for b in range(1, 129):
            for c in range(b + 1, 129):
                u = x + b
                v = x + c
                if u > 128 or v > 128:
                    continue
                w = b + c
                if w == x or w <= 128:
                    triples[x].append((u, v, w))

    # Precompute sum pairs: for each x, list of (a, b) with a+b==x, a<b
    sum_pairs = [[] for _ in range(129)]
    for x in range(1, 129):
        for a in range(1, x):
            b = x - a
            if b > a:
                sum_pairs[x].append((a, b))

    def can_add(S, x):
        """Check if S U {x} is valid."""
        # Case 1: x is a member
        for (u, v, w) in triples[x]:
            if u in S and v in S:
                if w in S or w == x:
                    return False

        # Case 2: x is a sum (a+b=x)
        for (a, b) in sum_pairs[x]:
            if a in S and b in S:
                for c in S:
                    if c == a or c == b:
                        continue
                    if (a + c) in S and (b + c) in S:
                        return False

        return True

    def is_valid(S):
        """Full validity check."""
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                if (a + b) not in S:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = S_list[k]
                    if (a + c) in S and (b + c) in S:
                        return False
        return True

    best_set = set()
    best_size = 0

    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    def greedy_construct(order):
        """Greedy construction given a fixed order of candidates."""
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.add(x)
        return S

    def local_search(S, max_iters=10000):
        """Improve S by trying additions and swaps."""
        S = set(S)
        not_in_S = set(range(1, 129)) - S
        iters = 0
        while iters < max_iters and time.time() < deadline:
            iters += 1
            progress = False

            # Try to add any element
            for x in sorted(not_in_S):
                if time.time() > deadline:
                    break
                if can_add(S, x):
                    S.add(x)
                    not_in_S.discard(x)
                    progress = True
                    update_best(S)
                    break

            if progress:
                continue

            # Try swaps: remove y from S, add x from not_in_S
            if not_in_S:
                x_candidates = list(not_in_S)
                random.shuffle(x_candidates)
                found_swap = False
                for x in x_candidates[:30]:
                    if time.time() > deadline:
                        break
                    S_list = list(S)
                    random.shuffle(S_list)
                    for y in S_list[:30]:
                        if time.time() > deadline:
                            break
                        S_temp = S - {y}
                        if can_add(S_temp, x):
                            S = S_temp
                            S.add(x)
                            not_in_S.discard(x)
                            not_in_S.add(y)
                            progress = True
                            found_swap = True
                            update_best(S)
                            break
                    if found_swap:
                        break

            if not progress:
                if not_in_S and S:
                    x_candidates = list(not_in_S)
                    random.shuffle(x_candidates)
                    found_swap = False
                    for x in x_candidates[:50]:
                        if time.time() > deadline:
                            break
                        S_list = list(S)
                        random.shuffle(S_list)
                        for y in S_list[:50]:
                            if time.time() > deadline:
                                break
                            S_temp = S - {y}
                            if can_add(S_temp, x):
                                S = S_temp
                                S.add(x)
                                not_in_S.discard(x)
                                not_in_S.add(y)
                                progress = True
                                found_swap = True
                                update_best(S)
                                break
                        if found_swap:
                            break
                if not progress:
                    break
        return S

    random.seed(42)

    # Phase 1: Multiple greedy constructions
    S = greedy_construct(list(range(1, 129)))
    update_best(S)

    S = greedy_construct(list(range(128, 0, -1)))
    update_best(S)

    for i in range(200):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S = greedy_construct(candidates)
        update_best(S)

    # Phase 2: Local search from best set
    if time.time() < deadline:
        S = local_search(best_set, max_iters=20000)
        update_best(S)

    # Phase 3: More random restarts with local search
    for i in range(100):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S = greedy_construct(candidates)
        S = local_search(S, max_iters=5000)
        update_best(S)

    # Phase 4: Final local search from best
    if time.time() < deadline:
        S = local_search(best_set, max_iters=20000)
        update_best(S)

    # Final: try to add any remaining elements in multiple passes
    if time.time() < deadline:
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, 129):
                if x in best_set:
                    continue
                if can_add(best_set, x):
                    best_set.add(x)
                    changed = True

    # Verify
    # if not is_valid(best_set):
    #     print("WARNING: invalid set found!")

    result = [int(x) for x in sorted(best_set)]
    return result
