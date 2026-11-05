# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    N = 192
    ALL = list(range(1, N + 1))

    def time_left():
        return deadline - time.time()

    def is_valid(s):
        """Full validity check."""
        lst = sorted(s)
        n = len(lst)
        if n < 4:
            return True
        for idx_a, a in enumerate(lst):
            others = [x for i, x in enumerate(lst) if i != idx_a]
            m = len(others)
            if m < 3:
                continue
            # Quick range check
            os = sorted(others)
            min_sum3 = os[0] + os[1] + os[2]
            max_sum3 = os[-1] + os[-2] + os[-3]
            first_mult = ((min_sum3 + a - 1) // a) * a
            if first_mult > max_sum3:
                continue
            # Check all triples
            for i in range(m):
                oi = others[i]
                for j in range(i + 1, m):
                    s2 = oi + others[j]
                    for k in range(j + 1, m):
                        if (s2 + others[k]) % a == 0:
                            return False
        return True

    def is_valid_incremental(s, x):
        """Check if adding x to set s keeps it valid. s is a set."""
        if x in s:
            return True
        # Check x: any 3 from s sum to multiple of x?
        s_list = sorted(s)
        m = len(s_list)
        if m >= 3:
            min_sum3 = s_list[0] + s_list[1] + s_list[2]
            max_sum3 = s_list[-1] + s_list[-2] + s_list[-3]
            first_mult = ((min_sum3 + x - 1) // x) * x
            if first_mult <= max_sum3:
                # Check all triples from s
                for i in range(m):
                    si = s_list[i]
                    for j in range(i + 1, m):
                        s2 = si + s_list[j]
                        for k in range(j + 1, m):
                            if (s2 + s_list[k]) % x == 0:
                                return False
        # Check each a in s: any triple from (s - {a}) U {x} sums to multiple of a?
        # Only need to check triples that include x (triples entirely in s already valid)
        for a in s_list:
            # Triples: x + y + z where y,z in s, y != a, z != a, y != z
            # Also: triples entirely in s\{a} are already valid, skip
            # So check: x + y + z for y,z in s\{a}, y != z
            # And: y + z + w entirely in s\{a} already valid
            # So only new triples contain x
            others_no_a = [y for y in s_list if y != a]
            m2 = len(others_no_a)
            if m2 < 2:
                continue
            for i in range(m2):
                yi = others_no_a[i]
                for j in range(i + 1, m2):
                    s_sum = x + yi + others_no_a[j]
                    if s_sum % a == 0:
                        return False
        return True

    def greedy_build(order, s_init=None):
        """Build a set by adding elements in given order if valid."""
        s = set(s_init) if s_init is not None else set()
        for x in order:
            if x in s:
                continue
            if is_valid_incremental(s, x):
                s.add(x)
        return s

    def local_search(s, rng, max_iters=150):
        """Local search with add, remove1-add2, remove2-add2 moves."""
        s = set(s)
        cur_size = len(s)
        no_improve = 0

        for _ in range(max_iters):
            if time_left() < 3:
                break
            improved = False

            # Try additions
            candidates = [x for x in ALL if x not in s]
            rng.shuffle(candidates)
            for x in candidates[:20]:
                if is_valid_incremental(s, x):
                    s.add(x)
                    cur_size += 1
                    improved = True
                    break

            if improved:
                no_improve = 0
                continue

            no_improve += 1

            # Try remove1-add2: remove one, add two
            elems = list(s)
            rng.shuffle(elems)
            done = False
            for rem in elems[:8]:
                if done:
                    break
                candidates = [x for x in ALL if x not in s]
                rng.shuffle(candidates)
                for add1 in candidates[:15]:
                    if done:
                        break
                    for add2 in candidates[:15]:
                        if add1 == add2:
                            continue
                        new_s = (s - {rem}) | {add1, add2}
                        if len(new_s) > cur_size and is_valid(new_s):
                            s = new_s
                            cur_size = len(s)
                            improved = True
                            done = True
                            break

            if improved:
                no_improve = 0
                continue

            # Try remove2-add2
            if no_improve > 5:
                elems = list(s)
                rng.shuffle(elems)
                done = False
                for i in range(min(6, len(elems))):
                    if done:
                        break
                    for j in range(i + 1, min(len(elems), i + 6)):
                        if done:
                            break
                        rem1, rem2 = elems[i], elems[j]
                        candidates = [x for x in ALL if x not in s]
                        rng.shuffle(candidates)
                        for add1 in candidates[:12]:
                            if done:
                                break
                            for add2 in candidates[:12]:
                                if add1 == add2:
                                    continue
                                new_s = (s - {rem1, rem2}) | {add1, add2}
                                if len(new_s) > cur_size and is_valid(new_s):
                                    s = new_s
                                    cur_size = len(s)
                                    improved = True
                                    done = True
                                    break

                if improved:
                    no_improve = 0
                    continue

            if no_improve > 30:
                break

        return s

    rng = random.Random(42)

    # Phase 1: Diverse greedy starts
    starts = []

    # Start with large numbers (safest)
    order_large = list(range(N, 0, -1))
    starts.append(greedy_build(order_large))

    # Start with large, then random
    order_mixed = list(range(N, 100, -1)) + list(range(100, 0, -1))
    rng.shuffle(order_mixed[20:])
    starts.append(greedy_build(order_mixed))

    # Pure random orders
    for _ in range(8):
        order = ALL[:]
        rng.shuffle(order)
        starts.append(greedy_build(order))

    # Biased: prefer large
    for _ in range(5):
        order = ALL[:]
        order.sort(key=lambda x: -x + rng.uniform(0, 50))
        starts.append(greedy_build(order))

    # Start from a large core then add small
    core = set(range(150, N + 1))
    # Filter core to valid
    valid_core = set()
    for x in sorted(core, reverse=True):
        if is_valid_incremental(valid_core, x):
            valid_core.add(x)
    order_rest = list(range(1, 150))
    rng.shuffle(order_rest)
    starts.append(greedy_build(order_rest, s_init=valid_core))

    best = max(starts, key=len)
    best_size = len(best)
    print(f"Initial best size: {best_size}", flush=True)

    # Phase 2: Local search from each start
    for i, start_s in enumerate(starts):
        if time_left() < 5:
            break
        result = local_search(start_s, rng, max_iters=100)
        if len(result) > best_size:
            best_size = len(result)
            best = result
            print(f"Improved to {best_size} at start {i}", flush=True)

    # Phase 3: Aggressive search with restarts
    for iteration in range(300):
        if time_left() < 8:
            break

        # Different restart strategies
        r = iteration % 6
        if r == 0:
            # Perturb best
            s = set(best)
            n_remove = rng.randint(3, 6)
            to_remove = rng.sample(list(s), min(n_remove, len(s)))
            s -= set(to_remove)
            order = [x for x in ALL if x not in s]
            rng.shuffle(order)
            start_s = greedy_build(order, s_init=s)
        elif r == 1:
            # Large-first with noise
            order = list(range(N, 0, -1))
            for i in range(0, len(order) - 1, 2):
                if rng.random() < 0.1:
                    order[i], order[i+1] = order[i+1], order[i]
            start_s = greedy_build(order)
        elif r == 2:
            # Random
            order = ALL[:]
            rng.shuffle(order)
            start_s = greedy_build(order)
        elif r == 3:
            # Large core + random small
            core = set(range(140, N + 1))
            valid_core = set()
            for x in sorted(core, reverse=True):
                if is_valid_incremental(valid_core, x):
                    valid_core.add(x)
            order_rest = list(range(1, 140))
            rng.shuffle(order_rest)
            start_s = greedy_build(order_rest, s_init=valid_core)
        elif r == 4:
            # Biased random
            order = ALL[:]
            order.sort(key=lambda x: -x + rng.uniform(0, 80))
            start_s = greedy_build(order)
        else:
            # Random subset seed
            k = rng.randint(35, 50)
            seed_set = set(rng.sample(ALL, k))
            valid_seed = set()
            for x in sorted(seed_set, reverse=True):
                if is_valid_incremental(valid_seed, x):
                    valid_seed.add(x)
            order_rest = [x for x in ALL if x not in valid_seed]
            rng.shuffle(order_rest)
            start_s = greedy_build(order_rest, s_init=valid_seed)

        result = local_search(start_s, rng, max_iters=80)
        if len(result) > best_size:
            best_size = len(result)
            best = result
            print(f"New best: {best_size} at iteration {iteration}", flush=True)

    # Final validation
    if not is_valid(best):
        # Fallback
        best = greedy_build(list(range(N, 0, -1)))
        while not is_valid(best) and len(best) > 0:
            best = set(list(best)[:-1])

    return list(best)

# EVOLVE-BLOCK-END
