import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    N = 16
    vertices = range(N)
    all_triples = list(itertools.combinations(vertices, 3))
    num_triples = len(all_triples)  # 560
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute K(2,2,2) configurations
    k222_configs = []
    triple_to_k222 = [[] for _ in range(num_triples)]

    for six_vertices in itertools.combinations(vertices, 6):
        a, b, c, d, e, f = six_vertices
        pairs_list = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]

        for p1, p2, p3 in pairs_list:
            config_triples = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        idx = triple_to_idx[t]
                        config_triples.append(idx)

            config_id = len(k222_configs)
            k222_configs.append(config_triples)

            for t_idx in config_triples:
                triple_to_k222[t_idx].append(config_id)

    num_k222 = len(k222_configs)

    def time_check():
        return time.time() >= deadline

    def greedy_init(rng):
        present = [False] * num_triples
        k222_counts = [0] * num_k222
        count = 0

        triple_degree = [len(triple_to_k222[i]) for i in range(num_triples)]
        order = list(range(num_triples))
        rng.shuffle(order)
        order.sort(key=lambda i: triple_degree[i])

        for ti in order:
            ok = True
            for cid in triple_to_k222[ti]:
                if k222_counts[cid] >= 7:
                    ok = False
                    break
            if ok:
                present[ti] = True
                count += 1
                for cid in triple_to_k222[ti]:
                    k222_counts[cid] += 1

        return present, k222_counts, count

    def can_add_fast(ti, k222_counts):
        for cid in triple_to_k222[ti]:
            if k222_counts[cid] >= 7:
                return False
        return True

    def do_add(ti, present, k222_counts):
        present[ti] = True
        for cid in triple_to_k222[ti]:
            k222_counts[cid] += 1

    def do_remove(ti, present, k222_counts):
        present[ti] = False
        for cid in triple_to_k222[ti]:
            k222_counts[cid] -= 1

    def local_search(present, k222_counts, count, rng, max_steps=200000):
        steps = 0
        no_improve = 0
        max_no_improve = 8000
        fluctuate_every = 3000
        best_count_local = count
        best_present_local = present[:]

        while steps < max_no_improve and not time_check():
            if steps > 0 and steps % fluctuate_every == 0:
                num_remove = rng.randint(3, 7)
                present_indices = [i for i in range(num_triples) if present[i]]
                if len(present_indices) > num_remove:
                    to_remove = rng.sample(present_indices, num_remove)
                    for ti in to_remove:
                        do_remove(ti, present, k222_counts)
                        count -= 1
                no_improve = 0
                steps += 1
                continue

            # Try to add a triple
            added = False
            for _ in range(30):
                ti = rng.randint(0, num_triples - 1)
                if present[ti]:
                    continue
                if can_add_fast(ti, k222_counts):
                    do_add(ti, present, k222_counts)
                    count += 1
                    added = True
                    no_improve = 0
                    if count > best_count_local:
                        best_count_local = count
                        best_present_local = present[:]
                    break

            if added:
                steps += 1
                continue

            # Try swap: remove one, add another
            swapped = False
            for _ in range(50):
                ti_remove = rng.randint(0, num_triples - 1)
                if not present[ti_remove]:
                    continue

                do_remove(ti_remove, present, k222_counts)
                count -= 1

                found = False
                for _ in range(60):
                    ti_add = rng.randint(0, num_triples - 1)
                    if present[ti_add]:
                        continue
                    if can_add_fast(ti_add, k222_counts):
                        do_add(ti_add, present, k222_counts)
                        count += 1
                        found = True
                        break

                if found:
                    swapped = True
                    no_improve = 0
                    if count > best_count_local:
                        best_count_local = count
                        best_present_local = present[:]
                else:
                    do_add(ti_remove, present, k222_counts)
                    count += 1

            if swapped:
                steps += 1
                continue

            no_improve += 1
            steps += 1

        return best_present_local, best_count_local

    best_global_count = 0
    best_global_result = []

    rng = random.Random(42)
    restart = 0

    while not time_check() and restart < 500:
        if time.time() > deadline - 3:
            break

        present, k222_counts, count = greedy_init(rng)

        if count > best_global_count:
            best_global_count = count
            best_global_result = [list(all_triples[i]) for i in range(num_triples) if present[i]]

        best_present, best_count = local_search(present, k222_counts, count, rng)

        if best_count > best_global_count:
            best_global_count = best_count
            best_global_result = [list(all_triples[i]) for i in range(num_triples) if best_present[i]]

        restart += 1

    if not best_global_result:
        best_global_result = [[0, 1, 2]]

    return best_global_result

if __name__ == "__main__":
    res = construct()
    print(f"Number of triples: {len(res)}")
