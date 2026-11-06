import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for 120s limit

    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)

    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute K(2,2,2) configurations
    k222_list = []
    triple_to_k222s = [[] for _ in range(num_triples)]

    for points in combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = points
        pairings = [
            ((p0,p1),(p2,p3),(p4,p5)),
            ((p0,p1),(p2,p4),(p3,p5)),
            ((p0,p1),(p2,p5),(p3,p4)),
            ((p0,p2),(p1,p3),(p4,p5)),
            ((p0,p2),(p1,p4),(p3,p5)),
            ((p0,p2),(p1,p5),(p3,p4)),
            ((p0,p3),(p1,p2),(p4,p5)),
            ((p0,p3),(p1,p4),(p2,p5)),
            ((p0,p3),(p1,p5),(p2,p4)),
            ((p0,p4),(p1,p2),(p3,p5)),
            ((p0,p4),(p1,p3),(p2,p5)),
            ((p0,p4),(p1,p5),(p2,p3)),
            ((p0,p5),(p1,p2),(p3,p4)),
            ((p0,p5),(p1,p3),(p2,p4)),
            ((p0,p5),(p1,p4),(p2,p3)),
        ]

        for pair1, pair2, pair3 in pairings:
            transversals = []
            for x in pair1:
                for y in pair2:
                    for z in pair3:
                        transversals.append((x, y, z))

            trans_sorted = [tuple(sorted(t)) for t in transversals]
            trans_indices = [triple_to_idx[t] for t in trans_sorted]

            k_idx = len(k222_list)
            k222_list.append(set(trans_indices))

            for idx in trans_indices:
                triple_to_k222s[idx].append(k_idx)

    num_k222s = len(k222_list)

    def time_left():
        return deadline - time.time()

    def is_valid_add(t_idx, counts):
        for k_idx in triple_to_k222s[t_idx]:
            if counts[k_idx] == 7:
                return False
        return True

    def greedy_init(seed=None):
        if seed is not None:
            random.seed(seed)
        active = [False] * num_triples
        counts = [0] * num_k222s
        current_size = 0
        order = list(range(num_triples))
        random.shuffle(order)

        for t_idx in order:
            if time_left() < 5:
                break
            if is_valid_add(t_idx, counts):
                active[t_idx] = True
                current_size += 1
                for k_idx in triple_to_k222s[t_idx]:
                    counts[k_idx] += 1
        return active, counts, current_size

    best_size = 0
    best_active = None

    # Local search with swap moves (remove one, add one or more)
    def local_search(active, counts, current_size, max_time=5.0):
        search_start = time.time()
        no_improve_count = 0
        max_no_improve = 100

        while time_left() > 5 and (time.time() - search_start) < max_time and no_improve_count < max_no_improve:
            improved = False
            # Try removing a random active triple and then greedily adding back
            t_remove = random.randint(0, num_triples - 1)
            if not active[t_remove]:
                continue

            # Remove
            active[t_remove] = False
            for k_idx in triple_to_k222s[t_remove]:
                counts[k_idx] -= 1
            size_before = current_size - 1

            # Try to add multiple triples after removal
            add_candidates = list(range(num_triples))
            random.shuffle(add_candidates)
            for t_add in add_candidates:
                if active[t_add]:
                    continue
                if is_valid_add(t_add, counts):
                    active[t_add] = True
                    current_size += 1
                    for k_idx in triple_to_k222s[t_add]:
                        counts[k_idx] += 1
                    improved = True

            if current_size > size_before:
                # Net gain, keep going
                no_improve_count = 0
                if current_size > best_size:
                    best_size = current_size
                    best_active = active[:]
            else:
                # No net gain, revert the removal (put it back)
                if is_valid_add(t_remove, counts):
                    active[t_remove] = True
                    current_size += 1
                    for k_idx in triple_to_k222s[t_remove]:
                        counts[k_idx] += 1
                no_improve_count += 1

        return active, counts, current_size

    # Weighted greedy: prioritize triples that block fewer configurations
    def weighted_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        active = [False] * num_triples
        counts = [0] * num_k222s
        current_size = 0

        # Compute weight for each triple: number of K(2,2,2) configs it's in
        weights = [len(triple_to_k222s[i]) for i in range(num_triples)]

        # Sort by weight (ascending) with some randomization
        order = list(range(num_triples))
        # Use a stable-ish sort with random tie-breaking
        for i in range(len(order) - 1, 0, -1):
            # Partial sort with randomization
            j = random.randint(0, i)
            if random.random() < 0.3:  # Sometimes randomize to avoid local optima
                order[i], order[j] = order[j], order[i]

        for t_idx in order:
            if time_left() < 5:
                break
            if is_valid_add(t_idx, counts):
                active[t_idx] = True
                current_size += 1
                for k_idx in triple_to_k222s[t_idx]:
                    counts[k_idx] += 1
        return active, counts, current_size

    # Run multiple strategies with different seeds
    num_restarts = 30
    for r in range(num_restarts):
        if time_left() < 10:
            break

        if r % 3 == 0:
            active, counts, current_size = greedy_init(seed=r * 1000 + 42)
        elif r % 3 == 1:
            active, counts, current_size = weighted_greedy(seed=r * 1000 + 43)
        else:
            active, counts, current_size = greedy_init(seed=r * 1000 + 44)

        if current_size > best_size:
            best_size = current_size
            best_active = active[:]

        # Apply local search
        if time_left() > 8:
            active, counts, current_size = local_search(active, counts, current_size, max_time=3.0)

        if current_size > best_size:
            best_size = current_size
            best_active = active[:]

    # Final Polish: Greedy Add on best solution
    if best_active:
        active = best_active[:]
        counts = [0] * num_k222s
        current_size = 0
        for i in range(num_triples):
            if active[i]:
                current_size += 1
                for k_idx in triple_to_k222s[i]:
                    counts[k_idx] += 1

        # Try to add any remaining valid triples
        changed = True
        while changed and time_left() > 2:
            changed = False
            for i in range(num_triples):
                if not active[i] and is_valid_add(i, counts):
                    active[i] = True
                    current_size += 1
                    for k_idx in triple_to_k222s[i]:
                        counts[k_idx] += 1
                    changed = True
                    if current_size > best_size:
                        best_size = current_size
                        best_active = active[:]

        result = []
        for i in range(num_triples):
            if best_active[i]:
                result.append(list(all_triples[i]))
        return result
    else:
        # Fallback
        active, counts, current_size = greedy_init()
        result = []
        for i in range(num_triples):
            if active[i]:
                result.append(list(all_triples[i]))
        return result

if __name__ == "__main__":
    result = construct()
    print(f"Number of triples: {len(result)}")
