import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 118

    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_id = {t: i for i, t in enumerate(triples)}

    # Precompute K(2,2,2)s
    k222_triples = []
    triple_to_k222s = [[] for _ in range(num_triples)]

    def get_partitions(vertices):
        v = list(vertices)
        partitions = []
        for i in range(1, 6):
            pair1 = (v[0], v[i])
            remaining = [v[j] for j in range(1, 6) if j != i]
            for j in range(1, 4):
                pair2 = (remaining[0], remaining[j])
                if j == 1:
                    pair3 = (remaining[2], remaining[3])
                elif j == 2:
                    pair3 = (remaining[1], remaining[3])
                else:
                    pair3 = (remaining[1], remaining[2])
                partitions.append([pair1, pair2, pair3])
        return partitions

    for six_subset in combinations(range(n), 6):
        for partition in get_partitions(six_subset):
            t_ids = []
            for x in partition[0]:
                for y in partition[1]:
                    for z in partition[2]:
                        t = tuple(sorted((x, y, z)))
                        t_id = triple_to_id[t]
                        t_ids.append(t_id)

            k_idx = len(k222_triples)
            k222_triples.append(t_ids)

            for t_id in t_ids:
                triple_to_k222s[t_id].append(k_idx)

    num_k222s = len(k222_triples)
    triple_k222_count = [len(triple_to_k222s[t]) for t in range(num_triples)]

    class Solver:
        __slots__ = ['present', 'k_counts', 'size']

        def __init__(self):
            self.present = set()
            self.k_counts = [0] * num_k222s
            self.size = 0

        def can_add(self, t_id):
            if t_id in self.present:
                return True
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            return True

        def add(self, t_id):
            if not self.can_add(t_id):
                return False
            self.present.add(t_id)
            self.size += 1
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] += 1
            return True

        def remove(self, t_id):
            if t_id not in self.present:
                return
            self.present.remove(t_id)
            self.size -= 1
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] -= 1

    def polish(solver, max_iters=50):
        """Systematically try to add all possible missing triples."""
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < DEADLINE:
            improved = False
            iters += 1
            for t_id in range(num_triples):
                if t_id not in solver.present:
                    if solver.add(t_id):
                        improved = True

    def get_tight_triples(solver):
        """Get triples that are in at least one tight K(2,2,2) (count == 7)."""
        tight = []
        for t_id in solver.present:
            for k_idx in triple_to_k222s[t_id]:
                if solver.k_counts[k_idx] == 7:
                    tight.append(t_id)
                    break
        return tight

    def random_1swap(solver, rng):
        """Conflict-aware 1-swap: remove one, add one that was blocked."""
        if not solver.present:
            return False

        present_list = list(solver.present)
        tight = get_tight_triples(solver)

        if tight and rng.random() < 0.7:
            t_out = rng.choice(tight)
        else:
            t_out = rng.choice(present_list)

        solver.remove(t_out)

        # Find candidates that can now be added
        candidates = []
        for t_id in range(num_triples):
            if t_id not in solver.present and t_id != t_out:
                if solver.can_add(t_id):
                    candidates.append(t_id)

        if not candidates:
            solver.add(t_out)
            return False

        t_in = rng.choice(candidates)
        if solver.add(t_in):
            return True
        else:
            solver.add(t_out)
            return False

    def random_2swap(solver, rng):
        """Conflict-aware 2-swap: remove two, add two."""
        if solver.size < 2:
            return False

        present_list = list(solver.present)
        tight = get_tight_triples(solver)

        if len(tight) >= 2 and rng.random() < 0.6:
            t_out1, t_out2 = rng.sample(tight, 2)
        elif len(tight) >= 1 and rng.random() < 0.8:
            t_out1 = rng.choice(tight)
            others = [t for t in present_list if t != t_out1]
            t_out2 = rng.choice(others)
        else:
            t_out1, t_out2 = rng.sample(present_list, 2)

        solver.remove(t_out1)
        solver.remove(t_out2)

        candidates = []
        for t_id in range(num_triples):
            if t_id not in solver.present and t_id != t_out1 and t_id != t_out2:
                if solver.can_add(t_id):
                    candidates.append(t_id)

        if len(candidates) < 2:
            solver.add(t_out1)
            solver.add(t_out2)
            return False

        t_in1, t_in2 = rng.sample(candidates, 2)
        ok1 = solver.add(t_in1)
        ok2 = solver.add(t_in2)

        if ok1 and ok2:
            return True
        else:
            if ok1:
                solver.remove(t_in1)
            if ok2:
                solver.remove(t_in2)
            solver.add(t_out1)
            solver.add(t_out2)
            return False

    def aggressive_1_for_2(solver, rng, max_attempts=30):
        """Remove one triple, try to add two."""
        if not solver.present:
            return False

        for _ in range(max_attempts):
            if time.time() > DEADLINE:
                break

            present_list = list(solver.present)
            tight = get_tight_triples(solver)

            if tight and rng.random() < 0.6:
                t_out = rng.choice(tight)
            else:
                t_out = rng.choice(present_list)

            solver.remove(t_out)

            addable = []
            for t_id in range(num_triples):
                if t_id not in solver.present and t_id != t_out:
                    if solver.can_add(t_id):
                        addable.append(t_id)

            if len(addable) >= 2:
                t_in1, t_in2 = rng.sample(addable, 2)
                ok1 = solver.add(t_in1)
                ok2 = solver.add(t_in2)
                if ok1 and ok2:
                    return True
                else:
                    if ok1:
                        solver.remove(t_in1)
                    if ok2:
                        solver.remove(t_in2)
            elif len(addable) == 1:
                ok = solver.add(addable[0])
                if ok:
                    return True
                else:
                    solver.add(t_out)
                    return False
            else:
                solver.add(t_out)
                return False

        solver.add(t_out) if t_out not in solver.present else None
        return False

    def greedy_init(seed):
        """Greedy initialization sorted by K(2,2,2) count with perturbation."""
        local_rng = random.Random(seed)
        solver = Solver()
        order = list(range(num_triples))
        order.sort(key=lambda t: triple_k222_count[t])
        # Add randomness near the top
        for i in range(1, min(len(order), 100)):
            if local_rng.random() < 0.3:
                swap_idx = i - local_rng.randint(0, min(i, 15))
                order[i], order[swap_idx] = order[swap_idx], order[i]

        for t_id in order:
            if time.time() > DEADLINE:
                break
            solver.add(t_id)

        polish(solver, max_iters=20)
        return solver

    best_size = 0
    best_present = set()

    rng = random.Random(12345)

    # Multiple restarts with different strategies
    restart_count = 0
    while time.time() < DEADLINE - 5:
        restart_count += 1
        seed = 42 + restart_count * 7919

        if restart_count <= 8:
            solver = greedy_init(seed)
        else:
            # Perturb best solution
            solver = Solver()
            for t_id in best_present:
                solver.present.add(t_id)
                solver.size += 1
            for k_idx in range(num_k222s):
                count = 0
                for t_id in k222_triples[k_idx]:
                    if t_id in solver.present:
                        count += 1
                solver.k_counts[k_idx] = count

            # Perturb: remove some triples
            n_perturb = min(20, max(5, solver.size // 10))
            present_list = list(solver.present)
            for _ in range(n_perturb):
                if solver.present:
                    t = rng.choice(list(solver.present))
                    solver.remove(t)

            polish(solver, max_iters=10)

        if solver.size > best_size:
            best_size = solver.size
            best_present = solver.present.copy()

        # Local search phase
        search_iters = 200
        for _ in range(search_iters):
            if time.time() > DEADLINE - 3:
                break

            # Try different move types with varying probability
            r = rng.random()
            if r < 0.35:
                success = random_1swap(solver, rng)
            elif r < 0.65:
                success = random_2swap(solver, rng)
            else:
                success = aggressive_1_for_2(solver, rng, max_attempts=15)

            if success:
                polish(solver, max_iters=5)
                if solver.size > best_size:
                    best_size = solver.size
                    best_present = solver.present.copy()

        # Final polish
        polish(solver, max_iters=30)
        if solver.size > best_size:
            best_size = solver.size
            best_present = solver.present.copy()

    return [list(triples[t_id]) for t_id in sorted(best_present)]
