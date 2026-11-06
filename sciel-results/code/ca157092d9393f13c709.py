import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115

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

    class Solver:
        def __init__(self):
            self.present = set()
            self.k_counts = [0] * num_k222s
            self.blocked_k222s = set()

        def can_add(self, t_id):
            if t_id in self.present:
                return True
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            return True

        def add(self, t_id):
            if t_id in self.present:
                return True
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            self.present.add(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] += 1
                if self.k_counts[k_idx] == 7:
                    self.blocked_k222s.add(k_idx)
            return True

        def remove(self, t_id):
            if t_id not in self.present:
                return
            self.present.remove(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] -= 1
                if self.k_counts[k_idx] == 6:
                    self.blocked_k222s.discard(k_idx)

        def size(self):
            return len(self.present)

    best_size = 0
    best_set = set()

    def polish(solver):
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for t_id in range(num_triples):
                if t_id not in solver.present:
                    blocked = False
                    for k_idx in triple_to_k222s[t_id]:
                        if k_idx in solver.blocked_k222s:
                            blocked = True
                            break
                    if not blocked:
                        solver.add(t_id)
                        improved = True
                        if time.time() > DEADLINE:
                            return

    def smart_swap(solver):
        if not solver.present:
            return False

        present_list = list(solver.present)
        random.shuffle(present_list)

        tight_triples = []
        for t_id in present_list:
            for k_idx in triple_to_k222s[t_id]:
                if solver.k_counts[k_idx] == 7:
                    tight_triples.append(t_id)
                    break

        if tight_triples and random.random() < 0.7:
            t_out = random.choice(tight_triples)
        else:
            t_out = random.choice(present_list)

        solver.remove(t_out)

        added_any = False
        for t_id in range(num_triples):
            if t_id not in solver.present and t_id != t_out:
                blocked = False
                for k_idx in triple_to_k222s[t_id]:
                    if k_idx in solver.blocked_k222s:
                        blocked = True
                        break
                if not blocked:
                    solver.add(t_id)
                    added_any = True

        if added_any:
            if not solver.add(t_out):
                return True
        else:
            solver.add(t_out)

        return False

    def smart_2swap(solver):
        if len(solver.present) < 2:
            return False

        present_list = list(solver.present)
        random.shuffle(present_list)

        tight_triples = []
        for t_id in present_list:
            for k_idx in triple_to_k222s[t_id]:
                if solver.k_counts[k_idx] == 7:
                    tight_triples.append(t_id)
                    break

        if len(tight_triples) >= 2 and random.random() < 0.6:
            t_out1, t_out2 = random.sample(tight_triples, 2)
        elif len(tight_triples) >= 1 and random.random() < 0.8:
            t_out1 = random.choice(tight_triples)
            t_out2 = random.choice([t for t in present_list if t != t_out1])
        else:
            t_out1, t_out2 = random.sample(present_list, 2)

        solver.remove(t_out1)
        solver.remove(t_out2)

        polish(solver)

        if solver.add(t_out1):
            pass
        if solver.add(t_out2):
            pass

        return True

    def aggressive_improve(solver):
        if not solver.present:
            return False

        present_list = list(solver.present)
        random.shuffle(present_list)

        for t_out in present_list[:5]:
            solver.remove(t_out)

            added_count = 0
            for t_id in range(num_triples):
                if t_id not in solver.present and t_id != t_out:
                    blocked = False
                    for k_idx in triple_to_k222s[t_id]:
                        if k_idx in solver.blocked_k222s:
                            blocked = True
                            break
                    if not blocked:
                        solver.add(t_id)
                        added_count += 1

            if added_count >= 2:
                return True
            else:
                solver.add(t_out)

        return False

    for seed in range(120):
        if time.time() > DEADLINE:
            break

        solver = Solver()

        order = list(range(num_triples))
        random.shuffle(order)
        for t_id in order:
            if time.time() > DEADLINE:
                break
            solver.add(t_id)

        polish(solver)

        if solver.size() > best_size:
            best_size = solver.size()
            best_set = set(solver.present)

        sa_start = time.time()
        sa_deadline = min(sa_start + 1.0, DEADLINE)

        while time.time() < sa_deadline:
            r = random.random()
            if r < 0.4:
                smart_swap(solver)
            elif r < 0.7:
                smart_2swap(solver)
            else:
                aggressive_improve(solver)

            polish(solver)

            if solver.size() > best_size:
                best_size = solver.size()
                best_set = set(solver.present)

    return [list(triples[t_id]) for t_id in best_set]
