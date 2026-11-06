import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 117  # Safety margin before 120s limit

    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)  # 560
    triple_to_id = {t: i for i, t in enumerate(triples)}

    # Precompute K(2,2,2) configurations
    def get_pair_partitions(six):
        """Generate all 15 ways to partition 6 vertices into 3 pairs."""
        v = list(six)
        parts = []
        for i in range(1, 6):
            pair1 = (v[0], v[i])
            rem = [v[j] for j in range(1, 6) if j != i]
            for j in range(1, 4):
                pair2 = (rem[0], rem[j])
                if j == 1:
                    pair3 = (rem[2], rem[3])
                elif j == 2:
                    pair3 = (rem[1], rem[3])
                else:
                    pair3 = (rem[1], rem[2])
                parts.append([pair1, pair2, pair3])
        return parts

    k222_triples = []
    triple_to_k222s = [[] for _ in range(num_triples)]

    for six_subset in combinations(range(n), 6):
        for partition in get_pair_partitions(six_subset):
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

    # Build numpy bitmask for each triple: which K(2,2,2)s it's in
    triple_kmask = [np.zeros(num_k222s, dtype=np.uint8) for _ in range(num_triples)]
    for t_id in range(num_triples):
        for k_idx in triple_to_k222s[t_id]:
            triple_kmask[t_id][k_idx] = 1

    # Build conflict bitmask: for each triple, which other triples conflict with it
    # Two triples conflict if they share a K(2,2,2) where together they'd complete it
    # Actually, conflict means: if both are present, some K(2,2,2) would have all 8
    # A triple t conflicts with triple t' if there exists a K(2,2,2) k such that
    # k contains t and t', and k has exactly 7 other triples present (but we need
    # to track counts, not just presence)

    # For the conflict graph (independent set formulation):
    # Two triples are in conflict if they appear in a common K(2,2,2)
    # This is a superset - the actual constraint is that no K(2,2,2) has all 8 present
    # For a proper independent set, we need: for any K(2,2,2), at most 7 of its 8 triples
    # are in the independent set.
    # This is NOT a simple pairwise conflict - it's a hypergraph independent set.
    # But we can still use local search effectively.

    # Precompute for each triple, list of K(2,2,2)s it belongs to (for fast access)
    triple_k222_list = triple_to_k222s

    # State management with numpy
    class State:
        def __init__(self):
            self.present = np.zeros(num_triples, dtype=np.uint8)
            self.k_counts = np.zeros(num_k222s, dtype=np.uint8)
            self.size = 0
            self.missing = list(range(num_triples))
            self.present_list = []

        def can_add(self, t_id):
            if self.present[t_id]:
                return True
            for k_idx in triple_k222_list[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            return True

        def add(self, t_id):
            if self.present[t_id]:
                return False
            for k_idx in triple_k222_list[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            self.present[t_id] = 1
            self.size += 1
            if t_id in self.missing:
                self.missing.remove(t_id)
            self.present_list.append(t_id)
            for k_idx in triple_k222_list[t_id]:
                self.k_counts[k_idx] += 1
            return True

        def remove(self, t_id):
            if not self.present[t_id]:
                return
            self.present[t_id] = 0
            self.size -= 1
            self.missing.append(t_id)
            self.present_list.remove(t_id)
            for k_idx in triple_k222_list[t_id]:
                self.k_counts[k_id] -= 1

        def present_list_copy(self):
            return self.present_list[:]

    # Fix the remove method typo
    class State:
        def __init__(self):
            self.present = np.zeros(num_triples, dtype=np.uint8)
            self.k_counts = np.zeros(num_k222s, dtype=np.uint8)
            self.size = 0
            self.missing_set = set(range(num_triples))
            self.present_list = []

        def can_add(self, t_id):
            if self.present[t_id]:
                return True
            for k_idx in triple_k222_list[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            return True

        def add(self, t_id):
            if self.present[t_id]:
                return False
            for k_idx in triple_k222_list[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            self.present[t_id] = 1
            self.size += 1
            self.missing_set.discard(t_id)
            self.present_list.append(t_id)
            for k_idx in triple_k222_list[t_id]:
                self.k_counts[k_idx] += 1
            return True

        def remove(self, t_id):
            if not self.present[t_id]:
                return
            self.present[t_id] = 0
            self.size -= 1
            self.missing_set.add(t_id)
            self.present_list.remove(t_id)
            for k_idx in triple_k222_list[t_id]:
                self.k_counts[k_idx] -= 1

        def present_list_copy(self):
            return self.present_list[:]

    def polish(state):
        """Systematically try to add all possible missing triples."""
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            missing_snapshot = list(state.missing_set)
            random.shuffle(missing_snapshot)
            for t_id in missing_snapshot:
                if not state.present[t_id]:
                    if state.add(t_id):
                        improved = True

    def greedy_init(seed):
        """Greedy initialization with LDF ordering."""
        local_rng = random.Random(seed)
        state = State()

        order = list(range(num_triples))
        local_rng.shuffle(order)

        for t_id in order:
            if time.time() > DEADLINE:
                break
            state.add(t_id)

        polish(state)
        return state

    def simulated_annealing(state, best_size, best_present, rng, max_iter=50000):
        """Simulated annealing to improve the solution."""
        current_size = state.size
        current_temp = 5.0
        cooling_rate = 0.9999
        min_temp = 0.01
        iterations = 0

        while time.time() < DEADLINE and current_temp > min_temp:
            iterations += 1

            # Choose a random move
            move_type = rng.random()

            if move_type < 0.5:
                # Try to add a random missing triple
                if state.missing_set:
                    t_in = rng.choice(list(state.missing_set))
                    if state.add(t_in):
                        new_size = state.size
                        if new_size > best_size:
                            best_size = new_size
                            best_present = state.present.copy()
                    else:
                        # Can't add, try removing a present triple instead
                        if state.present_list:
                            t_out = rng.choice(state.present_list)
                            state.remove(t_out)

            elif move_type < 0.8:
                # Try to remove a random present triple
                if state.present_list:
                    t_out = rng.choice(state.present_list)
                    state.remove(t_out)
                    # Then try to add some missing ones
                    added = 0
                    missing_snapshot = list(state.missing_set)
                    random.shuffle(missing_snapshot)
                    for t_in in missing_snapshot[:5]:
                        if state.add(t_in):
                            added += 1
                    # Check if net positive
                    if state.size <= current_size - 2:
                        # Revert - too much loss
                        pass  # Keep the state as is for now

            else:
                # Swap: remove one, try to add two
                if state.present_list and len(state.missing_set) >= 2:
                    t_out = rng.choice(state.present_list)
                    missing_list = list(state.missing_set)
                    t_in1, t_in2 = rng.sample(missing_list, 2)

                    state.remove(t_out)
                    added1 = state.add(t_in1)
                    added2 = state.add(t_in2)

                    if added1 and added2:
                        if state.size > best_size:
                            best_size = state.size
                            best_present = state.present.copy()
                    else:
                        # Revert
                        if added1:
                            state.remove(t_in1)
                        if added2:
                            state.remove(t_in2)
                        state.add(t_out)

            # Acceptance step (simplified - always accept for speed)
            # In proper SA, we'd compute energy difference and accept with probability
            current_temp *= cooling_rate

        return best_size, best_present

    def perturb_best(best_present, rng):
        """Create a perturbed copy of the best solution."""
        state = State()
        for i in range(num_triples):
            if best_present[i]:
                state.present[i] = 1
                state.size += 1
                state.present_list.append(i)
            else:
                state.missing_set.add(i)

        for k_idx in range(num_k222s):
            for t_id in k222_triples[k_idx]:
                if state.present[t_id]:
                    state.k_counts[k_idx] += 1

        # Perturb: remove some random present triples
        n_perturb = rng.randint(5, 15)
        present = state.present_list_copy()
        for _ in range(n_perturb):
            if present:
                t = rng.choice(present)
                state.remove(t)
                present = state.present_list_copy()

        polish(state)
        return state

    best_size = 0
    best_present = None
    rng = random.Random(42)

    # Phase 1: Multiple greedy restarts
    restart_count = 0
    while time.time() < DEADLINE:
        restart_count += 1
        seed = 42 + restart_count * 7919

        if restart_count <= 5:
            # Greedy init
            state = greedy_init(seed)
        elif best_present is not None:
            # Perturb best solution
            state = perturb_best(best_present, rng)
        else:
            state = greedy_init(seed)

        if state.size > best_size:
            best_size = state.size
            best_present = state.present.copy()

        # Simulated annealing phase
        if time.time() < DEADLINE:
            best_size, best_present = simulated_annealing(
                state, best_size, best_present, rng
            )

        # Final polish
        polish(state)
        if state.size > best_size:
            best_size = state.size
            best_present = state.present.copy()

        # Check if we should continue
        if time.time() > DEADLINE - 5:
            break

    # Final polish on best solution
    if best_present is not None and time.time() < DEADLINE:
        state = State()
        for i in range(num_triples):
            if best_present[i]:
                state.present[i] = 1
                state.size += 1
                state.present_list.append(i)
            else:
                state.missing_set.add(i)
        for k_idx in range(num_k222s):
            for t_id in k222_triples[k_idx]:
                if state.present[t_id]:
                    state.k_counts[k_idx] += 1

        # Multiple polish passes
        for _ in range(5):
            if time.time() > DEADLINE - 2:
                break
            polish(state)

        if state.size > best_size:
            best_size = state.size
            best_present = state.present.copy()

    return [list(triples[t_id]) for t_id in range(num_triples) if best_present is not None and best_present[t_id]]
