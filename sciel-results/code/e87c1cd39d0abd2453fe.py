import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 118  # Safety margin before 120s limit

    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)  # 560
    triple_to_id = {t: i for i, t in enumerate(triples)}

    # Precompute K(2,2,2) configurations
    k222_triples = []
    triple_to_k222s = [[] for _ in range(num_triples)]

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

    num_k222s = len(k222_triples)  # 120120

    # Count how many K(2,2,2)s each triple is in
    triple_k222_count = [len(triple_to_k222s[t]) for t in range(num_triples)]

    class State:
        def __init__(self):
            self.present = [False] * num_triples
            self.k_counts = [0] * num_k222s
            self.size = 0
            self.missing = set(range(num_triples))
            self.present_list = []

        def can_add(self, t_id):
            if self.present[t_id]:
                return True
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            return True

        def add(self, t_id):
            if not self.can_add(t_id):
                return False
            self.present[t_id] = True
            self.size += 1
            self.missing.discard(t_id)
            self.present_list.append(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] += 1
            return True

        def remove(self, t_id):
            if not self.present[t_id]:
                return
            self.present[t_id] = False
            self.size -= 1
            self.missing.add(t_id)
            self.present_list.remove(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] -= 1

        def present_list_copy(self):
            return self.present_list[:]

    def polish(state):
        """Systematically try to add all possible missing triples."""
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for t_id in range(num_triples):
                if not state.present[t_id]:
                    if state.add(t_id):
                        improved = True

    def polish_tight(state):
        """Polish focusing on triples that are in tight K(2,2,2)s (k_count >= 5)."""
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for t_id in state.missing:
                # Check if this triple is in any tight K(2,2,2)
                tight = False
                for k_idx in triple_to_k222s[t_id]:
                    if state.k_counts[k_idx] >= 5:
                        tight = True
                        break
                if tight and state.add(t_id):
                    improved = True

    def aggressive_improve(state):
        """Try net-positive exchanges: remove 1 present, add 2 missing."""
        improved = False
        present = state.present_list_copy()
        if not present:
            return improved
        
        # Sort present by "usefulness" - triples in many tight K(2,2,2)s are less useful
        # Actually, we want to remove triples that are blocking the most additions
        # A triple blocks additions if it's in K(2,2,2)s that are at count 7
        # Let's score each present triple by how many K(2,2,2)s it's in with count >= 6
        scores = {}
        for t_id in present:
            score = 0
            for k_idx in triple_to_k222s[t_id]:
                if state.k_counts[k_idx] >= 6:
                    score += (state.k_counts[k_idx] - 5)
            scores[t_id] = score
        
        # Try removing the most "blocking" triples
        sorted_present = sorted(present, key=lambda t: scores[t], reverse=True)
        
        for t_out in sorted_present[:20]:  # Try top 20 most blocking
            if time.time() > DEADLINE:
                break
            state.remove(t_out)
            added_count = 0
            added_triples = []
            for t_in in state.missing:
                if state.add(t_in):
                    added_count += 1
                    added_triples.append(t_in)
                if added_count >= 2:
                    break
            if added_count >= 2:
                improved = True
            else:
                # Revert
                for t_in in added_triples:
                    state.remove(t_in)
                state.add(t_out)
        
        return improved

    def greedy_init(seed):
        """Greedy initialization with LDF (Largest Degree First) ordering."""
        local_rng = random.Random(seed)
        state = State()
        
        # LDF: order by number of K(2,2,2)s (fewest constraints first)
        # With random tie-breaking
        order = list(range(num_triples))
        # Group by k222 count
        from collections import defaultdict
        groups = defaultdict(list)
        for t in order:
            groups[triple_k222_count[t]].append(t)
        
        # Shuffle within each group
        for count in groups:
            local_rng.shuffle(groups[count])
        
        # Sort groups by count
        sorted_groups = sorted(groups.keys())
        order = []
        for count in sorted_groups:
            order.extend(groups[count])
        
        for t_id in order:
            if time.time() > DEADLINE:
                break
            state.add(t_id)
        
        polish(state)
        return state

    def perturb_best(best_present, rng):
        """Create a perturbed copy of the best solution."""
        state = State()
        for i in range(num_triples):
            if best_present[i]:
                state.present[i] = True
                state.size += 1
                state.present_list.append(i)
            else:
                state.missing.add(i)
        
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

        if restart_count <= 10:
            # Greedy init with LDF
            state = greedy_init(seed)
        else:
            # Perturb best solution
            if best_present is None:
                state = greedy_init(seed)
            else:
                state = perturb_best(best_present, rng)

        if state.size > best_size:
            best_size = state.size
            best_present = state.present[:]

        # Local search phase
        for step in range(500):
            if time.time() > DEADLINE:
                break

            # 1. Try aggressive improvement (remove 1, add 2)
            if rng.random() < 0.4:
                if aggressive_improve(state):
                    polish(state)
                    if state.size > best_size:
                        best_size = state.size
                        best_present = state.present[:]
                    continue

            # 2. Try 1-for-2 swap
            present = state.present_list_copy()
            if not present:
                break
            t_out = rng.choice(present)
            
            # Pick 2 random missing triples
            t_in1 = -1
            t_in2 = -1
            missing_list = list(state.missing)
            if len(missing_list) < 2:
                continue
            t_in1, t_in2 = rng.sample(missing_list, 2)
            
            state.remove(t_out)
            added1 = state.add(t_in1)
            added2 = state.add(t_in2) if added1 else False
            
            if added1 and added2:
                if state.size > best_size:
                    best_size = state.size
                    best_present = state.present[:]
            else:
                if added1:
                    state.remove(t_in1)
                if added2:
                    state.remove(t_in2)
                state.add(t_out)
            
            # 3. Try 1-for-1 swap for diversification
            if rng.random() < 0.3:
                present = state.present_list_copy()
                if present:
                    t_out = rng.choice(present)
                    missing_list = list(state.missing)
                    if missing_list:
                        t_in = rng.choice(missing_list)
                        state.remove(t_out)
                        if not state.add(t_in):
                            state.add(t_out)

        # Final polish
        polish(state)
        polish_tight(state)
        if state.size > best_size:
            best_size = state.size
            best_present = state.present[:]

    # Final aggressive improvement pass
    if best_present is not None and time.time() < DEADLINE:
        state = State()
        for i in range(num_triples):
            if best_present[i]:
                state.present[i] = True
                state.size += 1
                state.present_list.append(i)
            else:
                state.missing.add(i)
        for k_idx in range(num_k222s):
            for t_id in k222_triples[k_idx]:
                if state.present[t_id]:
                    state.k_counts[k_idx] += 1
        
        aggressive_improve(state)
        polish(state)
        polish_tight(state)
        
        if state.size > best_size:
            best_size = state.size
            best_present = state.present[:]

    return [list(triples[t_id]) for t_id in range(num_triples) if best_present[t_id]]
