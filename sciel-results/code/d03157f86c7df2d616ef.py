import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115  # Safety margin before 120s limit
    
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
    
    num_k222s = len(k222_triples)
    
    # Count how many K(2,2,2)s each triple is in
    triple_k222_count = [len(triple_to_k222s[t]) for t in range(num_triples)]
    
    class State:
        def __init__(self):
            self.present = [False] * num_triples
            self.k_counts = [0] * num_k222s
            self.size = 0
        
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
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] += 1
            return True
        
        def remove(self, t_id):
            if not self.present[t_id]:
                return
            self.present[t_id] = False
            self.size -= 1
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] -= 1
        
        def present_list(self):
            return [i for i in range(num_triples) if self.present[i]]
    
    def polish(state, max_iters=50000):
        """Try to add any missing triple that fits, in order of least constraint."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and time.time() < DEADLINE:
            improved = False
            iter_count += 1
            # Get missing triples sorted by constraint count (ascending)
            missing = [(triple_k222_count[t], t) for t in range(num_triples) if not state.present[t]]
            missing.sort()
            for _, t_id in missing:
                if state.add(t_id):
                    improved = True
            # Also try random additions
            for _ in range(200):
                if time.time() > DEADLINE:
                    break
                t_id = random.randint(0, num_triples - 1)
                if not state.present[t_id] and state.add(t_id):
                    improved = True
    
    def simulated_annealing(state, initial_temp=10.0, cooling=0.999, max_iters=20000):
        """Simulated annealing local search."""
        temp = initial_temp
        current_size = state.size
        best_size = state.size
        best_present = state.present[:]
        
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
            if temp < 0.1:
                break
            
            # Random move: remove one present, try to add one missing
            present = state.present_list()
            if not present:
                break
            
            t_out = random.choice(present)
            t_in = random.randint(0, num_triples - 1)
            
            if t_in == t_out or state.present[t_in]:
                continue
            
            state.remove(t_out)
            
            if state.add(t_in):
                new_size = state.size
                delta = new_size - current_size
                if delta >= 0 or random.random() < np.exp(delta / temp):
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_present = state.present[:]
                else:
                    state.remove(t_in)
                    state.add(t_out)
            else:
                # Can't add t_in, try another
                for _ in range(5):
                    t_in2 = random.randint(0, num_triples - 1)
                    if t_in2 != t_out and not state.present[t_in2]:
                        if state.add(t_in2):
                            new_size = state.size
                            delta = new_size - current_size
                            if delta >= 0 or random.random() < np.exp(delta / temp):
                                current_size = new_size
                                if current_size > best_size:
                                    best_size = current_size
                                    best_present = state.present[:]
                            else:
                                state.remove(t_in2)
                                state.add(t_out)
                            break
                        else:
                            continue
                else:
                    state.add(t_out)
            
            temp *= cooling
        
        # Restore best if we ended up worse
        if best_size > state.size:
            # Reconstruct state from best_present
            for i in range(num_triples):
                if state.present[i]:
                    state.remove(i)
            for i in range(num_triples):
                if best_present[i]:
                    state.add(i)
        
        return best_size, best_present
    
    def greedy_init(seed):
        local_rng = random.Random(seed)
        state = State()
        order = list(range(num_triples))
        # Sort by number of K(2,2,2)s (fewer constraints first) with randomness
        base_order = sorted(range(num_triples), key=lambda t: triple_k222_count[t])
        # Add randomness by shuffling within groups of similar constraint count
        for i in range(len(base_order)):
            if i > 0:
                # Small chance to swap with nearby element
                if local_rng.random() < 0.1:
                    j = max(0, min(len(base_order)-1, i + local_rng.randint(-5, 5)))
                    base_order[i], base_order[j] = base_order[j], base_order[i]
        
        for t_id in base_order:
            if time.time() > DEADLINE:
                break
            state.add(t_id)
        polish(state)
        return state
    
    best_size = 0
    best_present = None
    rng = random.Random(42)
    
    # Main loop: multiple restarts
    restart_count = 0
    while time.time() < DEADLINE:
        restart_count += 1
        seed = 42 + restart_count * 7919
        
        if restart_count <= 3:
            # Greedy init
            state = greedy_init(seed)
        else:
            # Start from best and perturb
            state = State()
            for i in range(num_triples):
                if best_present and best_present[i]:
                    state.add(i)
            
            # Perturb: remove some random present triples
            n_perturb = min(15, state.size // 10 + 3)
            present = state.present_list()
            for _ in range(n_perturb):
                if present:
                    t = random.choice(present)
                    state.remove(t)
                    present = state.present_list()
            
            polish(state)
        
        if state.size > best_size:
            best_size = state.size
            best_present = state.present[:]
        
        # Simulated annealing
        sa_size, sa_present = simulated_annealing(state)
        if sa_size > best_size:
            best_size = sa_size
            best_present = sa_present[:]
        
        # Polish after SA
        polish(state)
        if state.size > best_size:
            best_size = state.size
            best_present = state.present[:]
    
    return [list(triples[t_id]) for t_id in range(num_triples) if best_present and best_present[t_id]]
