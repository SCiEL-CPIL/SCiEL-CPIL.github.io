import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115  # 35s safety margin before 150s kill
    
    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)  # 560
    triple_to_id = {t: i for i, t in enumerate(triples)}
    
    # Precompute K(2,2,2) configurations
    k222_triples = []  # list of lists of triple IDs
    triple_to_k222s = [[] for _ in range(num_triples)]
    
    def get_partitions(vertices):
        """Get all ways to partition 6 vertices into 3 pairs."""
        v = list(vertices)
        partitions = []
        for i in range(1, 6):
            pair1 = (v[0], v[i])
            remaining = [v[j] for j in range(1, 6) if j != i]
            for j in range(1, 4):
                pair2 = (remaining[0], remaining[j])
                remaining2 = [remaining[k] for k in range(1, 4) if k != j]
                pair3 = (remaining2[0], remaining2[1])
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
    
    # Solver class with efficient constraint tracking
    class Solver:
        def __init__(self):
            self.present = set()
            self.k_counts = [0] * num_k222s
            self.tight_k_set = set()  # K(2,2,2)s with count == 7
        
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
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] += 1
                if self.k_counts[k_idx] == 7:
                    self.tight_k_set.add(k_idx)
                elif self.k_counts[k_idx] == 8:
                    self.tight_k_set.discard(k_idx)
            return True
        
        def remove(self, t_id):
            if t_id not in self.present:
                return
            self.present.remove(t_id)
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] == 7:
                    self.tight_k_set.discard(k_idx)
                self.k_counts[k_idx] -= 1
                if self.k_counts[k_idx] == 7:
                    self.tight_k_set.add(k_idx)
        
        def size(self):
            return len(self.present)
        
        def get_tight_triples(self):
            """Return triples that are in at least one K(2,2,2) with count == 7."""
            tight = []
            seen = set()
            for k_idx in self.tight_k_set:
                for t_id in k222_triples[k_idx]:
                    if t_id in self.present and t_id not in seen:
                        tight.append(t_id)
                        seen.add(t_id)
            return tight
        
        def get_addable_missing(self):
            """Return list of missing triples that can be added, using tight_k_set for fast rejection."""
            addable = []
            for t_id in range(num_triples):
                if t_id in self.present:
                    continue
                # Fast check: if any K containing t_id is tight, skip
                blocked = False
                for k_idx in triple_to_k222s[t_id]:
                    if k_idx in self.tight_k_set:
                        blocked = True
                        break
                if not blocked:
                    addable.append(t_id)
            return addable
    
    best_size = 0
    best_set = set()
    
    def fast_polish(solver, max_iterations=200):
        """Add any missing triple that fits, using incremental updates."""
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < DEADLINE:
            improved = False
            iterations += 1
            # Get all addable missing triples
            addable = solver.get_addable_missing()
            for t_id in addable:
                if solver.can_add(t_id):
                    if solver.add(t_id):
                        improved = True
            if time.time() > DEADLINE:
                break
    
    def smart_swap(solver, rng):
        """1-swap: remove a triple, try to add a better one."""
        if not solver.present:
            return False
        
        present_list = list(solver.present)
        tight_triples = solver.get_tight_triples()
        
        # Prefer removing from tight constraints
        if tight_triples and rng.random() < 0.8:
            t_out = rng.choice(tight_triples)
        else:
            t_out = rng.choice(present_list)
        
        solver.remove(t_out)
        
        # Find all addable candidates
        addable = solver.get_addable_missing()
        
        if not addable:
            solver.add(t_out)
            return False
        
        # Pick randomly from addable
        t_in = rng.choice(addable)
        if solver.add(t_in):
            return True
        else:
            solver.add(t_out)
            return False
    
    def smart_2swap(solver, rng):
        """2-swap: remove two, add two."""
        if len(solver.present) < 2:
            return False
        
        present_list = list(solver.present)
        tight_triples = solver.get_tight_triples()
        
        if len(tight_triples) >= 2 and rng.random() < 0.7:
            t_out1, t_out2 = rng.sample(tight_triples, 2)
        elif len(tight_triples) >= 1 and rng.random() < 0.85:
            t_out1 = rng.choice(tight_triples)
            others = [t for t in present_list if t != t_out1]
            t_out2 = rng.choice(others)
        else:
            t_out1, t_out2 = rng.sample(present_list, 2)
        
        solver.remove(t_out1)
        solver.remove(t_out2)
        
        addable = solver.get_addable_missing()
        
        if len(addable) < 2:
            solver.add(t_out1)
            solver.add(t_out2)
            return False
        
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
            solver.add(t_out1)
            solver.add(t_out2)
            return False
    
    def remove_and_add_multiple(solver, rng, num_remove=1, num_add=2):
        """Remove num_remove triples, try to add num_add."""
        if len(solver.present) < num_remove:
            return False
        
        present_list = list(solver.present)
        tight_triples = solver.get_tight_triples()
        
        if tight_triples:
            pool = tight_triples if len(tight_triples) >= num_remove else present_list
        else:
            pool = present_list
        
        if len(pool) < num_remove:
            pool = present_list
        
        to_remove = rng.sample(pool, min(num_remove, len(pool)))
        
        for t in to_remove:
            solver.remove(t)
        
        addable = solver.get_addable_missing()
        
        if len(addable) < num_add:
            for t in to_remove:
                solver.add(t)
            return False
        
        to_add = rng.sample(addable, num_add)
        all_ok = True
        added = []
        for t in to_add:
            if solver.add(t):
                added.append(t)
            else:
                all_ok = False
                break
        
        if all_ok:
            return True
        else:
            for t in added:
                solver.remove(t)
            for t in to_remove:
                solver.add(t)
            return False
    
    # Main search
    rng = random.Random(42)
    
    # Phase 1: Multiple greedy starts, keep best
    for seed_idx in range(5):
        if time.time() > DEADLINE - 60:
            break
        
        test_solver = Solver()
        test_order = list(range(num_triples))
        rng.shuffle(test_order)
        
        for t_id in test_order:
            if time.time() > DEADLINE - 60:
                break
            test_solver.add(t_id)
        
        fast_polish(test_solver)
        
        if test_solver.size() > best_size:
            best_size = test_solver.size()
            best_set = set(test_solver.present)
    
    # Use best as starting point
    solver = Solver()
    solver.present = set(best_set)
    for k_idx in range(num_k222s):
        count = 0
        for t_id in k222_triples[k_idx]:
            if t_id in solver.present:
                count += 1
        solver.k_counts[k_idx] = count
        if count == 7:
            solver.tight_k_set.add(k_idx)
    
    # Phase 2: Local search
    step = 0
    no_improve_steps = 0
    last_best = best_size
    
    while time.time() < DEADLINE - 30:
        step += 1
        
        op = rng.random()
        if op < 0.35:
            smart_swap(solver, rng)
        elif op < 0.65:
            smart_2swap(solver, rng)
        elif op < 0.85:
            remove_and_add_multiple(solver, rng, num_remove=1, num_add=2)
        else:
            fast_polish(solver, max_iterations=50)
        
        if solver.size() > best_size:
            best_size = solver.size()
            best_set = set(solver.present)
            no_improve_steps = 0
        else:
            no_improve_steps += 1
        
        # Restart if stuck for too long
        if no_improve_steps > 300:
            no_improve_steps = 0
            # Try fresh start
            fresh_solver = Solver()
            fresh_order = list(range(num_triples))
            rng.shuffle(fresh_order)
            for t_id in fresh_order:
                if time.time() > DEADLINE - 30:
                    break
                fresh_solver.add(t_id)
            fast_polish(fresh_solver)
            
            if fresh_solver.size() > best_size:
                best_size = fresh_solver.size()
                best_set = set(fresh_solver.present)
                solver = fresh_solver
            elif fresh_solver.size() > solver.size() - 3 and rng.random() < 0.4:
                solver = fresh_solver
    
    # Final high-precision polish
    final_solver = Solver()
    final_solver.present = set(best_set)
    for k_idx in range(num_k222s):
        count = 0
        for t_id in k222_triples[k_idx]:
            if t_id in final_solver.present:
                count += 1
        final_solver.k_counts[k_idx] = count
        if count == 7:
            final_solver.tight_k_set.add(k_idx)
    
    # Multiple polish passes with increasing effort
    for polish_iter in range(5):
        if time.time() > DEADLINE - 5:
            break
        fast_polish(final_solver, max_iterations=500)
        if final_solver.size() > best_size:
            best_size = final_solver.size()
            best_set = set(final_solver.present)
    
    result = [list(triples[t_id]) for t_id in sorted(best_set)]
    return result
