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
    
    # Precompute for each triple, the list of k222s it belongs to, and for each k222,
    # which triples are "blocking" (i.e., k_count == 7)
    
    class Solver:
        def __init__(self):
            self.present = set()
            self.k_counts = [0] * num_k222s
            # For faster add: track which k222s are at capacity (count == 7)
            self.blocked_k222s = set()  # k222 indices where count == 7
            
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
    
    # Fast polish: try to add any missing triple that fits, using blocked_k222s for speed
    def polish(solver):
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for t_id in range(num_triples):
                if t_id not in solver.present:
                    # Quick check: is any of its k222s blocked?
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

    # Smart 1-swap: try to find a triple to remove and a triple to add that both fit
    def smart_swap(solver):
        if not solver.present:
            return False
        
        present_list = list(solver.present)
        random.shuffle(present_list)
        
        # Try a few random removals
        for t_out in present_list[:10]:
            solver.remove(t_out)
            
            # After removal, try to add many triples
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
                # Try to add t_out back if possible
                if not solver.add(t_out):
                    # Keep it out, we gained triples
                    return True
            else:
                solver.add(t_out)
            
        return False

    # 2-swap: remove 2, add up to 2 (or more via polish)
    def smart_2swap(solver):
        if len(solver.present) < 2:
            return False
        
        present_list = list(solver.present)
        random.shuffle(present_list)
        t_out1, t_out2 = present_list[0], present_list[1]
        
        solver.remove(t_out1)
        solver.remove(t_out2)
        
        # Polish to add as many as possible
        polish(solver)
        
        # Try to add back the removed ones
        added_back = 0
        if solver.add(t_out1):
            added_back += 1
        if solver.add(t_out2):
            added_back += 1
        
        # If we gained net triples, keep the change
        # (polish already maximized, so we just return True if size increased)
        return True

    # Multiple restarts with different random seeds
    for seed in range(80):
        if time.time() > DEADLINE:
            break
        
        solver = Solver()
        
        # Greedy Init with random order
        order = list(range(num_triples))
        random.shuffle(order)
        for t_id in order:
            if time.time() > DEADLINE:
                break
            solver.add(t_id)
            
        # Polish
        polish(solver)
        
        if solver.size() > best_size:
            best_size = solver.size()
            best_set = set(solver.present)
        
        # Local search phase
        sa_start = time.time()
        sa_deadline = min(sa_start + 1.2, DEADLINE)
        
        while time.time() < sa_deadline:
            if random.random() < 0.6:
                smart_swap(solver)
            else:
                smart_2swap(solver)
            
            polish(solver)
            
            if solver.size() > best_size:
                best_size = solver.size()
                best_set = set(solver.present)
                
    return [list(triples[t_id]) for t_id in best_set]
