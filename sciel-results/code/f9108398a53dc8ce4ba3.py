import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill
    
    n = 128
    TARGET_START = 128
    TARGET_END = 256
    TARGETS = list(range(TARGET_START, TARGET_END + 1))
    TARGET_COUNT = len(TARGETS)
    
    # Precompute target indices in convolution output
    # conv = np.convolve(v, v) has length 2*129-1 = 257, index t gives ordered pairs summing to t
    
    class FastEvaluator:
        """Incremental evaluator that tracks representation counts."""
        def __init__(self):
            self.v = np.zeros(129, dtype=np.int32)
            self.conv = np.zeros(257, dtype=np.int32)  # ordered pair counts
            self.A = set()
            self._update_conv()
        
        def _update_conv(self):
            self.conv = np.convolve(self.v, self.v, mode='full')
        
        def add(self, x):
            if x in self.A:
                return
            self.A.add(x)
            self.v[x] = 1
            # Update conv incrementally: adding x adds x to each position
            # conv[t] increases by v[t-x] + v[t-x] for the cross terms, plus 1 if t=2x
            # Actually simpler: just recompute conv for the affected range
            # For speed, update manually:
            # New contributions: for each y in A (including x), conv[x+y] += 1 (ordered)
            # But we need to be careful. Let's just recompute conv when needed.
            self._update_conv()
        
        def remove(self, x):
            if x not in self.A:
                return
            self.A.remove(x)
            self.v[x] = 0
            self._update_conv()
        
        def get_score(self):
            """Return (max_multiplicity, covered) for current set."""
            max_mult = 0
            covered = True
            for t in range(TARGET_START, TARGET_END + 1):
                if t >= len(self.conv):
                    covered = False
                    break
                ordered = self.conv[t]
                diag = 1 if (t % 2 == 0 and (t // 2) in self.A) else 0
                unordered = (ordered + diag) // 2
                if unordered == 0:
                    covered = False
                if unordered > max_mult:
                    max_mult = unordered
            return max_mult, covered
    
    def evaluate(A_set):
        """Return (max_multiplicity, covered) for the given set A."""
        v = np.zeros(129, dtype=np.int32)
        for x in A_set:
            v[x] = 1
        conv = np.convolve(v, v, mode='full')
        max_mult = 0
        covered = True
        for t in range(TARGET_START, TARGET_END + 1):
            if t >= len(conv):
                covered = False
                break
            ordered = conv[t]
            diag = 1 if (t % 2 == 0 and (t // 2) in A_set) else 0
            unordered = (ordered + diag) // 2
            if unordered == 0:
                covered = False
            if unordered > max_mult:
                max_mult = unordered
        return max_mult, covered
    
    def make_initials():
        initials = []
        initials.append(set(range(0, 129)))
        initials.append(set(range(64, 129)))
        initials.append(set(range(0, 129, 2)))
        random.seed(123)
        for size in [40, 50, 60, 70, 80, 90, 100]:
            s = set(random.sample(range(0, 129), size))
            initials.append(s)
        for k in [2, 3, 4]:
            s = set(range(0, 129, k))
            initials.append(s)
        s = set()
        for i in range(0, 129, 2):
            s.add(i)
            s.add(128 - i)
        initials.append(s)
        return initials
    
    def local_search(A_set, time_budget, seed=None):
        """Simulated annealing local search. Returns (best_A, best_score)."""
        if seed is not None:
            random.seed(seed)
        
        current_A = set(A_set)
        current_score, current_covered = evaluate(current_A)
        
        if not current_covered:
            # Try to fix coverage greedily
            for t in TARGETS:
                found = False
                for a in current_A:
                    b = t - a
                    if b >= a and b in current_A:
                        found = True
                        break
                if not found:
                    mid = t // 2
                    if 0 <= mid <= 128:
                        current_A.add(mid)
                    if t % 2 == 1 and mid + 1 <= 128:
                        current_A.add(mid + 1)
            current_score, current_covered = evaluate(current_A)
        
        best_A = set(current_A)
        best_score = current_score if current_covered else float('inf')
        
        T = 30.0
        T_min = 0.5
        cooling = 0.9995
        
        while time.time() < time_budget and T > T_min:
            # Generate a move: add, remove, or swap
            move_type = random.random()
            if len(current_A) > 5 and move_type < 0.4:
                # Remove
                x = random.choice(list(current_A))
                new_A = current_A - {x}
            elif move_type < 0.7:
                # Add
                x = random.randint(0, 128)
                if x in current_A:
                    continue
                new_A = current_A | {x}
            else:
                # Swap: remove one, add another
                if len(current_A) == 0:
                    continue
                x = random.choice(list(current_A))
                y = random.randint(0, 128)
                if y in current_A:
                    continue
                new_A = (current_A - {x}) | {y}
            
            new_score, new_covered = evaluate(new_A)
            
            if new_covered:
                new_cost = new_score
            else:
                new_cost = new_score + 10000
            
            if current_covered:
                current_cost = current_score
            else:
                current_cost = current_score + 10000
            
            delta = new_cost - current_cost
            
            if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                current_A = new_A
                current_score = new_score
                current_covered = new_covered
                
                if current_covered and current_score < best_score:
                    best_A = set(current_A)
                    best_score = current_score
            
            T *= cooling
        
        return best_A, best_score
    
    best_A = None
    best_score = float('inf')
    
    initials = make_initials()
    
    random.seed(42)
    random.shuffle(initials)
    
    num_starts = len(initials)
    quick_time = start_time + 15
    
    scored_initials = []
    for init in initials:
        if time.time() > quick_time:
            break
        s, c = evaluate(init)
        cost = s if c else s + 10000
        scored_initials.append((cost, init))
    
    scored_initials.sort(key=lambda x: x[0])
    
    for i, (cost, init) in enumerate(scored_initials):
        if time.time() >= deadline - 5:
            break
        remaining_starts = len(scored_initials) - i
        remaining_time = deadline - 5 - time.time()
        if remaining_time <= 0:
            break
        time_alloc = max(2, min(remaining_time / remaining_starts, 20))
        
        seed = 1000 + i
        ls_result, ls_score = local_search(init, time.time() + time_alloc, seed=seed)
        
        if ls_score < best_score:
            best_score = ls_score
            best_A = ls_result
    
    if time.time() < deadline - 10 and best_A is not None:
        num_extra = 3
        for j in range(num_extra):
            if time.time() >= deadline - 5:
                break
            random.seed(2000 + j)
            size = random.randint(40, 90)
            init = set(random.sample(range(0, 129), size))
            for t in TARGETS:
                found = False
                for a in sorted(init):
                    b = t - a
                    if b >= a and b in init:
                        found = True
                        break
                if not found:
                    mid = t // 2
                    if 0 <= mid <= 128:
                        init.add(mid)
                    if t % 2 == 1 and mid + 1 <= 128:
                        init.add(mid + 1)
            
            time_alloc = max(2, (deadline - 5 - time.time()) / (num_extra - j))
            ls_result, ls_score = local_search(init, time.time() + time_alloc, seed=3000 + j)
            
            if ls_score < best_score:
                best_score = ls_score
                best_A = ls_result
    
    if best_A is None or best_score == float('inf'):
        best_A = set(range(0, 129))
        best_score, best_covered = evaluate(best_A)
        if not best_covered:
            best_A = set(range(64, 129))
    
    final_score, final_covered = evaluate(best_A)
    if not final_covered:
        best_A = set(range(0, 129))
        final_score, final_covered = evaluate(best_A)
    
    return sorted(best_A)
