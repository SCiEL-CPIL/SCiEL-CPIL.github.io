import math
import time
import random

def score(U):
    """Compute the score for a finite set U containing 0."""
    if not U or 0 not in U:
        return 0
    U = sorted(set(U))
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    
    n_sums = len(sums)
    n_diffs = len(diffs)
    
    if n_sums == 0:
        return 0
        
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
        
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def _fast_score(U):
    """Optimized score calculation using sets."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    sumset = set()
    diffset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
            diffset.add(a - b)
            
    if not sumset:
        return 0.0
        
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

class SetOptimizer:
    def __init__(self, U):
        self.U = sorted(set(U))
        self.n = len(self.U)
        self.sum_counts = {}
        self.diff_counts = {}
        self._update_counts()
        self.current_score = self._calc_score()

    def _update_counts(self):
        self.sum_counts = {}
        self.diff_counts = {}
        for a in self.U:
            for b in self.U:
                s = a + b
                d = a - b
                self.sum_counts[s] = self.sum_counts.get(s, 0) + 1
                self.diff_counts[d] = self.diff_counts.get(d, 0) + 1

    def _calc_score(self):
        n_sums = len(self.sum_counts)
        n_diffs = len(self.diff_counts)
        if n_sums == 0:
            return 0.0
        ratio = n_diffs / n_sums
        if ratio <= 1.0:
            return 1.0
        max_u = max(self.U)
        if max_u == 0:
            return 1.0
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

    def _remove_element_impl(self, val):
        for b in self.U:
            s = val + b
            d = val - b
            self.sum_counts[s] -= 1
            if self.sum_counts[s] == 0:
                del self.sum_counts[s]
            self.diff_counts[d] -= 1
            if self.diff_counts[d] == 0:
                del self.diff_counts[d]
            
            s2 = b + val
            d2 = b - val
            self.sum_counts[s2] -= 1
            if self.sum_counts[s2] == 0:
                del self.sum_counts[s2]
            self.diff_counts[d2] -= 1
            if self.diff_counts[d2] == 0:
                del self.diff_counts[d2]

    def _add_element_impl(self, val):
        for b in self.U:
            s = val + b
            d = val - b
            self.sum_counts[s] = self.sum_counts.get(s, 0) + 1
            self.diff_counts[d] = self.diff_counts.get(d, 0) + 1
            
            s2 = b + val
            d2 = b - val
            self.sum_counts[s2] = self.sum_counts.get(s2, 0) + 1
            self.diff_counts[d2] = self.diff_counts.get(d2, 0) + 1

    def try_move(self, idx, new_val):
        old_val = self.U[idx]
        if old_val == new_val:
            return False
        
        # Check duplicate
        if new_val in self.U:
            if self.U.index(new_val) != idx:
                return False
            
        self._remove_element_impl(old_val)
        
        self.U[idx] = new_val
        self.U.sort()
        
        self._add_element_impl(new_val)
        
        new_score = self._calc_score()
        
        if new_score > self.current_score:
            self.current_score = new_score
            return True
        else:
            self.U[idx] = old_val
            self.U.sort()
            self._remove_element_impl(new_val)
            self._add_element_impl(old_val)
            return False

def _local_search(U, deadline):
    opt = SetOptimizer(U)
    
    while time.time() < deadline:
        improved = False
        max_u = max(opt.U)
        
        # Try moving each non-zero element
        for i in range(1, len(opt.U)):
            # Small steps
            for delta in range(-3, 4):
                new_val = opt.U[i] + delta
                if new_val < 0:
                    continue
                if opt.try_move(i, new_val):
                    improved = True
                    break
            if improved:
                break
                
            # Random jump
            if not improved:
                new_val = random.randint(1, max_u + 10)
                if opt.try_move(i, new_val):
                    improved = True
                    break
                    
        if not improved:
            # Try adding a new element?
            # Or just break
            break
            
    return opt.U, opt.current_score

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # Known optimal Golomb rulers
    known_gr = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 22],
        [0, 1, 4, 10, 12, 17, 22, 29],
        [0, 1, 4, 10, 12, 17, 22, 29, 36],
        [0, 1, 4, 10, 18, 23, 31, 40, 52, 59],
    ]
    
    candidates = list(known_gr)
    
    # Random sets
    random.seed(42)
    for _ in range(20):
        n = random.randint(5, 12)
        U = [0]
        current_max = 0
        for _ in range(n-1):
            next_val = current_max + random.randint(1, 15)
            U.append(next_val)
            current_max = next_val
        candidates.append(sorted(U))
        
    # Evaluate initial candidates
    for U in candidates:
        if time.time() > deadline: break
        s = _fast_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
            
    # Local search on top candidates
    # Sort candidates by score descending
    scored_candidates = [( _fast_score(U), U) for U in candidates]
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    for s, U in scored_candidates[:5]:
        if time.time() > deadline: break
        U_opt, s_opt = _local_search(U, deadline)
        if s_opt > best_score:
            best_score = s_opt
            best_set = U_opt[:]
            
    return sorted(set(best_set))
