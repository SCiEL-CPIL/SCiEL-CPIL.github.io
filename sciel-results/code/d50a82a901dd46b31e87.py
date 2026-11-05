import time
import random

def _max_weak_sidon_size(S):
    """Exact max size of weak-Sidon subset (all pair sums i<=j unique) of S."""
    n = len(S)
    if n <= 1:
        return n
    S_sorted = sorted(S)
    n = len(S_sorted)
    best = [0]
    
    def dfs(idx, current_size, used_sums):
        if current_size > best[0]:
            best[0] = current_size
        if idx == n:
            return
        if current_size + (n - idx) <= best[0]:
            return
        
        can_add = True
        new_sums = []
        for j in range(idx):
            s = S_sorted[idx] + S_sorted[j]
            if s in used_sums:
                can_add = False
                break
            new_sums.append(s)
        if can_add:
            s_diag = 2 * S_sorted[idx]
            if s_diag in used_sums:
                can_add = False
            else:
                new_sums.append(s_diag)
        
        if can_add:
            used_sums.update(new_sums)
            dfs(idx + 1, current_size + 1, used_sums)
            used_sums.difference_update(new_sums)
        
        dfs(idx + 1, current_size, used_sums)
    
    dfs(0, 0, set())
    return best[0]


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 70.0
    random.seed(42)
    
    current = list(range(12))
    current_score = _max_weak_sidon_size(current)
    best = current[:]
    best_score = current_score
    
    while time.time() - start_time < deadline:
        candidate = current[:]
        num_changes = random.randint(1, 3)
        for _ in range(num_changes):
            idx = random.randint(0, 11)
            candidate[idx] = random.randint(0, 128)
        
        if len(set(candidate)) != 12:
            continue
        
        score = _max_weak_sidon_size(candidate)
        if score < best_score:
            best_score = score
            best = candidate[:]
            current = candidate[:]
        elif score == best_score and random.random() < 0.3:
            current = candidate[:]
        elif score <= current_score + 1 and random.random() < 0.15:
            current = candidate[:]
    
    return best
# EVOLVE-BLOCK-END
