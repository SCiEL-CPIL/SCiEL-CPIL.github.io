import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    target_lo = 128
    target_hi = 256
    all_elements = list(range(0, 129))
    
    # Start with seed [64, 128]
    A = set(range(64, 129))
    
    def compute_counts(A_set):
        counts = [0] * (target_hi + 1)
        A_list = sorted(A_set)
        for i in range(len(A_list)):
            ai = A_list[i]
            for j in range(i, len(A_list)):
                s = ai + A_list[j]
                if target_lo <= s <= target_hi:
                    counts[s] += 1
        return counts
    
    counts = compute_counts(A)
    current_peak = max(counts[target_lo:target_hi+1])
    best_peak = current_peak
    best_A = set(A)
    
    def apply_add(counts_arr, A_set, x):
        changes = []
        A_list = sorted(A_set)
        for y in A_list:
            if y >= x:
                s = x + y
                if target_lo <= s <= target_hi:
                    counts_arr[s] += 1
                    changes.append((s, 1))
        A_set.add(x)
        return changes
    
    def apply_remove(counts_arr, A_set, x):
        changes = []
        A_set.discard(x)
        A_list = sorted(A_set)
        for y in A_list:
            if y >= x:
                s = x + y
                if target_lo <= s <= target_hi:
                    counts_arr[s] -= 1
                    changes.append((s, -1))
        s = 2 * x
        if target_lo <= s <= target_hi:
            counts_arr[s] -= 1
            changes.append((s, -1))
        return changes
    
    def undo(counts_arr, changes):
        for s, d in changes:
            counts_arr[s] -= d
    
    def check_coverage(counts_arr):
        for s in range(target_lo, target_hi + 1):
            if counts_arr[s] <= 0:
                return False
        return True
    
    def get_peak(counts_arr):
        return max(counts_arr[target_lo:target_hi+1])
    
    rng = random.Random(123)
    no_improve = 0
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try all removals in order
        A_list = sorted(A)
        for x in A_list:
            if time.time() >= deadline:
                break
            if x not in A:
                continue
            changes = apply_remove(counts, A, x)
            if check_coverage(counts):
                new_peak = get_peak(counts)
                if new_peak < current_peak:
                    current_peak = new_peak
                    improved = True
                    no_improve = 0
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = set(A)
                else:
                    undo(counts, changes)
                    A.add(x)
            else:
                undo(counts, changes)
                A.add(x)
        
        if improved:
            continue
        
        # Phase 2: Try swaps (random)
        candidates = [x for x in all_elements if x not in A]
        if candidates and A:
            A_list = list(A)
            for _ in range(300):
                if time.time() >= deadline:
                    break
                x = rng.choice(A_list)
                y = rng.choice(candidates)
                changes1 = apply_remove(counts, A, x)
                changes2 = apply_add(counts, A, y)
                if check_coverage(counts):
                    new_peak = get_peak(counts)
                    if new_peak < current_peak:
                        current_peak = new_peak
                        improved = True
                        no_improve = 0
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A = set(A)
                    else:
                        undo(counts, changes2)
                        undo(counts, changes1)
                        A.add(x)
                else:
                    undo(counts, changes2)
                    undo(counts, changes1)
                    A.add(x)
                if improved:
                    break
        
        if improved:
            continue
        
        # Phase 3: Perturbation - add random elements to escape local optima
        no_improve += 1
        if no_improve >= 20:
            candidates = [x for x in all_elements if x not in A]
            if candidates:
                num_add = min(rng.randint(2, 5), len(candidates))
                for _ in range(num_add):
                    if time.time() >= deadline:
                        break
                    x = rng.choice(candidates)
                    if x not in A:
                        changes = apply_add(counts, A, x)
                        current_peak = get_peak(counts)
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A = set(A)
            no_improve = 0
    
    return sorted(best_A)
