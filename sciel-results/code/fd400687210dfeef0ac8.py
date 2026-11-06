"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    def diffs(s):
        """Return all positive differences in sorted set s."""
        n = len(s)
        return [s[j] - s[i] for i in range(n) for j in range(i + 1, n)]
    
    def is_sidon(s):
        """Check if set has all distinct positive differences."""
        d = diffs(s)
        return len(set(d)) == len(d)
    
    def covers_1_to_30(s):
        """Check if all differences 1..30 are present."""
        d = set(diffs(s))
        return all(i in d for i in range(1, 31))
    
    def greedy_construct():
        """Greedy construction: realize each missing difference d with a new element."""
        a = [0]
        for d in range(1, 31):
            if time.time() > deadline:
                break
            if d in diffs(sorted(a)):
                continue
            found = False
            # Try adding x = v +/- d for existing v
            candidates = sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a))
            for x in candidates:
                trial = sorted(a + [x])
                if is_sidon(trial):
                    a = trial
                    found = True
                    break
            if not found:
                # Fallback: add two elements y, y+d in a high range
                y = 2 * max(a) + 3 * d + 1
                while time.time() < deadline:
                    trial = sorted(a + [y, y + d])
                    if is_sidon(trial):
                        a = trial
                        break
                    y += 1
        return a
    
    def local_search(a, max_iterations=5000):
        """Local search to reduce the largest element."""
        a = sorted(a)
        best_a = list(a)
        best_max = max(a)
        no_improve = 0
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            improved = False
            a = sorted(best_a)
            
            # Strategy 1: Try to replace the largest element with a smaller one
            if len(a) > 1:
                largest = a[-1]
                # Try a range of smaller values
                for new_val in range(largest - 1, max(0, largest - 500), -1):
                    if new_val in a[:-1]:
                        continue
                    trial = sorted(a[:-1] + [new_val])
                    if is_sidon(trial) and covers_1_to_30(trial):
                        best_a = trial
                        best_max = max(trial)
                        improved = True
                        break
            
            # Strategy 2: Try shifting individual elements down
            if not improved:
                a = sorted(best_a)
                for i in range(len(a) - 1, 0, -1):
                    if a[i] > 1:
                        new_val = a[i] - 1
                        if new_val not in a:
                            trial = list(a)
                            trial[i] = new_val
                            trial = sorted(trial)
                            if is_sidon(trial) and covers_1_to_30(trial):
                                best_a = trial
                                best_max = max(trial)
                                improved = True
                                break
            
            # Strategy 3: Try swapping two elements
            if not improved:
                a = sorted(best_a)
                for i in range(1, len(a) - 1):
                    for j in range(i + 1, len(a)):
                        if a[i] == a[j]:
                            continue
                        trial = list(a)
                        trial[i], trial[j] = trial[j], trial[i]
                        trial = sorted(trial)
                        if is_sidon(trial) and covers_1_to_30(trial):
                            if max(trial) < best_max:
                                best_a = trial
                                best_max = max(trial)
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                no_improve += 1
                if no_improve >= 50:
                    break
            else:
                no_improve = 0
        
        return best_a
    
    # Step 1: Greedy construction
    a = greedy_construct()
    
    # Step 2: Local search to minimize largest element
    if time.time() < deadline - 10:
        a = local_search(a)
    
    # Step 3: More aggressive local search if time remains
    if time.time() < deadline - 5:
        a = local_search(a, max_iterations=10000)
    
    return a
# EVOLVE-BLOCK-END
