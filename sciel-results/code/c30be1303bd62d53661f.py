"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    def is_sidon(s):
        d = diffs(s)
        return len(set(d)) == len(d)
    
    def covers_1_to_30(s):
        d = set(diffs(s))
        return all(i in d for i in range(1, 31))
    
    # Greedy: realize each missing difference d with a new element next to an existing one.
    for d in range(1, 31):
        if time.time() > deadline:
            break
        if d in diffs(sorted(a)):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                if time.time() > deadline:
                    break
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    
    # Local search: try to reduce the largest element
    a = sorted(a)
    best_a = list(a)
    best_max = max(a)
    
    # Try to replace the largest element with a smaller one
    # Or try to shift elements down
    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        a = sorted(best_a)
        # Try removing the largest element and finding a replacement
        if len(a) > 1:
            largest = a[-1]
            # Try smaller values for the largest element
            for new_val in range(largest - 1, -1, -1):
                if new_val in a[:-1]:
                    continue
                trial = sorted(a[:-1] + [new_val])
                if is_sidon(trial) and covers_1_to_30(trial):
                    best_a = trial
                    best_max = max(trial)
                    improved = True
                    break
        
        # Try shifting non-zero elements down by 1 if possible
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
        
        if not improved:
            break
    
    return best_a
# EVOLVE-BLOCK-END
