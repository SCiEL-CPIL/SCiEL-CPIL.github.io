import sys
import time

def construct():
    """
    Finds a Sidon set containing 0, covering differences 1..30, 
    with the minimum possible largest element.
    """
    required_diffs = set(range(1, 31))
    
    def backtrack(D, k):
        """
        Tries to find a Sidon set of size k in [0, D] covering 1..30.
        Forces A[0]=0 and A[k-1]=D to break symmetry.
        """
        A = [0] * k
        A[0] = 0
        A[-1] = D
        diffs = set([D]) # Difference between 0 and D
        covered = required_diffs.intersection(diffs)
        
        def solve(idx, last_val):
            if idx == k - 1:
                if covered == required_diffs:
                    return A[:]
                return None
            
            start_val = last_val + 1
            end_val = D - (k - 1 - idx)
            
            if start_val > end_val:
                return None
                
            candidates = []
            for x in range(start_val, end_val + 1):
                ok = True
                new_diffs = []
                for a in A[:idx]:
                    d = x - a
                    if d in diffs:
                        ok = False
                        break
                    new_diffs.append(d)
                
                if ok:
                    d_end = D - x
                    if d_end in diffs:
                        ok = False
                    else:
                        new_diffs.append(d_end)
                        
                    if ok:
                        new_cov = len(required_diffs.intersection(new_diffs) - covered)
                        candidates.append((new_cov, x, new_diffs))
            
            candidates.sort(key=lambda t: -t[0])
            
            for _, x, new_diffs in candidates:
                A[idx] = x
                old_covered = set(covered)
                
                diffs.update(new_diffs)
                covered.update(required_diffs.intersection(new_diffs))
                
                res = solve(idx + 1, x)
                if res is not None:
                    return res
                
                A[idx] = 0
                diffs.difference_update(new_diffs)
                covered = old_covered
                
            return None
        
        return solve(1, 0)

    def find_set_for_D(D, min_k=9, max_k=12):
        for k in range(min_k, max_k + 1):
            if k * (k - 1) // 2 > D:
                break
            result = backtrack(D, k)
            if result is not None:
                return result
        return None

    start_time = time.time()
    timeout = 110
    
    for D in range(36, 1000):
        if time.time() - start_time > timeout:
            return greedy_construct()
        
        result = find_set_for_D(D)
        if result is not None:
            return sorted(result)
            
    return greedy_construct()

def greedy_construct():
    """Fallback greedy constructor."""
    a = [0]
    target = set(range(1, 31))
    current_diffs = set()
    
    while not target.issubset(current_diffs):
        best_x = None
        best_score = -1
        max_a = max(a)
        # Search range: 1 to max_a + 100
        for x in range(1, max_a + 101):
            if x in a:
                continue
            trial = a + [x]
            trial.sort()
            # Check Sidon
            valid = True
            trial_diffs = set()
            for i in range(len(trial)):
                for j in range(i + 1, len(trial)):
                    d = trial[j] - trial[i]
                    if d in trial_diffs:
                        valid = False
                        break
                    trial_diffs.add(d)
            if not valid:
                continue
            new_diffs = trial_diffs - current_diffs
            score = len(new_diffs & target)
            if score > best_score:
                best_score = score
                best_x = x
                if best_score == len(target - current_diffs):
                    break
        if best_x is None:
            x = max_a + 1
            while True:
                trial = a + [x]
                trial.sort()
                valid = True
                trial_diffs = set()
                for i in range(len(trial)):
                    for j in range(i + 1, len(trial)):
                        d = trial[j] - trial[i]
                        if d in trial_diffs:
                            valid = False
                            break
                        trial_diffs.add(d)
                if valid:
                    best_x = x
                    break
                x += 1
        
        a.append(best_x)
        # Update current_diffs
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                current_diffs.add(a[j] - a[i])
        
    return a

sys.setrecursionlimit(10000)
