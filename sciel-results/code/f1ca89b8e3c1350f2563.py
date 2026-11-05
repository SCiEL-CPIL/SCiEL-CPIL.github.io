import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    required_diffs = set(range(1, 31))
    
    def is_sidon(S):
        if len(S) != len(set(S)):
            return False
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def get_diffs(S):
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(S[j] - S[i])
        return diffs

    def score_solution(S):
        if not is_sidon(S):
            return (10**9, 10**9, 10**9)
        diffs = get_diffs(S)
        missing = len(required_diffs - diffs)
        max_elem = max(S) if S else 0
        size = len(S)
        return (missing, max_elem, size)

    def better(s1, s2):
        return s1 < s2

    best_solution = None
    best_score = None

    def random_initial_set():
        S = [0]
        max_val = 50
        attempts = 0
        target_size = 8
        while len(S) < target_size and attempts < 500:
            x = random.randint(1, max_val)
            if x not in S:
                trial = sorted(S + [x])
                if is_sidon(trial):
                    S = trial
            attempts += 1
        return S

    while time.time() < deadline:
        # Initialize
        if best_solution is None or random.random() < 0.1:
            current = random_initial_set()
        else:
            current = list(best_solution)
            # Perturb
            if len(current) > 3:
                for _ in range(random.randint(1, 2)):
                    idx = random.randint(0, len(current) - 1)
                    current.pop(idx)
            # Randomly add a random element
            x = random.randint(0, 100)
            if x not in current:
                current.append(x)
            current = sorted(set(current))
            
        if not is_sidon(current):
            current = random_initial_set()
            
        current_score = score_solution(current)
        if best_score is None or better(current_score, best_score):
            best_score = current_score
            best_solution = current
            
        no_improvement = 0
        while time.time() < deadline and no_improvement < 500:
            improved = False
            current_set = set(current)
            best_neighbor = None
            best_neighbor_score = current_score
            
            # 1. Try Swaps (Replace y with x)
            # Sample some y from current
            # Sample some x from candidate range
            
            # To avoid O(N^2) swaps with large N, we limit samples
            # But N is small (<=32)
            
            # Candidate Xs: Small numbers 0..100, and some randoms
            candidate_xs = list(range(0, 101))
            if random.random() < 0.2:
                candidate_xs.extend([random.randint(0, 500) for _ in range(20)])
                
            # If we are not covering all diffs, we might need larger Xs?
            # Usually small Xs are better for minimizing max.
            
            for y in current:
                # Try replacing y with x
                # Optimization: If y is small and x is large, likely worse for max_elem
                # But might cover missing diffs.
                for x in candidate_xs:
                    if x == y:
                        continue
                    if x in current_set:
                        continue
                    trial = [z for z in current if z != y]
                    trial.append(x)
                    trial.sort()
                    if not is_sidon(trial):
                        continue
                    trial_score = score_solution(trial)
                    if better(trial_score, best_neighbor_score):
                        best_neighbor_score = trial_score
                        best_neighbor = trial
            
            # 2. Try Add (if size < 32)
            if len(current) < 32:
                for x in candidate_xs:
                    if x in current_set:
                        continue
                    trial = sorted(current + [x])
                    if not is_sidon(trial):
                        continue
                    trial_score = score_solution(trial)
                    if better(trial_score, best_neighbor_score):
                        best_neighbor_score = trial_score
                        best_neighbor = trial

            # 3. Try Remove (if size > 1)
            if len(current) > 1:
                for i in range(len(current)):
                    trial = current[:i] + current[i+1:]
                    if not is_sidon(trial):
                        continue
                    trial_score = score_solution(trial)
                    if better(trial_score, best_neighbor_score):
                        best_neighbor_score = trial_score
                        best_neighbor = trial

            if best_neighbor is not None:
                current = best_neighbor
                current_score = best_neighbor_score
                improved = True
                if better(current_score, best_score):
                    best_score = current_score
                    best_solution = current
            else:
                no_improvement += 1
                # Kick
                if no_improvement % 50 == 0:
                    # Random restart of local search
                    break

    if best_solution is None:
        return [0, 1, 4, 10, 18, 28, 40, 54, 70, 88] # Fallback valid Sidon?
        # Check: 0,1,4,10,18,28,40,54,70,88
        # Diffs: 1,4,10,18,28,40,54,70,88
        # 4-1=3, 10-4=6...
        # This fallback might not be valid or cover 1..30.
        # Better fallback: Greedy
        a = [0]
        for d in range(1, 31):
             if d in get_diffs(a): continue
             for x in range(0, 1000):
                 if x in a: continue
                 trial = sorted(a + [x])
                 if is_sidon(trial):
                     if d in get_diffs(trial):
                         a = trial
                         break
        return a

    return best_solution
# EVOLVE-BLOCK-END
