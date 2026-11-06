import time
import random
from itertools import combinations

def construct():
    """
    Find a Sidon set containing 0 with all differences 1..30, minimizing the largest element.
    Uses simulated annealing with a bitset-based conflict check.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Leave 5s margin before 120s kill
    
    # We need a Sidon set S with 0 in S, |S| <= 32, max(S) minimized,
    # and {d in 1..30} ⊆ {y-x : x,y in S, y>x}.
    
    # A Sidon set of size n has at most n(n-1)/2 distinct positive differences.
    # To cover 30 differences, we need n(n-1)/2 >= 30 => n >= 9 (9*8/2=36).
    # So minimum size is 9. Let's aim for small sets first.
    
    def is_sidon(S):
        """Check if S is a Sidon set (all positive differences distinct)."""
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    def covers_all(S):
        """Check if S covers all differences 1..30."""
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                d = S[j] - S[i]
                if 1 <= d <= 30:
                    diffs.add(d)
        return diffs == set(range(1, 31))
    
    def score_set(S):
        """
        Score: lower is better.
        Primary: -1 if not all diffs covered, else 0.
        Secondary: max(S) if valid, else large penalty.
        """
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                d = S[j] - S[i]
                if 1 <= d <= 30:
                    diffs.add(d)
        missing = 30 - len(diffs)
        if missing > 0:
            return (1, max(S) + 1000 * missing)
        else:
            return (0, max(S))
    
    # Start with a greedy construction to get a valid initial solution
    def greedy_init():
        S = [0]
        for d in range(1, 31):
            if d in [y - x for i, x in enumerate(S) for y in S[i+1:]]:
                continue
            # Try to add a single element that creates difference d
            found = False
            for x in sorted(set(v + s for v in S for s in (d, -d) if v + s >= 0) - set(S)):
                trial = sorted(S + [x])
                if is_sidon(trial):
                    S = trial
                    found = True
                    break
            if not found:
                # Add two elements
                y = 2 * max(S) + 3 * d + 1
                while True:
                    trial = sorted(S + [y, y + d])
                    if is_sidon(trial):
                        S = trial
                        break
                    y += 1
        return S
    
    best_S = greedy_init()
    best_score = score_set(best_S)
    
    # Simulated annealing
    # State: a set of integers (as a tuple for hashability)
    # Move: pick a random element, change its value slightly
    # Or: add/remove an element
    
    def random_neighbor(S):
        """Generate a random neighbor of S."""
        S = list(S)
        r = random.random()
        if r < 0.3 and len(S) > 9:
            # Remove a random element (not 0)
            candidates = [i for i, x in enumerate(S) if x != 0]
            if candidates:
                idx = random.choice(candidates)
                S.pop(idx)
        elif r < 0.6 and len(S) < 32:
            # Add a new random element
            max_val = max(S) if S else 0
            new_x = random.randint(1, max_val + 50)
            if new_x not in S:
                S.append(new_x)
        else:
            # Move a random element (not 0)
            candidates = [i for i, x in enumerate(S) if x != 0]
            if candidates:
                idx = random.choice(candidates)
                old_x = S[idx]
                S[idx] = old_x + random.randint(-10, 10)
                if S[idx] < 0:
                    S[idx] = random.randint(1, max(S) + 10)
                # Ensure no duplicates
                if S.count(S[idx]) > 1:
                    S[idx] = old_x
        return tuple(sorted(set(S)))
    
    # Annealing parameters
    T = 1000.0
    T_min = 1.0
    alpha = 0.9995  # Cooling rate
    steps = 0
    max_steps = 2000000
    
    current_S = tuple(sorted(best_S))
    current_score = best_score
    
    while time.time() - start_time < DEADLINE and steps < max_steps:
        steps += 1
        new_S = random_neighbor(current_S)
        new_score = score_set(new_S)
        
        # Accept if better, or with probability exp(-delta/T)
        delta = (new_score[0] - current_score[0]) + (new_score[1] - current_score[1]) / 1000.0
        if delta < 0 or random.random() < (1.0 if T > T_min else 0.0):
            if delta < 0 or (T > T_min and random.random() < min(1.0, max(0.0, -delta / max(T, 1e-10)))):
                current_S = new_S
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_S = list(new_S)
        
        T = max(T_min, T * alpha)
    
    return best_S
