import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    N = 512

    # Precompute coprime bitmask: coprime_mask[i] has bit j set if gcd(i,j)==1
    coprime_mask = [0] * N
    for i in range(1, N):
        mask = 0
        for j in range(1, N):
            if i != j and gcd(i, j) == 1:
                mask |= (1 << j)
        coprime_mask[i] = mask

    def is_coprime_to_all(a, S):
        """Check if a is coprime to all elements in S using bitmask."""
        if not S:
            return True
        # Build mask of S
        mask = 0
        for b in S:
            mask |= (1 << b)
        return (coprime_mask[a] & mask) == mask

    def score(S):
        return sum(1.0 / (512.0 - a) for a in S)

    def greedy_select(candidates, background):
        """Greedy select from candidates, must be coprime to background and each other."""
        selected = []
        selected_mask = 0
        for a in candidates:
            # Check coprime to background
            bg_mask = 0
            for b in background:
                bg_mask |= (1 << b)
            if (coprime_mask[a] & bg_mask) != bg_mask:
                continue
            # Check coprime to selected
            if selected and (coprime_mask[a] & selected_mask) != selected_mask:
                continue
            selected.append(a)
            selected_mask |= (1 << a)
        return selected

    # Multiple greedy initializations with different orderings
    best_S = None
    best_score = -1.0

    orderings = [
        list(range(511, 0, -1)),  # descending
        list(range(1, 512)),       # ascending
        list(range(511, 0, -2)) + list(range(510, 0, -2)),  # odd first, then even
        list(range(510, 0, -2)) + list(range(511, 0, -2)),  # even first, then odd
    ]

    for ordering in orderings:
        S = []
        S_mask = 0
        for a in ordering:
            if (coprime_mask[a] & S_mask) == S_mask:
                S.append(a)
                S_mask |= (1 << a)
        s = score(S)
        if s > best_score:
            best_score = s
            best_S = S[:]

    current_S = best_S[:]
    current_score = best_score
    current_mask = 0
    for a in current_S:
        current_mask |= (1 << a)

    improved = True
    all_candidates = list(range(511, 0, -1))

    while improved and time.time() < deadline:
        improved = False

        # Try removing each element and re-greedy-filling
        for a in list(current_S):
            if time.time() >= deadline:
                break
            bg = [b for b in current_S if b != a]
            bg_mask = current_mask & ~(1 << a)

            valid_candidates = [c for c in all_candidates
                               if c not in current_S and (coprime_mask[c] & bg_mask) == bg_mask]
            B = greedy_select(valid_candidates, bg)

            if B:
                new_score = current_score - (1.0 / (512.0 - a)) + score(B)
                if new_score > current_score + 1e-9:
                    current_S = bg + B
                    current_score = new_score
                    current_mask = bg_mask
                    for b in B:
                        current_mask |= (1 << b)
                    improved = True
                    break

        # Try adding a single element
        if not improved and time.time() < deadline:
            for c in all_candidates:
                if c in current_S:
                    continue
                if (coprime_mask[c] & current_mask) == current_mask:
                    new_score = current_score + 1.0 / (512.0 - c)
                    if new_score > current_score + 1e-9:
                        current_S.append(c)
                        current_mask |= (1 << c)
                        current_score = new_score
                        improved = True
                        break

        # Try swap: remove a, add c
        if not improved and time.time() < deadline:
            for a in list(current_S):
                if time.time() >= deadline:
                    break
                bg_mask = current_mask & ~(1 << a)
                bg = [b for b in current_S if b != a]
                for c in all_candidates:
                    if c in current_S:
                        continue
                    if (coprime_mask[c] & bg_mask) == bg_mask:
                        new_score = current_score - (1.0 / (512.0 - a)) + 1.0 / (512.0 - c)
                        if new_score > current_score + 1e-9:
                            current_S = bg + [c]
                            current_score = new_score
                            current_mask = bg_mask | (1 << c)
                            improved = True
                            break
                if improved:
                    break

    return current_S
