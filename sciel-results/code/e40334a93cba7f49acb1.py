"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    N = 512

    # Precompute coprime relationship
    coprime = [[False] * N for _ in range(N)]
    for i in range(1, N):
        for j in range(i, N):
            if gcd(i, j) == 1:
                coprime[i][j] = True
                coprime[j][i] = True

    def score(S):
        return sum(1.0 / (512.0 - a) for a in S)

    def is_coprime_to_all(a, S):
        for b in S:
            if not coprime[a][b]:
                return False
        return True

    def greedy_fill(candidates, background):
        """Greedy fill: pick from candidates (sorted desc) those coprime to background and to each other."""
        selected = []
        for a in candidates:
            if is_coprime_to_all(a, background) and is_coprime_to_all(a, selected):
                selected.append(a)
        return selected

    # 1. Initial greedy: pick largest coprime numbers
    S = []
    for a in range(511, 0, -1):
        if is_coprime_to_all(a, S):
            S.append(a)

    current_score = score(S)
    S_set = set(S)
    all_candidates = list(range(511, 0, -1))

    # 2. Local search: try removing each element and re-greedy-filling the rest
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for a in list(S_set):
            if time.time() >= deadline:
                break
            background = S_set - {a}
            valid_candidates = [c for c in all_candidates if c not in S_set and is_coprime_to_all(c, background)]
            B = greedy_fill(valid_candidates, background)
            if B:
                new_score = current_score - (1.0 / (512.0 - a)) + score(B)
                if new_score > current_score + 1e-9:
                    S_set = background.union(B)
                    current_score = new_score
                    improved = True
                    break

    return list(S_set)
# EVOLVE-BLOCK-END
