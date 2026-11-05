import math
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin

    # Precompute weights
    weights = [1.0 / (512 - a) for a in range(512)]

    # Precompute coprime matrix for fast lookup
    coprime = [[False] * 512 for _ in range(512)]
    for a in range(1, 512):
        for b in range(a + 1, 512):
            coprime[a][b] = (math.gcd(a, b) == 1)
            coprime[b][a] = coprime[a][b]
    for a in range(1, 512):
        coprime[a][a] = True  # self-coprime (not used but safe)

    # Greedy initialization: pick largest weights first
    chosen = []
    for a in range(511, 0, -1):
        if all(coprime[a][b] for b in chosen):
            chosen.append(a)

    best_set = chosen[:]
    best_score = sum(weights[a] for a in chosen)

    def is_pairwise_coprime(s):
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if not coprime[s[i]][s[j]]:
                    return False
        return True

    def greedy_fill(remaining_set, max_iter=512):
        """Fill remaining_set with best available coprime elements."""
        s = list(remaining_set)
        s_set = set(s)
        for b in range(511, 0, -1):
            if b in s_set:
                continue
            if all(coprime[b][c] for c in s):
                s.append(b)
                s_set.add(b)
        return s

    # Local search: try removing one element and re-greedy-filling
    improved = True
    while improved:
        improved = False
        for i, a in enumerate(best_set):
            # Try removing a
            remaining = [x for j, x in enumerate(best_set) if j != i]
            new_set = greedy_fill(remaining)

            new_score = sum(weights[b] for b in new_set)
            if new_score > best_score:
                best_score = new_score
                best_set = new_set
                improved = True
                break

        if time.time() > deadline:
            break

    # Second pass: try swapping one element with another
    improved = True
    while improved:
        improved = False
        for i, a in enumerate(best_set):
            # Try replacing a with each candidate b not in set
            remaining = [x for j, x in enumerate(best_set) if j != i]
            rem_set = set(remaining)
            best_candidate = None
            best_candidate_score = best_score
            for b in range(511, 0, -1):
                if b in rem_set or b == a:
                    continue
                if all(coprime[b][c] for c in remaining):
                    new_set = remaining + [b]
                    new_score = sum(weights[x] for x in new_set)
                    if new_score > best_candidate_score:
                        best_candidate_score = new_score
                        best_candidate = b

            if best_candidate is not None:
                best_set = [best_candidate if x == a else x for x in best_set]
                best_score = best_candidate_score
                improved = True
                break

        if time.time() > deadline:
            break

    return best_set
