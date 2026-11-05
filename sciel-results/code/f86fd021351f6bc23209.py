"""Combined program: beam search over candidate order + multi-move local search polish."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    N = 512
    MAX_A = 511

    # --- Precomputation: prime factors for 1..511 ---
    prime_factors = [set() for _ in range(N)]
    for p in range(2, N):
        # check if p is prime by seeing if it has no smaller factor
        is_p_prime = True
        for q in range(2, p):
            if p % q == 0:
                is_p_prime = False
                break
        if is_p_prime:
            for m in range(p, N, p):
                prime_factors[m].add(p)

    # --- Weights ---
    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)

    # Candidates ordered by weight descending (a = 511, 510, ..., 1)
    candidates = list(range(MAX_A, 0, -1))

    # --- Helpers ---
    def used_primes_of(s):
        up = set()
        for a in s:
            up |= prime_factors[a]
        return up

    def set_score(s):
        return sum(scores[a] for a in s)

    def is_coprime_set(s):
        up = set()
        for a in s:
            if prime_factors[a] & up:
                return False
            up |= prime_factors[a]
        return True

    # --- Beam search over candidates in weight-descending order ---
    # State: (score, frozenset_of_used_primes, tuple_of_chosen)
    beam_width = 4000
    states = [(0.0, frozenset(), ())]

    best_score = 0.0
    best_chosen = ()

    for a in candidates:
        if time.time() > deadline:
            break

        w_a = scores[a]
        fac_a = prime_factors[a]
        new_states = []

        for (sc, up, ch) in states:
            # Exclude a
            new_states.append((sc, up, ch))
            # Include a if compatible
            if not (fac_a & up):
                nsc = sc + w_a
                new_states.append((nsc, up | fac_a, ch + (a,)))
                if nsc > best_score:
                    best_score = nsc
                    best_chosen = ch + (a,)

        # Prune to top beam_width by score
        new_states.sort(key=lambda x: x[0], reverse=True)
        states = new_states[:beam_width]

    # --- Local search polish on the best found set ---
    current = list(best_chosen)
    current_set = set(current)
    current_score = best_score

    def find_best_add(used, exclude):
        for a in candidates:
            if a in exclude:
                continue
            if not (prime_factors[a] & used):
                return a
        return None

    def hill_climb(max_iters=20000):
        nonlocal current, current_set, current_score
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False

            up = used_primes_of(current)

            # Move 1: add
            add = find_best_add(up, current_set)
            if add is not None:
                current.append(add)
                current_set.add(add)
                current_score += scores[add]
                improved = True
                continue

            # Move 2: 1-for-1 swap
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i + 1:]
                rem_set = set(remaining)
                rem_score = current_score - scores[removed]
                rem_up = used_primes_of(remaining)

                add = find_best_add(rem_up, rem_set)
                if add is not None:
                    ns = rem_score + scores[add]
                    if ns > current_score + 1e-15:
                        current = remaining + [add]
                        current_set = rem_set | {add}
                        current_score = ns
                        improved = True
                        break
            if improved:
                continue

            # Move 3: 1-for-2 swap (remove one, add two)
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i + 1:]
                rem_set = set(remaining)
                rem_score = current_score - scores[removed]
                rem_up = used_primes_of(remaining)

                # Collect valid candidates (coprime to rem_up, not in rem_set)
                valid = []
                for a in candidates:
                    if a in rem_set:
                        continue
                    if prime_factors[a] & rem_up:
                        continue
                    valid.append(a)
                    if len(valid) > 60:
                        break

                best_pair_score = -1.0
                best_pair = None
                for xi in range(len(valid)):
                    a = valid[xi]
                    fa = prime_factors[a]
                    for xj in range(xi + 1, len(valid)):
                        b = valid[xj]
                        if fa & prime_factors[b]:
                            continue
                        s = rem_score + scores[a] + scores[b]
                        if s > best_pair_score:
                            best_pair_score = s
                            best_pair = (a, b)

                if best_pair is not None and best_pair_score > current_score + 1e-15:
                    a, b = best_pair
                    current = remaining + [a, b]
                    current_set = rem_set | {a, b}
                    current_score = best_pair_score
                    improved = True
                    break
            if improved:
                continue

            # Move 4: 2-for-1 swap (remove two, add one) — frees up primes
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(current)):
                    if time.time() > deadline:
                        break
                    r1, r2 = current[i], current[j]
                    remaining = [x for x in current if x != r1 and x != r2]
                    rem_set = set(remaining)
                    rem_score = current_score - scores[r1] - scores[r2]
                    rem_up = used_primes_of(remaining)

                    add = find_best_add(rem_up, rem_set)
                    if add is not None:
                        ns = rem_score + scores[add]
                        if ns > current_score + 1e-15:
                            current = remaining + [add]
                            current_set = rem_set | {add}
                            current_score = ns
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue

            # No improving move found
            break

    hill_climb()

    result = sorted(current)

    # Final validation: pairwise coprime
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback to greedy
                chosen = []
                for a in range(MAX_A, 0, -1):
                    if all(gcd(a, b) == 1 for b in chosen):
                        chosen.append(a)
                return chosen

    return result
# EVOLVE-BLOCK-END
