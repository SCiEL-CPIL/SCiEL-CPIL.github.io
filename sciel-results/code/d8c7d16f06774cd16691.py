import time
import random
import itertools

DEADLINE = 140.0  # safety margin before 150s kill


def construct(q, n, radius):
    start = time.time()
    if q != 3 or n != 6 or radius != 1:
        # fallback: direct sum for general case
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # All 729 words
    all_words = [list(w) for w in itertools.product(range(3), repeat=6)]
    N = len(all_words)

    # Precompute: for each word, the set of words within distance <= 1 (including itself)
    # Use integer representation for speed
    def to_int(w):
        v = 0
        for i in range(6):
            v = v * 3 + w[i]
        return v

    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[to_int(w)] = i

    # Precompute neighbor sets (distance <= 1)
    neighbors = []
    for w in all_words:
        nb = set([to_int(w)])
        for pos in range(6):
            orig = w[pos]
            for val in range(3):
                if val != orig:
                    w2 = w[:]
                    w2[pos] = val
                    nb.add(to_int(w2))
        neighbors.append(nb)

    # Initial solution: tetracode x torus-2D
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    torus_centers = [[0, 0], [1, 1], [2, 2]]
    init_solution = [h + t for h in tetracode for t in torus_centers]
    init_set = set(to_int(w) for w in init_solution)

    # Verify coverage
    def is_covered(sol_set):
        for i in range(N):
            covered = False
            for nb in neighbors[i]:
                if nb in sol_set:
                    covered = True
                    break
            if not covered:
                return False
        return True

    # Greedy: start from empty, add words that cover most uncovered
    def greedy_cover():
        uncovered = set(range(N))
        sol = []
        # Precompute for each word index, which word indices it covers
        covers = []
        for i in range(N):
            covers.append(set(neighbors[i]))
        while uncovered:
            best_i = -1
            best_count = -1
            # Sample if too many to check all
            if len(uncovered) > 10000:
                candidates = list(uncovered)[:200]
            else:
                candidates = list(uncovered)
            for i in candidates:
                c = len(covers[i] & uncovered)
                if c > best_count:
                    best_count = c
                    best_i = i
            if best_i == -1:
                break
            sol.append(all_words[best_i])
            sol_set_i = set(neighbors[best_i])
            uncovered -= sol_set_i
        return sol

    # Start with the 27-word solution, try to improve with local search
    best_sol = init_solution[:]
    best_size = len(best_sol)

    # Local search: try removing a word and see if coverage still holds
    def try_remove(sol, idx_to_remove):
        sol_set = set(to_int(w) for w in sol)
        w_remove = to_int(sol[idx_to_remove])
        sol_set.remove(w_remove)
        # Check if all words still covered
        # Only need to check words that were ONLY covered by w_remove
        # i.e., words in neighbors[idx_to_remove] that aren't covered by any other word in sol
        # This is expensive; do full check for simplicity with early exit
        for i in range(N):
            covered = False
            for nb in neighbors[i]:
                if nb in sol_set:
                    covered = True
                    break
            if not covered:
                return False
        return True

    # Iterative removal
    changed = True
    sol = best_sol[:]
    while changed:
        changed = False
        for i in range(len(sol)):
            if time.time() > DEADLINE:
                break
            if try_remove(sol, i):
                sol.pop(i)
                changed = True
                best_size = len(sol)
        if time.time() > DEADLINE:
            break

    best_sol = sol[:]
    best_size = len(sol)

    # Try swap-based local search: remove one, add another
    # For each word in solution, try replacing with a word not in solution
    # that covers some currently-uncovered-by-that-word words
    # This is complex; try a simpler approach: random restarts with greedy

    # Try multiple random initializations with greedy + removal
    random.seed(42)
    for restart in range(50):
        if time.time() > DEADLINE - 5:
            break
        # Random subset of size ~73
        size = best_size - restart // 10  # try slightly smaller
        if size < 60:
            size = 60
        # Greedy from random start
        sol = []
        sol_set = set()
        # Add random words
        for _ in range(min(size, 30)):
            idx = random.randint(0, N - 1)
            if to_int(all_words[idx]) not in sol_set:
                sol.append(all_words[idx])
                sol_set.add(to_int(all_words[idx]))
        # Greedily add to cover
        uncovered = set(range(N))
        for i in range(N):
            if any(nb in sol_set for nb in neighbors[i]):
                uncovered.discard(i)
        covers = [set(neighbors[i]) for i in range(N)]
        while uncovered and len(sol) < best_size:
            if time.time() > DEADLINE - 2:
                break
            best_i = -1
            best_count = -1
            # Sample from uncovered
            sample = list(uncovered)
            random.shuffle(sample)
            for i in sample[:100]:
                c = len(covers[i] & uncovered)
                if c > best_count:
                    best_count = c
                    best_i = i
            if best_i == -1 or best_count == 0:
                break
            sol.append(all_words[best_i])
            sol_set.add(to_int(all_words[best_i]))
            uncovered -= covers[best_i]
        # Removal pass
        changed = True
        while changed:
            changed = False
            for i in range(len(sol)):
                if time.time() > DEADLINE - 1:
                    break
                if try_remove(sol, i):
                    sol.pop(i)
                    changed = True
        if len(sol) < best_size:
            best_sol = sol[:]
            best_size = len(sol)

    return best_sol
