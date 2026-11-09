import time

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before the 150 s wall-clock kill


def _greedy():
    """Valid Sidon set containing 0 that realizes differences 1..30.

    Guarantees a correct (if not minimal) answer and supplies the diameter
    upper bound that caps the exact search below.
    """
    a = [0]

    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]

    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                break
        else:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a


def construct():
    # This is a minimum-diameter Golomb-ruler (Sidon set) problem:
    #   * contain 0,
    #   * every difference d in 1..30 appears exactly once,
    #   * all pairwise differences distinct (Sidon),
    #   * minimise the largest element.
    #
    # A set of n marks has n*(n-1)/2 differences, so we need n >= 9
    # (9 marks -> 36 diffs: 30 for 1..30, 6 spare).
    #
    # Exact method: iterative deepening on the diameter M.  For each M run a
    # *complete* backtracking DFS that places marks in [0, M] (0 fixed),
    # covering differences 1..30 while keeping the Sidon property.  The first
    # M with a solution is provably optimal.  A valid greedy set is the
    # guaranteed fallback and also provides the upper bound U that caps M.
    #
    # Pruning:
    #   * Sidon via a set of used differences (every pairwise diff distinct).
    #   * bitmask of which of 1..30 are already covered.
    #   * feasibility: the remaining required differences must be realisable
    #     by the differences the still-placeable marks can still create.

    FULL = (1 << 30) - 1  # bits 0..29 represent differences 1..30

    greedy = _greedy()
    U = max(greedy)

    def solve(M):
        marks = [0]
        used = set()  # every difference created so far

        def feasible(need):
            last = marks[-1]
            k = len(marks)
            addable = M - last
            if addable <= 0:
                return need == 0
            # The j-th additional mark (j = 1..addable) adds (k + j) new
            # differences (one per existing mark, one per earlier added mark).
            total_new = 0
            for j in range(1, addable + 1):
                total_new += k + j
            return need <= total_new

        def dfs(mask):
            if mask == FULL:
                return True
            if time.time() > _DEADLINE:
                return False
            need = bin(FULL ^ mask).count("1")
            if not feasible(need):
                return False
            last = marks[-1]

            # Candidate next marks, ordered by how many new small differences
            # they cover (descending) so a solution is found as fast as
            # possible.  Ordering never prunes a valid branch, so the search
            # stays complete for the "no solution" verdict.
            cands = []
            for x in range(last + 1, M + 1):
                ok = True
                cov = 0
                for m in marks:
                    d = x - m
                    if d in used:
                        ok = False
                        break
                    if 1 <= d <= 30:
                        bit = 1 << (d - 1)
                        if mask & bit:
                            ok = False
                            break
                        cov += 1
                if ok:
                    cands.append((cov, x))
            cands.sort(reverse=True)

            for _cov, x in cands:
                marks.append(x)
                for m in marks[:-1]:
                    used.add(x - m)
                mask_new = mask
                for m in marks[:-1]:
                    d = x - m
                    if 1 <= d <= 30:
                        mask_new |= 1 << (d - 1)
                if dfs(mask_new):
                    return True
                for m in marks[:-1]:
                    used.discard(x - m)
                marks.pop()
            return False

        if dfs(0):
            return list(marks)
        return None

    # Lower bound is 30 (difference 30 must appear); upper bound is the
    # greedy diameter U.  First solvable M is the optimum.
    for M in range(30, U + 1):
        if time.time() > _DEADLINE - 3:
            break
        res = solve(M)
        if res is not None:
            return res

    # Search did not finish in time: return the valid greedy construction.
    return greedy


BET: {mechanism: "Iterative deepening on the diameter M with a complete backtracking DFS (bitmask of covered 1..30 differences + Sidon via a used-difference set) plus a difference-feasibility prune finds the provably minimum-diameter Golomb ruler; a valid greedy set is the guaranteed fallback and also supplies the upper bound that caps the deepening, so the exact search both proves optimality and cannot return an invalid set.", q: 0.6, delta: [0.0, 0.5]}
