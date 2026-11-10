import time
import random
from math import gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin vs 60s/90s kill
    N = 128

    # Precompute gcd matrix
    gcd_mat = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute conflict pairs: for each v, list of (a,b) such that {a,b,v} is a bad triple
    conflict_pairs = [[] for _ in range(N)]
    # Also collect all bad triples (as index tuples) for the ILP
    bad_triples = []
    for a in range(N):
        for b in range(a+1, N):
            ga = gcd_mat[a+1][b+1]
            for c in range(b+1, N):
                if gcd_mat[a+1][c+1] == ga and gcd_mat[b+1][c+1] == ga:
                    # Triple (a,b,c) is bad
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
                    bad_triples.append((a, b, c))

    def can_add(v, selected_set):
        for (a, b) in conflict_pairs[v]:
            if a in selected_set and b in selected_set:
                return False
        return True

    def greedy(order, selected_set=None):
        if selected_set is None:
            selected_set = set()
        selected = []
        for v in order:
            if v not in selected_set and can_add(v, selected_set):
                selected.append(v)
                selected_set.add(v)
        return selected, selected_set

    def local_improve(selected, selected_set, deadline):
        improved = True
        while improved:
            improved = False
            for v in range(N):
                if v in selected_set:
                    continue
                if can_add(v, selected_set):
                    selected.append(v)
                    selected_set.add(v)
                    improved = True
        return selected, selected_set

    def local_search(selected, selected_set, deadline, rng):
        """Swaps and kicks to improve."""
        for _ in range(50):
            if time.time() > deadline:
                break
            # Try to add any
            improved = False
            for v in range(N):
                if v not in selected_set and can_add(v, selected_set):
                    selected.append(v)
                    selected_set.add(v)
                    improved = True
            if improved:
                continue
            # Try 1-for-1 swaps
            swapped = False
            out_list = list(selected_set)
            rng.shuffle(out_list)
            for x_out in out_list:
                if time.time() > deadline:
                    return selected, selected_set
                # Try removing x_out and adding some x_in
                selected_set.discard(x_out)
                if x_out in selected:
                    selected.remove(x_out)
                success = False
                for x_in in range(N):
                    if x_in in selected_set:
                        continue
                    if can_add(x_in, selected_set):
                        selected_set.add(x_in)
                        selected.append(x_in)
                        success = True
                        break
                if success:
                    swapped = True
                    break
                else:
                    # restore
                    selected_set.add(x_out)
                    selected.append(x_out)
            if swapped:
                continue
            # Kick: remove k random, refill
            if len(selected) > 5:
                k = min(len(selected), rng.randint(2, 4))
                to_remove = rng.sample(selected, k)
                for x in to_remove:
                    selected_set.discard(x)
                    selected.remove(x)
                order = list(range(N))
                rng.shuffle(order)
                greedy(order, selected_set)
                # Re-sync selected list
                selected = sorted(selected_set)
                local_improve(selected, selected_set, deadline)

        return selected, selected_set

    best = []
    best_set = set()
    best_size = 0
    rng = random.Random(42)

    # Phase 1: Deterministic + random greedy restarts (fast, reliable)
    orders = [
        list(range(N)),
        list(range(N-1, -1, -1)),
        sorted(range(N), key=lambda x: len(conflict_pairs[x])),
        sorted(range(N), key=lambda x: -len(conflict_pairs[x])),
    ]
    for order in orders:
        if time.time() > deadline:
            break
        selected, selected_set = greedy(order)
        selected, selected_set = local_improve(selected, selected_set, deadline)
        if len(selected) > best_size:
            best_size = len(selected)
            best = list(selected)
            best_set = set(selected)

    # Phase 2: Random restarts with local search
    max_restarts = 400
    for i in range(max_restarts):
        if time.time() > deadline:
            break
        order = list(range(N))
        rng.shuffle(order)
        selected, selected_set = greedy(order)
        selected, selected_set = local_improve(selected, selected_set, deadline)
        selected, selected_set = local_search(selected, selected_set, deadline, rng)
        if len(selected) > best_size:
            best_size = len(selected)
            best = list(selected)
            best_set = set(selected)

    # Phase 3: ILP via scipy.optimize.milp (structural: exact/upper-bound)
    # Variables x[0..N-1] in {0,1}. Maximize sum x.
    # Constraint per bad triple (a,b,c): x[a]+x[b]+x[c] <= 2.
    if time.time() < deadline - 8 and len(bad_triples) > 0:
        try:
            c_obj = -np.ones(N)  # minimize -sum => maximize sum
            A = []
            for (a, b, c) in bad_triples:
                row = np.zeros(N)
                row[a] = 1.0
                row[b] = 1.0
                row[c] = 1.0
                A.append(row)
            A = np.array(A) if A else np.zeros((1, N))
            lb = np.full(len(bad_triples), 0.0)
            ub = np.full(len(bad_triples), 2.0)
            cons = LinearConstraint(A, lb, ub)
            bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
            integrality = np.ones(N)
            res = milp(c=c_obj, constraints=cons, integrality=integrality, bounds=bounds,
                       options={"time_limit": max(1, deadline - time.time() - 3)})
            if res.x is not None:
                sol = [i for i in range(N) if res.x[i] > 0.5]
                # Validate (defensive): ensure no bad triple fully present
                sol_set = set(sol)
                valid = True
                for (a, b, c) in bad_triples:
                    if a in sol_set and b in sol_set and c in sol_set:
                        valid = False
                        break
                if valid and len(sol) > best_size:
                    best_size = len(sol)
                    best = list(sol)
                    best_set = set(sol)
                # Also polish the ILP solution with local search if it's a good seed
                if valid and len(sol) > 0:
                    sel2, set2 = local_search(list(sol), set(sol), deadline, rng)
                    if len(sel2) > best_size:
                        best_size = len(sel2)
                        best = list(sel2)
                        best_set = set(sel2)
        except Exception:
            pass

    if not best:
        best = [0]

    return sorted([x+1 for x in best])
