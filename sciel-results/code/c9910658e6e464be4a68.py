"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 256
    squares = set()
    for k in range(2, 23):
        sq = k * k
        if sq <= 2 * N:
            squares.add(sq)

    # Nodes where 2*a is a square are individually invalid
    individually_invalid = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            individually_invalid.add(a)

    valid_nodes = [i for i in range(1, N + 1) if i not in individually_invalid]
    valid_set = set(valid_nodes)

    # Build conflict graph
    conflict = {}
    for i in valid_nodes:
        conflicts = set()
        for s in squares:
            j = s - i
            if 1 <= j <= N and j != i and j in valid_set:
                conflicts.add(j)
        conflict[i] = conflicts

    def greedy_min_degree(nodes):
        remaining = set(nodes)
        selected = set()
        while remaining:
            best_node = None
            best_deg = float('inf')
            for node in remaining:
                deg = len(conflict[node] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_node = node
            selected.add(best_node)
            remaining.discard(best_node)
            remaining -= conflict[best_node]
        return selected

    def local_search(sol, time_limit):
        sol = set(sol)
        unselected = [n for n in valid_nodes if n not in sol]
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding
            for node in unselected:
                if node in sol:
                    continue
                if not (conflict[node] & sol):
                    sol.add(node)
                    unselected.remove(node)
                    improved = True
            # Try 1-swap: remove one selected, add one unselected
            for node in unselected:
                if node in sol:
                    continue
                conflicting = conflict[node] & sol
                if len(conflicting) == 1:
                    rem = next(iter(conflicting))
                    # Check if removing rem and adding node is beneficial
                    # Net gain: +1 (add node) -1 (remove rem) = 0, but maybe node allows more adds
                    sol.discard(rem)
                    sol.add(node)
                    improved = True
            # Try 2-swap: remove two selected, add one unselected
            unselected = [n for n in valid_nodes if n not in sol]
            for node in unselected:
                if node in sol:
                    continue
                conflicting = conflict[node] & sol
                if len(conflicting) == 2:
                    rem1, rem2 = list(conflicting)
                    sol.discard(rem1)
                    sol.discard(rem2)
                    sol.add(node)
                    improved = True
        return sol

    best_sol = set()
    best_size = 0

    # Multiple restarts with different orderings
    rng = random.Random(42)
    num_restarts = 50

    for r in range(num_restarts):
        if time.time() >= deadline - 5.0:
            break

        # Shuffle valid nodes for different greedy orderings
        shuffled = valid_nodes[:]
        rng.shuffle(shuffled)

        # Greedy
        sol = greedy_min_degree(shuffled)
        # Local search
        sol = local_search(sol, deadline - 3.0)

        if len(sol) > best_size:
            best_size = len(sol)
            best_sol = set(sol)

    # Final local search on best solution
    best_sol = local_search(best_sol, deadline - 2.0)

    # Verify
    result = sorted(best_sol)
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            r2 = math.isqrt(s)
            if r2 * r2 == s:
                # Fallback to seed
                return list(range(1, 257, 3))

    if not result:
        return [1]

    return result
# EVOLVE-BLOCK-END
