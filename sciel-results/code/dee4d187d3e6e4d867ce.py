"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 256
    squares = set()
    for k in range(2, 23):
        squares.add(k * k)

    individually_invalid = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            individually_invalid.add(a)

    valid_nodes = [i for i in range(1, N + 1) if i not in individually_invalid]
    valid_set = set(valid_nodes)

    conflict = {}
    for i in valid_nodes:
        conflicts = set()
        for s in squares:
            j = s - i
            if 1 <= j <= N and j != i and j in valid_set:
                conflicts.add(j)
        conflict[i] = conflicts

    remaining = set(valid_nodes)
    selected = []

    while remaining:
        best_node = None
        best_degree = float('inf')
        for node in remaining:
            deg = len(conflict[node] & remaining)
            if deg < best_degree:
                best_degree = deg
                best_node = node
        selected.append(best_node)
        remaining.discard(best_node)
        for nb in conflict[best_node]:
            remaining.discard(nb)

    best_solution = set(selected)
    best_size = len(best_solution)

    # Try adding unselected nodes
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for node in valid_nodes:
            if node in best_solution:
                continue
            can_add = True
            for nb in conflict[node]:
                if nb in best_solution:
                    can_add = False
                    break
            if can_add:
                best_solution.add(node)
                best_size += 1
                improved = True

    # Try swap-based improvement
    unselected = [n for n in valid_nodes if n not in best_solution]

    for _ in range(100):
        if time.time() >= deadline:
            break
        improved = False
        for a in unselected:
            if a in best_solution:
                continue
            conflicting_selected = [nb for nb in conflict[a] if nb in best_solution]
            if len(conflicting_selected) > 1:
                temp_removed = set(conflicting_selected)
                temp_add = {a}
                for b in unselected:
                    if b in best_solution or b == a or b in temp_removed:
                        continue
                    if conflict[b] & temp_add:
                        continue
                    if conflict[b] & (best_solution - temp_removed):
                        continue
                    temp_add.add(b)
                if len(temp_add) > len(temp_removed):
                    best_solution -= temp_removed
                    best_solution |= temp_add
                    best_size = len(best_solution)
                    improved = True
                    unselected = [n for n in valid_nodes if n not in best_solution]
                    break
        if not improved:
            break

    result = sorted(best_solution)
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            r = math.isqrt(s)
            if r * r == s:
                return list(range(1, 257, 3))

    if not result:
        return [1]

    return result
# EVOLVE-BLOCK-END
