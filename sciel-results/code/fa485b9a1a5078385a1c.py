"""Combine max-clique branch-and-bound (structural) with randomized greedy restarts (refinement)."""
# EVOLVE-BLOCK-START
import time
import random
import math

_deadline_global = 0.0

def construct(n_rows, k):
    global _deadline_global
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin under 150s limit
    _deadline_global = deadline

    # Fallback deterministic greedy (guaranteed valid, small scope)
    best_rows = _greedy_construct(n_rows, k)
    best_scope = max(max(r) for r in best_rows) if best_rows else float('inf')

    # Phase 1: Randomized greedy with random restarts (fast, many trials)
    while time.time() < deadline - 30.0:
        rows, scope = _randomized_greedy(n_rows, k, best_scope)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
            lower_bound = n_rows * (k + 1) * k // 2
            if best_scope <= lower_bound:
                break

    # Phase 2: Maximum clique branch-and-bound (structural)
    if n_rows <= 7 and k <= 6 and time.time() < deadline - 5.0:
        improved = _max_clique_search(n_rows, k, best_scope, deadline)
        if improved is not None:
            scope = max(max(r) for r in improved)
            if scope < best_scope:
                best_scope = scope
                best_rows = improved

    return best_rows


def _greedy_construct(n_rows, k):
    """Deterministic greedy: pick smallest valid next mark."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _randomized_greedy(n_rows, k, current_best_scope):
    """Randomized greedy construction with random candidate selection."""
    used = set()
    rows = []

    for i in range(n_rows):
        row = [0]
        last = 0

        while len(row) < k + 1:
            candidates = []
            x = last
            search_limit = last + 80

            while x < search_limit and len(candidates) < 15:
                x += 1
                if x >= current_best_scope:
                    return None, float('inf')
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    candidates.append((x, new_diffs))

            if not candidates:
                return None, float('inf')

            if random.random() < 0.25:
                x, new_diffs = candidates[0]
            else:
                pool = candidates[:min(6, len(candidates))]
                x, new_diffs = random.choice(pool)

            row.append(x)
            used.update(new_diffs)
            last = x

        rows.append(row)

    scope = max(max(r) for r in rows)
    return rows, scope


def _max_clique_search(n_rows, k, initial_scope, deadline):
    """Maximum clique search with proper row storage."""
    lower_bound = n_rows * (k + 1) * k // 2

    for scope in range(initial_scope - 1, max(lower_bound - 1, 50), -1):
        if time.time() > deadline - 2.0:
            break

        valid_rows = []

        def backtrack(row, used_diffs, max_scope):
            if time.time() > _deadline_global:
                return
            if len(row) == k + 1:
                valid_rows.append((frozenset(used_diffs), list(row)))
                return
            last = row[-1]
            for x in range(last + 1, max_scope + 1):
                new_diffs = frozenset(x - a for a in row)
                if not (new_diffs & used_diffs):
                    row.append(x)
                    backtrack(row, used_diffs | new_diffs, max_scope)
                    row.pop()

        backtrack([0], frozenset(), scope)
        if len(valid_rows) < n_rows:
            continue

        n_nodes = len(valid_rows)
        diffs_list = [r[0] for r in valid_rows]

        adj = [set() for _ in range(n_nodes)]
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if not (diffs_list[i] & diffs_list[j]):
                    adj[i].add(j)
                    adj[j].add(i)

        found = _find_clique(adj, n_nodes, n_rows, deadline)
        if found:
            result_rows = [valid_rows[idx][1] for idx in found]
            return result_rows

    return None


def _find_clique(adj, n_nodes, target_size, deadline):
    """Find a clique of target_size using backtracking with degree ordering."""
    degrees = [len(adj[i]) for i in range(n_nodes)]
    order = sorted(range(n_nodes), key=lambda i: -degrees[i])
    pos_to_node = order
    node_to_pos = {node: pos for pos, node in enumerate(order)}

    adj_ordered = [set() for _ in range(n_nodes)]
    for i in range(n_nodes):
        for j in adj[pos_to_node[i]]:
            if node_to_pos[j] > i:
                adj_ordered[i].add(node_to_pos[j])

    best_clique = []

    def backtrack(candidates, current_clique):
        if time.time() > deadline - 1.0:
            return False
        if len(current_clique) == target_size:
            best_clique[:] = current_clique
            return True
        if len(candidates) < target_size - len(current_clique):
            return False

        for i in range(len(candidates)):
            node = candidates[i]
            new_clique = current_clique + [node]
            new_candidates = [c for c in candidates[i+1:] if c in adj_ordered[node]]
            if backtrack(new_candidates, new_clique):
                return True

        return False

    if backtrack(list(range(n_nodes)), []):
        return [pos_to_node[i] for i in best_clique]
    return None
# EVOLVE-BLOCK-END
