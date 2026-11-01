import time
import random
import sys

def construct(n_rows, k):
    """
    Constructs n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct over all rows.
    Minimizes the largest entry (the scope).
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit

    def now():
        return time.time()

    # ---------- Baseline greedy (gives an upper bound for pruning) ----------
    def greedy_baseline(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new = {x - a for a in row}
                    if not (new & used):
                        break
                    x += 1
                row.append(x)
                used |= new
            rows.append(row)
        return rows, used

    baseline_rows, baseline_used = greedy_baseline(n_rows, k)
    baseline_scope = max(max(r) for r in baseline_rows)

    # ---------- Beam search over partial rows ----------
    # A state: (row_tuple, used_set, scope)
    # We build rows one at a time; within a row we build elements one at a time.

    def beam_search(n_rows, k, beam_width, prune_scope, time_budget):
        # State: (row, used, total_scope)
        # We'll do it row by row. For each row, beam search the elements.
        best_rows = None
        best_scope = prune_scope  # we want to beat this

        # Start with empty list of completed rows
        # state = (completed_rows_tuple, current_partial_row_tuple, used_set, scope)
        # To keep memory manageable, we only keep top beam_width states per "layer"

        # Layer: (row_idx, elem_idx)
        # We build row by row. For each row, we beam-search its elements.

        # Actually, let's do a full beam search where each state is a partial
        # assignment of all rows. That's too complex. Instead:
        # Build rows sequentially. For each row, do a local beam search
        # over its elements, given the used set from previous rows.

        # Simpler: build all rows sequentially with beam search within each row.
        # But that's not truly global. Let's do a proper beam search.

        # State: (rows_so_far_tuple, partial_row_tuple, used_set)
        # We expand by adding one element to the partial row.
        # When partial row is complete (len = k+1), we start a new row.

        # To keep it tractable, beam_width states total.

        # Initial state
        initial_state = ((), (0,), set())
        beam = [initial_state]

        # We'll track the best complete solution
        best_complete = None
        best_complete_scope = prune_scope

        # Max scope we allow for pruning: prune_scope
        # We expand states, and prune any state whose current scope already
        # exceeds prune_scope, or which cannot possibly beat best_complete_scope.

        # Time check
        t0 = now()
        time_limit = t0 + time_budget

        # We'll do iterative deepening on the scope bound.
        # Actually, let's just do beam search with a fixed beam width and
        # prune by the current best complete scope.

        # For efficiency, we'll limit the number of states and use a
        # priority queue (min-heap) ordered by current scope.

        import heapq

        # Heap: (scope, counter, rows_so_far, partial_row, used_set)
        counter = 0
        heap = []
        heapq.heappush(heap, (0, counter, (), (0,), set()))
        counter += 1

        # We'll expand states level by level. Each level = one element added
        # to the current partial row, or completion of a row.
        # Total levels = n_rows * (k+1) - 1 (since each row starts with 0)
        # Actually, each row has k+1 elements, first is 0. So we need k elements
        # per row. Total expansions = n_rows * k.

        total_expansions = n_rows * k
        for level in range(total_expansions):
            if now() > time_limit:
                break

            new_beam = []
            # Expand all states in the heap
            # But heap can grow large. Let's limit it.
            # Actually, we should process all states at this level.
            # Let's use a list instead of heap for level-by-level.

            # Hmm, this is getting complex. Let me restructure.

            break  # This approach is too complex. Let me try a different one.

        return best_complete

    # ---------- Better approach: sequential row construction with beam search ----------
    # Build rows one at a time. For each row, do a beam search over its elements.
    # The beam width controls how many partial rows we keep.

    def build_row_beam(used, k, beam_width, max_scope, time_budget):
        """
        Build a single row of k+1 elements starting at 0, given the used set.
        Returns a list of (row, new_used) pairs, beam_width best ones.
        """
        t0 = now()
        time_limit = t0 + time_budget

        # State: (partial_row_tuple, used_set, scope)
        # Start with (0,)
        beam = [((0,), set(), 0)]

        for elem_idx in range(1, k + 1):
            if now() > time_limit:
                break

            new_beam = []
            for (partial_row, partial_used, scope) in beam:
                last = partial_row[-1]
                # Try next values
                # We want to keep scope small, so try small values first
                # But we also want diversity, so try a range
                start_val = last + 1
                # Limit the range to avoid infinite loop
                # max_scope is the bound on the largest element
                max_val = min(max_scope, last + k + 50)  # heuristic
                # Actually, we need to be more careful. Let's just try a window.
                # For beam search, we want to explore a range.
                # Let's try from start_val to start_val + window, where window grows.
                # For simplicity, try up to max_val but limit the number of candidates.

                # Collect valid candidates
                valid_candidates = []
                for x in range(start_val, max_val + 1):
                    new_diffs = {x - a for a in partial_row}
                    if not (new_diffs & used) and not (new_diffs & partial_used):
                        valid_candidates.append(x)
                    if len(valid_candidates) >= beam_width * 2:
                        break

                if not valid_candidates:
                    continue

                # For each valid candidate, create a new state
                for x in valid_candidates[:beam_width]:
                    new_row = partial_row + (x,)
                    new_diffs = {x - a for a in partial_row}
                    new_partial_used = partial_used | new_diffs
                    new_scope = max(scope, x)
                    new_beam.append((new_row, new_partial_used, new_scope))

            # Keep only top beam_width states
            if not new_beam:
                break

            # Sort by scope, keep top beam_width
            new_beam.sort(key=lambda s: s[2])
            beam = new_beam[:beam_width]

            if not beam:
                break

        # Convert beam to list of (row, used)
        results = []
        for (row, partial_used, scope) in beam:
            if len(row) == k + 1:
                results.append((list(row), partial_used, scope))
        return results

    # ---------- Full construction with sequential beam search ----------
    def construct_beam(n_rows, k, beam_width, time_budget):
        t0 = now()
        time_limit = t0 + time_budget

        # We'll build rows sequentially. For each row, we do a beam search.
        # But we want to optimize the global scope, so we need to consider
        # the interaction between rows.
        #
        # Approach: build rows one at a time. For each row, generate beam_width
        # candidates. Then, for each combination of candidates (which is too many),
        # we'd need to check. Instead, we'll do a greedy selection: for each row,
        # pick the candidate that minimizes the global scope, given the used set
        # from previous rows.
        #
        # This is a heuristic, but it should work well.

        rows = []
        used = set()
        current_scope = 0

        for row_idx in range(n_rows):
            if now() > time_limit:
                break

            # Determine the max scope we can allow for this row
            # We want the global scope to be as small as possible.
            # Let's try to keep each row's scope <= current_scope + some margin.
            # Actually, we want to minimize the max over all rows.
            # Let's just do a beam search with a reasonable max_scope.

            # Try increasing max_scope until we find a valid row
            # Start with a small value and increase
            max_scope = current_scope + 10
            found = False

            while max_scope <= baseline_scope + 50:
                if now() > time_limit:
                    break

                candidates = build_row_beam(used, k, beam_width, max_scope, 5.0)
                if candidates:
                    # Pick the best candidate (smallest scope)
                    best_candidate = min(candidates, key=lambda c: c[2])
                    row, new_used, scope = best_candidate
                    rows.append(row)
                    used |= new_used
                    current_scope = max(current_scope, scope)
                    found = True
                    break
                max_scope += 5

            if not found:
                # Fallback: use greedy for this row
                row = [0]
                while len(row) < k + 1:
                    x = row[-1] + 1
                    while True:
                        new = {x - a for a in row}
                        if not (new & used):
                            break
                        x += 1
                    row.append(x)
                    used |= new
                rows.append(row)
                current_scope = max(current_scope, max(row))

        return rows, current_scope

    # ---------- Polish: exhaustive single-element optimization ----------
    def polish(rows, used, time_budget):
        """
        Try to reduce the scope by moving individual elements to smaller values.
        """
        t0 = now()
        time_limit = t0 + time_budget
        current_scope = max(max(r) for r in rows)

        # Pre-compute diffs for each row
        row_diffs = []
        for r in rows:
            diffs = set()
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.add(r[j] - r[i])
            row_diffs.append(diffs)

        # Try to reduce each element
        for _ in range(100):  # Multiple passes
            if now() > time_limit:
                break

            improved = False
            for r_idx in range(len(rows)):
                row = rows[r_idx]
                for pos in range(1, len(row)):
                    if now() > time_limit:
                        break

                    current_val = row[pos]
                    lower_bound = row[pos - 1] + 1
                    upper_bound = current_val - 1
                    if pos < len(row) - 1:
                        upper_bound = min(upper_bound, row[pos + 1] - 1)

                    if lower_bound > upper_bound:
                        continue

                    # Collect old diffs involving position pos
                    old_diffs = set()
                    for i in range(len(row)):
                        if i != pos:
                            if row[pos] > row[i]:
                                old_diffs.add(row[pos] - row[i])
                            else:
                                old_diffs.add(row[i] - row[pos])

                    available_diffs = used - old_diffs

                    # Try to find a smaller valid value
                    for v in range(lower_bound, upper_bound + 1):
                        new_diffs = set()
                        for i in range(len(row)):
                            if i != pos:
                                if v > row[i]:
                                    new_diffs.add(v - row[i])
                                else:
                                    new_diffs.add(row[i] - v)

                        if not (new_diffs & available_diffs):
                            # Valid move
                            row[pos] = v
                            used -= old_diffs
                            used |= new_diffs
                            row_diffs[r_idx] = set()
                            for i in range(len(row)):
                                for j in range(i + 1, len(row)):
                                    row_diffs[r_idx].add(row[j] - row[i])

                            new_scope = max(max(r) for r in rows)
                            if new_scope < current_scope:
                                current_scope = new_scope
                                improved = True
                            break

            if not improved:
                break

        return rows, current_scope

    # ---------- Main algorithm ----------
    best_rows = baseline_rows
    best_scope = baseline_scope

    # Try beam search with different parameters
    beam_configs = [
        (5, 30),   # beam_width=5, time_budget=30s
        (10, 30),  # beam_width=10, time_budget=30s
        (20, 30),  # beam_width=20, time_budget=30s
    ]

    for beam_width, time_budget in beam_configs:
        if now() > deadline - 15:
            break

        try:
            rows, scope = construct_beam(n_rows, k, beam_width, time_budget)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
        except Exception:
            pass

    # Polish the best solution
    if now() < deadline - 10:
        try:
            opt_rows = [r[:] for r in best_rows]
            # Recompute used set
            opt_used = set()
            for r in opt_rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        opt_used.add(r[j] - r[i])

            opt_rows, final_scope = polish(opt_rows, opt_used, deadline - 5)
            if final_scope < best_scope:
                best_scope = final_scope
                best_rows = opt_rows
        except Exception:
            pass

    return best_rows
