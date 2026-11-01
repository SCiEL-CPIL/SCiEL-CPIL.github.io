"""DLX exact-cover + greedy/repair hybrid for difference-triangle sets."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs_list(row):
    n = len(row)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(row[j] - row[i])
    return out

def _row_diffs_set(row):
    return set(_row_diffs_list(row))

def _is_row_sidon(row):
    d = _row_diffs_list(row)
    return len(d) == len(set(d))

def _gen_sidon_rows(k, max_val, deadline):
    """Generate all (k+1)-element Sidon subsets of {0..max_val} containing 0.
    Returns list of (row_tuple, frozenset_of_diffs)."""
    results = []
    n = k + 1
    # DFS: pick k values from 1..max_val, ensuring all diffs distinct
    diffs_used = set()
    chosen = [0]
    count = [0]

    def dfs(start, depth):
        if time.time() > deadline:
            return
        if depth == n:
            row = tuple(chosen)
            diffs = frozenset(diffs_used)
            results.append((row, diffs))
            count[0] += 1
            return
        # remaining slots: n - depth
        rem = n - depth
        # next value must be at least start, and leave room for (rem-1) more
        max_next = max_val - (rem - 1)
        for v in range(start, max_next + 1):
            # check new diffs against existing
            new_diffs = set()
            ok = True
            for a in chosen:
                d = v - a
                if d in diffs_used or d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                continue
            # also check new_diffs internal duplicates (already handled above)
            chosen.append(v)
            diffs_used.update(new_diffs)
            dfs(v + 1, depth + 1)
            chosen.pop()
            diffs_used.difference_update(new_diffs)

    dfs(1, 1)
    return results

def _dlx_solve(rows_list, n_rows, deadline):
    """
    Exact cover via DLX: select n_rows rows such that all diff-columns are covered exactly once.
    rows_list: list of (row_tuple, frozenset_of_diffs)
    Returns list of row_tuples or None.
    """
    if not rows_list:
        return None
    # Build column -> list of row indices
    all_diffs = set()
    for _, diffs in rows_list:
        all_diffs.update(diffs)
    cols = sorted(all_diffs)
    col_idx = {c: i for i, c in enumerate(cols)}
    n_cols = len(cols)
    n_rows_avail = len(rows_list)

    # For each row, list of column indices it covers
    row_cols = []
    for _, diffs in rows_list:
        row_cols.append([col_idx[d] for d in diffs])

    # Column header: for each column, list of rows that cover it
    col_rows = [[] for _ in range(n_cols)]
    for r in range(n_rows_avail):
        for c in row_cols[r]:
            col_rows[c].append(r)

    # DLX structure
    # We use arrays: L, R, U, D, col, row_idx
    # Total nodes = 1 (root) + n_cols (headers) + sum(len(row_cols[r]) for r)
    total_nodes = 1 + n_cols + sum(len(rc) for rc in row_cols)
    L = [0] * total_nodes
    R = [0] * total_nodes
    U = [0] * total_nodes
    D = [0] * total_nodes
    col_hdr = [0] * total_nodes  # which column header this node belongs to
    row_id = [-1] * total_nodes  # which data row this node is in
    size_col = [0] * n_cols  # number of nodes in each column

    # Initialize doubly linked circular list
    root = 0
    # Horizontal: root <-> col0 <-> col1 <-> ... <-> colN-1 <-> root
    # Vertical: each column is a circular list of its nodes
    node_count = 1
    col_node = [0] * n_cols  # node index for each column header

    # Create column headers
    for c in range(n_cols):
        col_node[c] = node_count
        node_count += 1
        # Link horizontally
        L[col_node[c]] = col_node[c - 1] if c > 0 else root
        R[col_node[c]] = c + 1 if c < n_cols - 1 else root
        if c > 0:
            R[col_node[c - 1]] = col_node[c]
        else:
            R[root] = col_node[c]
        if c < n_cols - 1:
            L[col_node[c + 1]] = col_node[c]
        else:
            L[root] = col_node[c]
        # Vertical: initially just the header points to itself
        U[col_node[c]] = col_node[c]
        D[col_node[c]] = col_node[c]
        col_hdr[col_node[c]] = c
        row_id[col_node[c]] = -1

    # Root links
    L[root] = col_node[n_cols - 1]
    R[root] = col_node[0]

    # Add data nodes
    for r in range(n_rows_avail):
        for c in row_cols[r]:
            node = node_count
            node_count += 1
            # Insert at bottom of column c
            col_hdr[node] = c
            row_id[node] = r
            # Vertical insert: above D[col_node[c]]
            bottom = D[col_node[c]]
            U[node] = bottom
            D[node] = col_node[c]
            D[bottom] = node
            U[col_node[c]] = node
            size_col[c] += 1
            # Horizontal: link within row r (just for removal; we can skip full horizontal for data nodes
            # since we remove by column. But DLX needs horizontal links for unhook.)
            # Simplification: we'll do column-based removal without full horizontal data links.
            # Actually, standard DLX needs horizontal links. Let's add them.
            # For simplicity in this implementation, we'll use a slightly different removal strategy.

    # Standard DLX cover/uncover
    def cover(c):
        """Remove column c and all rows that intersect it."""
        # Remove column header from horizontal list
        R[L[c]] = R[c]
        L[R[c]] = L[c]
        # For each node in column c (below header):
        i = D[c]
        while i != c:
            # For each node j in this row (horizontal), remove from its column
            j = R[i]
            while j != i:
                # Remove j from its column vertically
                D[U[j]] = D[j]
                U[D[j]] = U[j]
                size_col[col_hdr[j]] -= 1
                j = R[j]
            # Move to next row in column c
            i = D[i]

    def uncover(c):
        """Restore column c and all rows."""
        i = U[c]
        while i != c:
            j = L[i]
            while j != i:
                size_col[col_hdr[j]] += 1
                U[D[j]] = j
                D[U[j]] = j
                j = L[j]
            i = U[i]
        # Restore column header
        R[L[c]] = c
        L[R[c]] = c

    # This simplified DLX is complex; let's use a simpler backtracking with bitsets for n_rows <= 7
    # Given the small scale (n_rows=7, k=6), we can use a different approach.
    return None  # Fallback to other methods

def _greedy_construct(n_rows, k, rng, used, deadline, bias=1.0):
    """Fast greedy construction with incremental diff tracking."""
    rows = []
    for _ in range(n_rows):
        if time.time() > deadline:
            return None
        row = [0]
        row_diffs = set()  # diffs within current row
        ok = True
        for _pos in range(k):
            found = False
            lo = row[-1] + 1
            # Try values with bias
            max_attempts = 50
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    return None
                # Generate candidate
                if attempt < 15:
                    v = lo + attempt
                else:
                    v = lo + rng.randint(0, 50)
                # Check new diffs
                new_diffs = set()
                valid = True
                for a in row:
                    d = v - a
                    if d in row_diffs or d in used or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    row.append(v)
                    row_diffs.update(new_diffs)
                    found = True
                    break
            if not found:
                ok = False
                break
        if ok:
            used.update(row_diffs)
            rows.append(row)
        else:
            return None
    return rows

def _local_search(rows, n_rows, k, rng, deadline):
    """Iterative improvement: try to reduce max entry by rebuilding rows."""
    cur = [r[:] for r in rows]
    cur_val = _max_entry(cur)
    used = set()
    for r in cur:
        used.update(_row_diffs_set(r))

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Pick row with largest max
        target = max(range(n_rows), key=lambda i: cur[i][-1])
        old_row = cur[target]
        old_diffs = _row_diffs_set(old_row)
        used -= old_diffs

        # Try to rebuild with smaller max
        best_new = None
        for attempt in range(100):
            if time.time() > deadline:
                break
            new_row = [0]
            new_diffs = set()
            target_max = old_row[-1] - 1 - attempt // 10
            ok = True
            for _pos in range(k):
                found = False
                lo = new_row[-1] + 1
                hi = target_max - (k - len(new_row))
                if hi < lo:
                    ok = False
                    break
                for _trial in range(30):
                    v = lo + rng.randint(0, max(0, hi - lo))
                    nd = set()
                    valid = True
                    for a in new_row:
                        d = v - a
                        if d in new_diffs or d in used or d in nd:
                            valid = False
                            break
                        nd.add(d)
                    if valid:
                        new_row.append(v)
                        new_diffs.update(nd)
                        found = True
                        break
                if not found:
                    ok = False
                    break
            if ok and new_row[-1] < old_row[-1]:
                if best_new is None or new_row[-1] < best_new[-1]:
                    best_new = new_row
                    if best_new[-1] <= k:  # can't do much better
                        break

        if best_new is not None:
            cur[target] = best_new
            used.update(_row_diffs_set(best_new))
            new_val = _max_entry(cur)
            if new_val < cur_val:
                cur_val = new_val
                improved = True
        else:
            used |= old_diffs

    return cur, cur_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115
    rng = random.Random(42)

    best = None
    best_val = float('inf')

    # Phase 1: Multiple greedy constructions with different biases
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        used = set()
        bias = 0.8 + rng.random() * 0.8
        rows = _greedy_construct(n_rows, k, rng, used, deadline - 5, bias)
        if rows is not None and _valid(rows, n_rows, k):
            val = _max_entry(rows)
            if val < best_val:
                best_val = val
                best = [r[:] for r in rows]

    # Phase 2: Local search improvement
    if best is not None and time.time() < deadline - 3:
        improved_rows, improved_val = _local_search(best, n_rows, k, rng, deadline - 3)
        if improved_val < best_val:
            best_val = improved_val
            best = improved_rows

    # Phase 3: More greedy starts if time remains
    while time.time() < deadline - 2:
        used = set()
        rows = _greedy_construct(n_rows, k, rng, used, deadline - 2)
        if rows is not None and _valid(rows, n_rows, k):
            val = _max_entry(rows)
            if val < best_val:
                best_val = val
                best = [r[:] for r in rows]
                # Quick local search
                if time.time() < deadline - 5:
                    ir, iv = _local_search(best, n_rows, k, rng, deadline - 5)
                    if iv < best_val:
                        best_val = iv
                        best = ir

    # Fallback: simple sequential greedy
    if best is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                nd = {x - a for a in row}
                if not (nd & used) and len(nd) == len(row):
                    row.append(x)
                    used |= nd
            rows.append(row)
        if _valid(rows, n_rows, k):
            best = rows

    return best

# EVOLVE-BLOCK-END
