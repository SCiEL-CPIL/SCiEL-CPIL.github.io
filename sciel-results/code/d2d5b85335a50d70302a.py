# EVOLVE-BLOCK-START
import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # safety margin before 60s kill

def _build_sum_diff(elems):
    """Compute sum_bitmask and diff_bitmask for a list of elements."""
    sum_bm = 0
    diff_bm = 0
    m = len(elems)
    for i in range(m):
        a = elems[i]
        # a + a
        sum_bm |= (1 << (a + a))
        for j in range(i + 1, m):
            b = elems[j]
            sum_bm |= (1 << (a + b))
            d = abs(a - b)
            diff_bm |= (1 << d)
    return sum_bm, diff_bm

def _can_add(elem, elems, sum_bm, diff_bm, other_diff_bm):
    """Check if elem can be added. Returns True if valid."""
    # Check sum collisions: elem+elem and elem+a for all a in elems
    if (sum_bm >> (elem + elem)) & 1:
        return False
    for a in elems:
        s = elem + a
        if (sum_bm >> s) & 1:
            return False
    # Check difference collisions
    for a in elems:
        d = abs(elem - a)
        if (diff_bm >> d) & 1:
            return False
    # Check cross difference with other set
    # d must not be in other_diff_bm
    if (other_diff_bm >> d) & 1:
        return False
    return True

def _greedy_build(order_a, order_b, max_each=MAX_EACH):
    """Greedily build two sets following given element orders."""
    A = []
    B = []
    sum_bm_a = 0
    diff_bm_a = 0
    sum_bm_b = 0
    diff_bm_b = 0

    for x in order_a:
        if len(A) >= max_each:
            break
        # Check if x can be added to A
        if (sum_bm_a >> (x + x)) & 1:
            continue
        ok = True
        for a in A:
            s = x + a
            if (sum_bm_a >> s) & 1:
                ok = False
                break
            d = abs(x - a)
            if (diff_bm_a >> d) & 1:
                ok = False
                break
            if (diff_bm_b >> d) & 1:
                ok = False
                break
        if ok:
            A.append(x)
            sum_bm_a |= (1 << (x + x))
            for a in A[:-1]:
                sum_bm_a |= (1 << (x + a))
                diff_bm_a |= (1 << abs(x - a))

    for x in order_b:
        if len(B) >= max_each:
            break
        if (sum_bm_b >> (x + x)) & 1:
            continue
        ok = True
        for b in B:
            s = x + b
            if (sum_bm_b >> s) & 1:
                ok = False
                break
            d = abs(x - b)
            if (diff_bm_b >> d) & 1:
                ok = False
                break
            if (diff_bm_a >> d) & 1:
                ok = False
                break
        if ok:
            B.append(x)
            sum_bm_b |= (1 << (x + x))
            for b in B[:-1]:
                sum_bm_b |= (1 << (x + b))
                diff_bm_b |= (1 << abs(x - b))

    return A, B

def _greedy_interleaved(order, max_each=MAX_EACH):
    """Build A and B from a single interleaved order."""
    A = []
    B = []
    sum_bm_a = 0
    diff_bm_a = 0
    sum_bm_b = 0
    diff_bm_b = 0

    for x in order:
        # Try adding to A first, then B
        added = False
        if len(A) < max_each:
            if not ((sum_bm_a >> (x + x)) & 1):
                ok = True
                for a in A:
                    s = x + a
                    if (sum_bm_a >> s) & 1:
                        ok = False
                        break
                    d = abs(x - a)
                    if (diff_bm_a >> d) & 1 or (diff_bm_b >> d) & 1:
                        ok = False
                        break
                if ok:
                    A.append(x)
                    sum_bm_a |= (1 << (x + x))
                    for a in A[:-1]:
                        sum_bm_a |= (1 << (x + a))
                        diff_bm_a |= (1 << abs(x - a))
                    added = True

        if not added and len(B) < max_each:
            if not ((sum_bm_b >> (x + x)) & 1):
                ok = True
                for b in B:
                    s = x + b
                    if (sum_bm_b >> s) & 1:
                        ok = False
                        break
                    d = abs(x - b)
                    if (diff_bm_b >> d) & 1 or (diff_bm_a >> d) & 1:
                        ok = False
                        break
                if ok:
                    B.append(x)
                    sum_bm_b |= (1 << (x + x))
                    for b in B[:-1]:
                        sum_bm_b |= (1 << (x + b))
                        diff_bm_b |= (1 << abs(x - b))

    return A, B

def _backtrack_search(order_a, order_b, max_each, time_limit):
    """Backtracking search to maximize |A|+|B|."""
    best_total = 0
    best_A = []
    best_B = []

    A = []
    B = []
    sum_bm_a = 0
    diff_bm_a = 0
    sum_bm_b = 0
    diff_bm_b = 0

    def _update_best():
        nonlocal best_total, best_A, best_B
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = list(A)
            best_B = list(B)

    def _search(idx_a, idx_b):
        nonlocal sum_bm_a, diff_bm_a, sum_bm_b, diff_bm_b
        if time.time() > time_limit:
            return

        _update_best()

        # Try to add next element to A
        if idx_a < len(order_a) and len(A) < max_each:
            x = order_a[idx_a]
            if not ((sum_bm_a >> (x + x)) & 1):
                ok = True
                for a in A:
                    s = x + a
                    if (sum_bm_a >> s) & 1:
                        ok = False
                        break
                    d = abs(x - a)
                    if (diff_bm_a >> d) & 1 or (diff_bm_b >> d) & 1:
                        ok = False
                        break
                if ok:
                    A.append(x)
                    old_sum_a = sum_bm_a
                    old_diff_a = diff_bm_a
                    sum_bm_a |= (1 << (x + x))
                    for a in A[:-1]:
                        sum_bm_a |= (1 << (x + a))
                        diff_bm_a |= (1 << abs(x - a))
                    _search(idx_a + 1, idx_b)
                    A.pop()
                    sum_bm_a = old_sum_a
                    diff_bm_a = old_diff_a

        # Try to add next element to B
        if idx_b < len(order_b) and len(B) < max_each:
            x = order_b[idx_b]
            if not ((sum_bm_b >> (x + x)) & 1):
                ok = True
                for b in B:
                    s = x + b
                    if (sum_bm_b >> s) & 1:
                        ok = False
                        break
                    d = abs(x - b)
                    if (diff_bm_b >> d) & 1 or (diff_bm_a >> d) & 1:
                        ok = False
                        break
                if ok:
                    B.append(x)
                    old_sum_b = sum_bm_b
                    old_diff_b = diff_bm_b
                    sum_bm_b |= (1 << (x + x))
                    for b in B[:-1]:
                        sum_bm_b |= (1 << (x + b))
                        diff_bm_b |= (1 << abs(x - b))
                    _search(idx_a, idx_b + 1)
                    B.pop()
                    sum_bm_b = old_sum_b
                    diff_bm_b = old_diff_b

        # Skip: don't add this element to either set (move both pointers)
        # This is a simplification - in practice we'd branch more carefully
        if idx_a < len(order_a) and idx_b < len(order_b):
            _search(idx_a + 1, idx_b + 1)
        elif idx_a < len(order_a):
            _search(idx_a + 1, idx_b)
        elif idx_b < len(order_b):
            _search(idx_a, idx_b + 1)

    _search(0, 0)
    return best_A, best_B

def _validate(A, B):
    """Validate that A and B are valid Sidon sets with disjoint nonzero differences."""
    if not A or not B:
        return False
    if len(A) > MAX_EACH or len(B) > MAX_EACH:
        return False
    for x in A + B:
        if x < 1 or x > N:
            return False
    # Check A is Sidon
    sums_a = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sums_a:
                return False
            sums_a.add(s)
    # Check B is Sidon
    sums_b = set()
    for i in range(len(B)):
        for j in range(i, len(B)):
            s = B[i] + B[j]
            if s in sums_b:
                return False
            sums_b.add(s)
    # Check differences are disjoint
    diffs_a = set()
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                d = abs(A[i] - A[j])
                if d > 0:
                    diffs_a.add(d)
    diffs_b = set()
    for i in range(len(B)):
        for j in range(len(B)):
            if i != j:
                d = abs(B[i] - B[j])
                if d > 0:
                    diffs_b.add(d)
    if diffs_a & diffs_b:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_A = [1]
    best_B = [2]
    best_score = 2

    # Try multiple random strategies
    strategies = 0
    while time.time() < deadline - 2.0:
        strategies += 1
        strategy_type = strategies % 4

        if strategy_type == 0:
            # Random order for A, random order for B
            order_a = list(range(1, N + 1))
            random.shuffle(order_a)
            order_b = list(range(1, N + 1))
            random.shuffle(order_b)
            A, B = _greedy_build(order_a, order_b)
        elif strategy_type == 1:
            # Interleaved random order
            order = list(range(1, N + 1))
            random.shuffle(order)
            A, B = _greedy_interleaved(order)
        elif strategy_type == 2:
            # Greedy with preference for small elements in A, large in B
            order_a = list(range(1, N + 1))
            random.shuffle(order_a)
            order_b = list(range(N, 0, -1))
            random.shuffle(order_b)
            A, B = _greedy_build(order_a, order_b)
        else:
            # Greedy with preference for large elements in A, small in B
            order_a = list(range(N, 0, -1))
            random.shuffle(order_a)
            order_b = list(range(1, N + 1))
            random.shuffle(order_b)
            A, B = _greedy_build(order_a, order_b)

        if A and B:
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)

        # Occasionally try a limited backtracking search
        if strategies % 5 == 0 and time.time() < deadline - 10.0:
            order_a = list(range(1, N + 1))
            random.shuffle(order_a)
            order_b = list(range(1, N + 1))
            random.shuffle(order_b)
            # Limit backtracking depth by truncating orders
            bt_a = order_a[:32]
            bt_b = order_b[:32]
            A_bt, B_bt = _backtrack_search(bt_a, bt_b, MAX_EACH, deadline - 5.0)
            if A_bt and B_bt:
                score_bt = len(A_bt) + len(B_bt)
                if score_bt > best_score:
                    best_score = score_bt
                    best_A = list(A_bt)
                    best_B = list(B_bt)

    # Final validation
    if not _validate(best_A, best_B):
        # Fallback to simple valid sets
        best_A = [1, 2, 4, 8]
        best_B = [16, 32, 64, 128]

    return [best_A, best_B]
# EVOLVE-BLOCK-END
