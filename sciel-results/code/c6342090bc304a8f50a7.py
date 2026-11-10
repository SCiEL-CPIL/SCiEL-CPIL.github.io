"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

_N = 256
_KILL = 90.0
_MARGIN = 8.0

def _check3(a, b, x):
    return (2 * x == a + b) or (2 * a == x + b) or (2 * b == x + a)

def _conflict(S):
    """Return a set of numbers in [1, _N] that would form a 3-AP with two existing elements."""
    blocked = set()
    S_list = list(S)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i + 1, m):
            b = S_list[j]
            # a, b, x => x = 2b - a
            x = 2 * b - a
            if 1 <= x <= _N and x not in S:
                blocked.add(x)
            # a, x, b => x = (a+b)//2 if even
            s = a + b
            if s % 2 == 0:
                x = s // 2
                if 1 <= x <= _N and x not in S:
                    blocked.add(x)
            # x, a, b => x = 2a - b
            x = 2 * a - b
            if 1 <= x <= _N and x not in S:
                blocked.add(x)
    return blocked

def _backtrack_search(start_idx, current_set, blocked, inv, deadline, best_w, best_set):
    """
    Recursive backtracking: try to add elements from start_idx upward.
    Returns (best_w, best_set) found within time limit.
    """
    if time.time() > deadline:
        return best_w, best_set
    
    w = sum(inv[x] for x in current_set)
    if w > best_w:
        best_w = w
        best_set = set(current_set)
    
    # Try adding each candidate from start_idx to _N
    for x in range(start_idx, _N + 1):
        if time.time() > deadline:
            return best_w, best_set
        if x in blocked:
            continue
        if x in current_set:
            continue
        
        # Add x
        new_blocked = set(blocked)
        new_set = current_set | {x}
        
        # Update blocked: for each existing element a in current_set, x forms 3-APs
        for a in current_set:
            # x, a, ? => ? = 2a - x
            q = 2 * a - x
            if 1 <= q <= _N:
                new_blocked.add(q)
            # ?, x, a => ? = 2x - a
            q = 2 * x - a
            if 1 <= q <= _N:
                new_blocked.add(q)
            # x, ?, a => ? = (x+a)//2
            s = x + a
            if s % 2 == 0:
                q = s // 2
                if 1 <= q <= _N:
                    new_blocked.add(q)
        
        # Remove x from blocked if it was there (it's now in set)
        new_blocked.discard(x)
        
        best_w, best_set = _backtrack_search(x + 1, new_set, new_blocked, inv, deadline, best_w, best_set)
        
        # Prune: if even adding all remaining possible elements can't beat best_w, skip
        # (optimization: estimate max possible gain)
        remaining = _N - x
        max_possible = w + sum(inv[i] for i in range(x + 1, _N + 1) if i not in new_blocked and i not in new_set)
        if max_possible <= best_w:
            continue
    
    return best_w, best_set

def _greedy_desc(rng, deadline):
    """Greedy: add elements in descending order of reciprocal (i.e., ascending numeric), skip if blocked."""
    S = set()
    blocked = set()
    for x in range(1, _N + 1):
        if time.time() > deadline:
            break
        if x in blocked:
            continue
        S.add(x)
        # Update blocked
        for a in list(S):
            if a == x:
                continue
            q = 2 * a - x
            if 1 <= q <= _N:
                blocked.add(q)
            q = 2 * x - a
            if 1 <= q <= _N:
                blocked.add(q)
            s = x + a
            if s % 2 == 0:
                q = s // 2
                if 1 <= q <= _N:
                    blocked.add(q)
        blocked.discard(x)
    return S

def _random_greedy(rng, deadline):
    """Randomized greedy: shuffle order, add if not blocked."""
    order = list(range(1, _N + 1))
    rng.shuffle(order)
    S = set()
    blocked = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in blocked:
            continue
        S.add(x)
        for a in list(S):
            if a == x:
                continue
            q = 2 * a - x
            if 1 <= q <= _N:
                blocked.add(q)
            q = 2 * x - a
            if 1 <= q <= _N:
                blocked.add(q)
            s = x + a
            if s % 2 == 0:
                q = s // 2
                if 1 <= q <= _N:
                    blocked.add(q)
        blocked.discard(x)
    return S

def _hill_climb(S0, inv, deadline):
    """Local search: try swaps and adds to improve weight."""
    S = set(S0)
    weight = sum(inv[x] for x in S)
    best_S = set(S)
    best_w = weight
    eps = 1e-12
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = -1.0
        best_action = None
        best_x = None
        best_y = None
        
        # Try single adds
        blocked = _conflict(S)
        for x in range(1, _N + 1):
            if x not in S and x not in blocked:
                g = inv[x]
                if g > best_gain:
                    best_gain = g
                    best_action = 'add'
                    best_x = x
                    best_y = None
        
        # Try swaps: remove x, then add best y
        for x in list(S):
            if time.time() > deadline:
                break
            S.discard(x)
            blocked = _conflict(S)
            w0 = weight - inv[x]
            for y in range(1, _N + 1):
                if y not in S and y not in blocked:
                    g = inv[y] - inv[x]
                    if g > best_gain:
                        best_gain = g
                        best_action = 'swap'
                        best_x = x
                        best_y = y
                    break
            S.add(x)
            weight = w0 + inv[x]
        
        if best_gain > eps:
            if best_action == 'add':
                S.add(best_x)
                weight += inv[best_x]
            else:
                S.discard(best_x)
                weight -= inv[best_x]
                S.add(best_y)
                weight += inv[best_y]
            if weight > best_w:
                best_w = weight
                best_S = set(S)
            improved = True
    
    return best_S

def construct():
    start = time.time()
    deadline = start + _KILL - _MARGIN
    rng = random.Random(20240517)
    inv = [0.0] + [1.0 / x for x in range(1, _N + 1)]
    
    def w(S):
        return sum(inv[x] for x in S)
    
    best_S = None
    best_w = -1.0
    
    # Phase 1: Backtracking search with time limit
    bt_deadline = min(start + 30.0, deadline - 10.0)
    bt_w, bt_S = _backtrack_search(1, set(), set(), inv, bt_deadline, -1.0, set())
    if bt_S and len(bt_S) > 0:
        cw = w(bt_S)
        if cw > best_w:
            best_w = cw
            best_S = bt_S
    
    # Phase 2: Greedy initializations + hill climbing
    inits = []
    inits.append(_greedy_desc(rng, deadline))
    for _ in range(5):
        inits.append(_random_greedy(rng, deadline))
    
    for S in inits:
        if time.time() > deadline - 5.0:
            break
        opt = _hill_climb(S, inv, deadline)
        cw = w(opt)
        if cw > best_w:
            best_w = cw
            best_S = opt
    
    # Phase 3: Random restarts from best
    if best_S is not None:
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            pert = set(best_S)
            drops = min(3, len(pert))
            for _ in range(drops):
                if pert:
                    pert.discard(random.choice(list(pert)))
            if not pert:
                pert = {1}
            opt = _hill_climb(pert, inv, deadline)
            cw = w(opt)
            if cw > best_w:
                best_w = cw
                best_S = opt
    
    if best_S is None or len(best_S) == 0:
        best_S = {1}
    
    res = sorted(best_S)
    
    # Defensive verification
    Sset = set(res)
    ok = True
    for i, a in enumerate(res):
        for b in res[i+1:]:
            c = 2 * b - a
            if c in Sset and c > b:
                ok = False
            d = 2 * a - b
            if d in Sset and d < a:
                ok = False
            s = a + b
            if s % 2 == 0:
                m = s // 2
                if m in Sset and m != a and m != b:
                    ok = False
    if not ok:
        S = _greedy_desc(rng, deadline)
        res = sorted(S)
    
    return res
# EVOLVE-BLOCK-END
