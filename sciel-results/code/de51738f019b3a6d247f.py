"""Combined strategy: High-value biased initialization + Aggressive local search with Remove-Refill"""
import time
import random
from fractions import Fraction

_N = 256
_KILL = 90.0
_MARGIN = 8.0

def _check3(a, b, x):
    return (2 * x == a + b) or (2 * a == x + b) or (2 * b == x + a)

def _conflict(S):
    """Compute conflict count for each position: how many 3-APs would be formed if x were added"""
    conf = [0] * (_N + 1)
    S_list = list(S)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i + 1, m):
            b = S_list[j]
            x = 2 * b - a
            if 1 <= x <= _N:
                conf[x] += 1
            x = 2 * a - b
            if 1 <= x <= _N:
                conf[x] += 1
            s = a + b
            if s % 2 == 0:
                x = s // 2
                if 1 <= x <= _N:
                    conf[x] += 1
    return conf

def _add_upd(conf, S, s):
    """Increment conflict counts when adding s to S"""
    for a in S:
        if a == s:
            continue
        x = 2 * s - a
        if 1 <= x <= _N:
            conf[x] += 1
        x = 2 * a - s
        if 1 <= x <= _N:
            conf[x] += 1
        t = s + a
        if t % 2 == 0:
            x = t // 2
            if 1 <= x <= _N:
                conf[x] += 1

def _rem_upd(conf, S, s):
    """Decrement conflict counts when removing s from S"""
    for a in S:
        if a == s:
            continue
        x = 2 * s - a
        if 1 <= x <= _N:
            conf[x] -= 1
        x = 2 * a - s
        if 1 <= x <= _N:
            conf[x] -= 1
        t = s + a
        if t % 2 == 0:
            x = t // 2
            if 1 <= x <= _N:
                conf[x] -= 1

def _randomized_greedy(rng, direction='asc', noise=0.0):
    """Randomized greedy: pick elements in a biased order, add if no conflict"""
    S = []
    S_set = set()
    conf = [0] * (_N + 1)
    
    if direction == 'asc':
        order = list(range(1, _N + 1))
    elif direction == 'desc':
        order = list(range(_N, 0, -1))
    else:
        order = list(range(1, _N + 1))
        rng.shuffle(order)
    
    if noise > 0:
        for i in range(len(order) - 1):
            if rng.random() < noise:
                j = rng.randint(i, len(order) - 1)
                order[i], order[j] = order[j], order[i]
    
    for x in order:
        if conf[x] == 0:
            S.append(x)
            S_set.add(x)
            _add_upd(conf, S_set, x)
    
    return S

def _hill_climb_aggressive(S0, inv, deadline, rng):
    """Aggressive hill climbing with add, swap, and remove-refill moves"""
    S = set(S0)
    conf = _conflict(S)
    weight = sum(inv[x] for x in S)
    best_S = set(S)
    best_w = weight
    eps = 1e-12
    
    improved = True
    no_improve_count = 0
    while improved and time.time() < deadline:
        improved = False
        best_gain = -1.0
        best_typ = None
        best_x = None
        best_y = None
        best_remove_set = None
        best_add_set = None
        
        # 1. Try all single adds
        for x in range(1, _N + 1):
            if x not in S and conf[x] == 0:
                g = inv[x]
                if g > best_gain:
                    best_gain = g
                    best_typ = 'add'
                    best_x = x
                    best_y = None
        
        # 2. Try all swaps: remove x, add best y
        for x in list(S):
            if time.time() > deadline:
                break
            S.discard(x)
            _rem_upd(conf, S, x)
            w0 = weight - inv[x]
            
            best_y_for_swap = None
            best_gain_for_swap = -1.0
            for y in range(1, _N + 1):
                if y not in S and conf[y] == 0:
                    g = inv[y] - inv[x]
                    if g > best_gain_for_swap:
                        best_gain_for_swap = g
                        best_y_for_swap = y
            
            if best_y_for_swap is not None and best_gain_for_swap > best_gain:
                best_gain = best_gain_for_swap
                best_typ = 'swap'
                best_x = x
                best_y = best_y_for_swap
            
            _add_upd(conf, S, x)
            S.add(x)
            weight = w0 + inv[x]
        
        # 3. Try Remove-Refill: Remove k random elements, greedily refill with best available
        # We try this occasionally to escape local minima
        if no_improve_count > 2 and rng.random() < 0.3:
            k = rng.choice([2, 3, 4])
            if len(S) > k:
                remove_candidates = list(S)
                rng.shuffle(remove_candidates)
                remove_set = set(remove_candidates[:k])
                
                # Remove them
                S_copy = set(S)
                conf_copy = list(conf)
                w_copy = weight
                for x in remove_set:
                    S_copy.discard(x)
                    _rem_upd(conf_copy, S_copy, x)
                    w_copy -= inv[x]
                
                # Candidates for refill: removed elements + all currently blocked (conf > 0) + all not in S
                # Actually, we can just consider all x not in S_copy.
                # To be efficient, we can sort candidates by inv[x] descending.
                # But we need to check conf_copy.
                
                # Let's just iterate through all x not in S_copy in descending order of inv[x]
                # Since inv[x] = 1/x, this is ascending order of x.
                added = set()
                new_w = w_copy
                for y in range(1, _N + 1):
                    if y in S_copy:
                        continue
                    if conf_copy[y] == 0:
                        S_copy.add(y)
                        _add_upd(conf_copy, S_copy, y)
                        new_w += inv[y]
                        added.add(y)
                
                gain = new_w - weight
                if gain > best_gain:
                    best_gain = gain
                    best_typ = 'refill'
                    best_remove_set = remove_set
                    best_add_set = added
        
        if best_gain > eps:
            if best_typ == 'add':
                _add_upd(conf, S, best_x)
                S.add(best_x)
                weight += inv[best_x]
            elif best_typ == 'swap':
                x = best_x
                y = best_y
                S.discard(x)
                _rem_upd(conf, S, x)
                weight -= inv[x]
                _add_upd(conf, S, y)
                S.add(y)
                weight += inv[y]
            elif best_typ == 'refill':
                for x in best_remove_set:
                    S.discard(x)
                    _rem_upd(conf, S, x)
                    weight -= inv[x]
                for y in best_add_set:
                    S.add(y)
                    _add_upd(conf, S, y)
                    weight += inv[y]
            
            if weight > best_w:
                best_w = weight
                best_S = set(S)
                no_improve_count = 0
            else:
                no_improve_count += 1
            improved = True
        else:
            improved = False
            no_improve_count += 1
    
    return best_S

def _perturb(S, rng, num_drops=3):
    """Perturb a set by randomly dropping elements"""
    pert = set(S)
    drops = min(num_drops, len(pert))
    for _ in range(drops):
        if pert:
            pert.discard(rng.choice(list(pert)))
    if not pert:
        pert = {1}
    return pert

def construct():
    start = time.time()
    deadline = start + _KILL - _MARGIN
    rng = random.Random(42)
    
    inv = [0.0] + [1.0 / x for x in range(1, _N + 1)]
    
    def w(S):
        return sum(inv[x] for x in S)
    
    inits = []
    
    # High-value biased initializations
    inits.append(_randomized_greedy(rng, 'asc', noise=0.0))
    inits.append(_randomized_greedy(rng, 'asc', noise=0.05))
    inits.append(_randomized_greedy(rng, 'asc', noise=0.1))
    inits.append(_randomized_greedy(rng, 'asc', noise=0.2))
    inits.append(_randomized_greedy(rng, 'desc', noise=0.0)) # Low-value biased, might find different structure
    inits.append(_randomized_greedy(rng, 'rand', noise=0.0))
    
    best_S = None
    best_w = -1.0
    
    for S in inits:
        if time.time() > deadline - 5.0:
            break
        opt = _hill_climb_aggressive(S, inv, deadline - 3.0, rng)
        cw = w(opt)
        if cw > best_w:
            best_w = cw
            best_S = opt
    
    # Random restarts from best
    if best_S is not None:
        restart_count = 0
        while time.time() < deadline - 2.0 and restart_count < 200:
            restart_count += 1
            pert = _perturb(best_S, rng, num_drops=rng.choice([1, 2, 3, 4]))
            opt = _hill_climb_aggressive(pert, inv, deadline - 2.0, rng)
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
    for a in res:
        for b in res:
            if a >= b:
                continue
            c = 2 * b - a
            if c in Sset and c > b:
                ok = False
    
    if not ok:
        S = _randomized_greedy(rng, 'asc', noise=0.0)
        res = sorted(S)
    
    return res
