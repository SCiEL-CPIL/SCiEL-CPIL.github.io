# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 575.0  # safety margin before 600s limit


def _compute_forbidden(m):
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    return forbidden


def _squares_mod(p):
    """Return set of nonzero quadratic residues mod p."""
    return {pow(x, 2, p) for x in range(1, p)}


def _is_conflict_fast(a, b, sq5, sq41):
    """Fast conflict check using CRT decomposition."""
    d = (a - b) % 205
    d5 = d % 5
    if d5 == 0 or d5 not in sq5:
        return False
    d41 = d % 41
    return d41 in sq41


def _greedy_with_order(m, sq5, sq41, rng, order):
    """Greedy with specified order."""
    selected = set()
    for cand in order:
        ok = True
        for s in selected:
            if _is_conflict_fast(cand, s, sq5, sq41):
                ok = False
                break
        if ok:
            selected.add(cand)
    return selected


def _local_search(m, sq5, sq41, selected, rng, max_rounds=200):
    """Local search: additions and 1-swaps."""
    selected = set(selected)
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        cur_size = len(selected)
        # Try pure additions
        for cand in range(m):
            if cand in selected:
                continue
            ok = True
            for s in selected:
                if _is_conflict_fast(cand, s, sq5, sq41):
                    ok = False
                    break
            if ok:
                selected.add(cand)
                improved = True
                break
        if improved:
            continue
        # Try 1-swap
        sel_list = list(selected)
        rng.shuffle(sel_list)
        cand_list = list(range(m))
        rng.shuffle(cand_list)
        for s in sel_list:
            if len(selected) <= cur_size:
                break
            remaining = selected - {s}
            for cand in cand_list:
                if cand in remaining:
                    continue
                ok = True
                for r in remaining:
                    if _is_conflict_fast(cand, r, sq5, sq41):
                        ok = False
                        break
                if ok:
                    selected = remaining | {cand}
                    improved = True
                    break
            if improved:
                break
    return selected


def _generate_orders(m, rng, n_orders=100):
    """Generate diverse orderings for greedy."""
    orders = []
    # Random orders
    for _ in range(50):
        order = list(range(m))
        rng.shuffle(order)
        orders.append(order)
    
    # Degree-based orders (compute conflict degrees)
    degrees = [0] * m
    for i in range(m):
        for j in range(i + 1, m):
            if _is_conflict_fast(i, j, _squares_mod(5), _squares_mod(41)):
                degrees[i] += 1
                degrees[j] += 1
    
    # Sort by degree descending
    order_deg = sorted(range(m), key=lambda x: -degrees[x])
    orders.append(order_deg)
    orders.append(order_deg[::-1])
    
    # Shuffled degree orders
    for _ in range(48):
        order = order_deg[:]
        rng.shuffle(order)
        orders.append(order)
    
    return orders


def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    forbidden = _compute_forbidden(m)
    sq5 = _squares_mod(5)
    sq41 = _squares_mod(41)

    rng = random.Random(12345)
    best = set()
    
    # Precompute orders
    orders = _generate_orders(m, rng, n_orders=100)
    
    # Run greedy + local search with various orders
    for i, order in enumerate(orders):
        if time.time() > _DEADLINE - 10:
            break
        sel = _greedy_with_order(m, sq5, sq41, rng, order)
        sel = _local_search(m, sq5, sq41, sel, rng, max_rounds=100)
        if len(sel) > len(best):
            best = set(sel)
    
    # Additional random restarts if time remains
    extra = 0
    while time.time() < _DEADLINE - 5.0 and extra < 100:
        extra += 1
        order = list(range(m))
        rng.shuffle(order)
        sel = _greedy_with_order(m, sq5, sq41, rng, order)
        sel = _local_search(m, sq5, sq41, sel, rng, max_rounds=50)
        if len(sel) > len(best):
            best = set(sel)

    return np.array(sorted(best), dtype=np.int64)


# EVOLVE-BLOCK-END
