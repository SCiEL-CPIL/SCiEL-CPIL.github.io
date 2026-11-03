"""Difference triangle set (7,5) - Multi-row SA with Sum-of-Entries cost."""
import random
import time
import math

_DEADLINE = 143.0  # safety margin under 150s wall-clock


def _scope(rows):
    m = 0
    for r in rows:
        if r:
            v = r[-1]
            if v > m:
                m = v
    return m


def _validate(rows):
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                d = row[j] - ai
                if d in used:
                    return False
                used.add(d)
    return True


def _greedy(n_rows, k, rng=None):
    """Greedy construction: smallest next mark with no new difference collisions."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _compute_cost(rows):
    """
    Cost = Scope + 1000 * Collisions + 0.01 * SumOfEntries
    We want to minimize scope, but sum of entries is a good proxy for compactness.
    """
    used = set()
    collisions = 0
    sum_entries = 0
    scope = 0
    for row in rows:
        if row:
            scope = max(scope, row[-1])
            sum_entries += sum(row)
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                d = row[j] - ai
                if d in used:
                    collisions += 1
                used.add(d)
    # Weighting: Scope is king. Collisions are fatal. Sum of entries guides packing.
    return scope + 1000 * collisions + 0.01 * sum_entries


def _sa_optimize(rows, n_rows, k, deadline, rng):
    """
    Simultaneous SA on all rows.
    Moves:
    1. Decrease a random element in a random row by 1.
    2. Increase a random element in a random row by 1 (rare, for escape).
    3. Swap two elements in the same row (maintains order? No, we must maintain order).
       Actually, swapping breaks the "increasing" constraint unless we re-sort, 
       but re-sorting changes differences? No, differences are invariant to order of set, 
       but the problem requires "increasing integers". 
       Wait, the problem says "list of k+1 integers that starts at 0 and increases strictly".
       So we must maintain the sorted order. 
       Instead of swapping, we can do a "shift" move: shift a subsequence.
    """
    rows = [list(r) for r in rows]
    current_cost = _compute_cost(rows)
    best_rows = [list(r) for r in rows]
    best_cost = current_cost
    
    T = 10.0
    cooling = 0.9999
    min_T = 0.01
    
    start_time = time.time()
    
    while time.time() < deadline and T > min_T:
        r = rng.randrange(n_rows)
        row = rows[r]
        
        # Choose a position to modify (1 to k, index 0 is fixed at 0)
        j = rng.randrange(1, k + 1)
        
        if j >= len(row):
            continue
            
        old_val = row[j]
        
        # Determine move: mostly decrease, sometimes increase
        if rng.random() < 0.8:
            # Decrease
            new_val = old_val - 1
            if new_val <= row[j-1]:
                # Cannot decrease below previous element
                T *= cooling
                continue
        else:
            # Increase
            new_val = old_val + 1
            if new_val >= row[j+1] if j+1 < len(row) else False:
                 # Cannot increase above next element
                 T *= cooling
                 continue
            # Also, if this was the last element, we might increase scope, which is bad, 
            # but SA allows it to escape.
            
        # Apply move
        row[j] = new_val
        new_cost = _compute_cost(rows)
        
        # Acceptance
        delta = new_cost - current_cost
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_rows = [list(x) for x in rows]
        else:
            # Revert
            row[j] = old_val
            
        T *= cooling
        
    return best_rows, best_cost


def _local_search_compress(rows, n_rows, k, deadline, rng):
    """
    Deterministic local search: try to decrease any element by 1 if it doesn't increase cost.
    This is a greedy hill-climbing step.
    """
    rows = [list(r) for r in rows]
    best_cost = _compute_cost(rows)
    best_rows = [list(r) for r in rows]
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for r in range(n_rows):
            row = rows[r]
            # Iterate from largest index to smallest to allow cascading decreases
            for j in range(k, 0, -1):
                old_val = row[j]
                new_val = old_val - 1
                if new_val <= row[j-1]:
                    continue
                
                row[j] = new_val
                new_cost = _compute_cost(rows)
                
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [list(x) for x in rows]
                    changed = True
                else:
                    # Revert if it didn't improve
                    row[j] = old_val
                    
    return best_rows, best_cost


def construct(n_rows, k):
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(12345)
    
    # 1. Initial valid solution via greedy
    rows = _greedy(n_rows, k, rng)
    best = [list(r) for r in rows]
    best_cost = _compute_cost(best)
    
    # 2. Multi-row SA with restarts
    # We allocate 80% of time to SA, 20% to local compression
    sa_deadline = start + (_DEADLINE * 0.8)
    
    # Run SA for a while, then restart if no improvement for a long time
    # Simple strategy: Run SA for the allocated time.
    # To improve on the current program, we use a different cost function and move set.
    # The current program uses a row-rebuild which is expensive. 
    # This one uses simple element shifts which are O(N) per move.
    
    improved_rows, improved_cost = _sa_optimize(rows, n_rows, k, sa_deadline, rng)
    
    if improved_cost < best_cost:
        best = improved_rows
        best_cost = improved_cost
        
    # 3. Final local compression
    if time.time() < deadline - 1:
        comp_rows, comp_cost = _local_search_compress(best, n_rows, k, deadline - 1, rng)
        if comp_cost < best_cost:
            best = comp_rows
            best_cost = comp_cost
            
    # Final validation
    if not _validate(best):
        best = _greedy(n_rows, k, rng)
        
    return best
