import time
import math
import random
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 110.0


def _validate(classes):
    """Verify pairwise disjointness."""
    n = len(classes)
    for i in range(n):
        m1, r1 = classes[i]
        for j in range(i + 1, n):
            m2, r2 = classes[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                return False
    return True


def _score(classes):
    """Compute exact rational score, return as float."""
    if not classes:
        return 0.0
    total = Fraction(0, 1)
    for m, r in classes:
        total += Fraction(1, m)
    return float(total)


def _find_valid_residues(m, placed):
    """Find all valid residues r for modulus m given placed moduli."""
    valid = []
    for r in range(m):
        ok = True
        for m2, r2 in placed:
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                ok = False
                break
        if ok:
            valid.append(r)
    return valid


def _greedy_construct(order):
    """
    Greedy construction: place moduli in the given order, picking the residue
    that leaves the most options for remaining moduli.
    """
    placed = []
    placed_moduli = set()
    
    for m in order:
        valid_residues = _find_valid_residues(m, placed)
        if not valid_residues:
            continue
        
        # Pick the residue that maximizes flexibility
        # For each candidate residue, count how many residues remain available
        # for each unplaced modulus
        best_r = valid_residues[0]
        best_score = -1
        
        # Sample a subset of future moduli for speed
        future_moduli = [m2 for m2 in order if m2 not in placed_moduli and m2 != m]
        # Limit to first 50 for speed
        sample = future_moduli[:50]
        
        for r in valid_residues:
            score = 0
            for m2 in sample:
                # Count valid residues for m2 given placed + (m, r)
                count = 0
                for r2 in range(m2):
                    ok = True
                    # Check against new placement
                    g = math.gcd(m, m2)
                    if r % g == r2 % g:
                        ok = False
                    else:
                        # Check against existing
                        for m3, r3 in placed:
                            g2 = math.gcd(m2, m3)
                            if r2 % g2 == r3 % g2:
                                ok = False
                                break
                    if ok:
                        count += 1
                if count == 0:
                    score -= 1000
                else:
                    score += count / m2  # more options is better, weighted by 1/m2
            if score > best_score:
                best_score = score
                best_r = r
        
        placed.append((m, best_r))
        placed_moduli.add(m)
    
    return [[m, r] for m, r in placed]


def _local_search(classes, rng, max_iters=5000):
    """
    Local search: try adding, removing, or swapping moduli to improve the score.
    """
    current = set(classes)
    current_score = _score(list(current))
    best = set(current)
    best_score = current_score
    
    all_moduli = list(range(20, 201))
    
    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
        
        # Try adding a random unplaced modulus
        unplaced = [m for m in all_moduli if m not in current]
        if unplaced:
            m = rng.choice(unplaced)
            valid_residues = _find_valid_residues(m, list(current))
            if valid_residues:
                r = rng.choice(valid_residues)
                new_set = current | {(m, r)}
                # Check all pairwise
                if _validate(list(new_set)):
                    new_score = _score(list(new_set))
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        if current_score > best_score:
                            best = set(current)
                            best_score = current_score
                        continue
        
        # Try removing a random placed modulus
        if len(current) > 1:
            to_remove = rng.choice(list(current))
            new_set = current - {to_remove}
            new_score = _score(list(new_set))
            if new_score > current_score:
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best = set(current)
                    best_score = current_score
                continue
        
        # Try swapping: remove one, add another
        if current and unplaced:
            to_remove = rng.choice(list(current))
            to_add = rng.choice(unplaced)
            temp_set = current - {to_remove}
            valid_residues = _find_valid_residues(to_add, list(temp_set))
            if valid_residues:
                r = rng.choice(valid_residues)
                new_set = temp_set | {(to_add, r)}
                if _validate(list(new_set)):
                    new_score = _score(list(new_set))
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        if current_score > best_score:
                            best = set(current)
                            best_score = current_score
                        continue
    
    return list(best)


def _greedy_with_backtrack(order, depth_limit=15):
    """
    Greedy with limited backtracking for the first few placements.
    """
    best_result = []
    best_score = 0.0
    
    def backtrack(idx, placed, placed_moduli, current_score):
        nonlocal best_result, best_score
        
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            return
        
        # Upper bound: current + sum of 1/m for all remaining moduli
        remaining_sum = sum(1.0 / m for m in order[idx:] if m not in placed_moduli)
        if current_score + remaining_sum <= best_score:
            return
        
        if idx >= len(order):
            if current_score > best_score:
                best_score = current_score
                best_result = list(placed)
            return
        
        m = order[idx]
        valid_residues = _find_valid_residues(m, placed)
        
        if not valid_residues:
            backtrack(idx + 1, placed, placed_moduli, current_score)
            return
        
        # Limit branching for deeper levels
        if idx < depth_limit:
            candidates = valid_residues
        else:
            # Just pick the first valid residue
            candidates = valid_residues[:1]
        
        for r in candidates:
            placed.append((m, r))
            placed_moduli.add(m)
            backtrack(idx + 1, placed, placed_moduli, current_score + 1.0 / m)
            placed.pop()
            placed_moduli.discard(m)
    
    backtrack(0, [], set(), 0.0)
    return best_result


def construct():
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return [[20 * j, j - 1] for j in range(1, 11)]

    all_moduli = list(range(20, 201))
    
    best_classes = None
    best_score_val = 0.0
    
    # Strategy 1: Greedy in ascending order (small moduli first, highest value)
    orders = [
        sorted(all_moduli),                # ascending
        sorted(all_moduli, reverse=True),  # descending
    ]
    
    for order in orders:
        elapsed = time.time() - _START_TIME
        if elapsed > _TIME_LIMIT - 10:
            break
        classes = _greedy_construct(order)
        if classes and _validate(classes):
            s = _score(classes)
            if s > best_score_val:
                best_score_val = s
                best_classes = classes
    
    # Strategy 2: Greedy with backtracking on ascending order
    elapsed = time.time() - _START_TIME
    if elapsed < _TIME_LIMIT - 20:
        classes = _greedy_with_backtrack(sorted(all_moduli), depth_limit=12)
        if classes and _validate(classes):
            s = _score(classes)
            if s > best_score_val:
                best_score_val = s
                best_classes = classes
    
    # Strategy 3: Randomized restarts with local search
    rng = random.Random(42)
    max_restarts = 30
    
    for restart in range(max_restarts):
        elapsed = time.time() - _START_TIME
        if elapsed > _TIME_LIMIT - 10:
            break
        
        # Random order biased toward small moduli
        order = all_moduli[:]
        # Partial shuffle: keep first 30 in order, shuffle the rest
        k = 30
        order[:k] = sorted(order[:k])
        rng.shuffle(order[k:])
        
        classes = _greedy_construct(order)
        if classes and _validate(classes):
            # Local search improvement
            improved = _local_search(classes, rng, max_iters=2000)
            if improved and _validate(improved):
                s = _score(improved)
                if s > best_score_val:
                    best_score_val = s
                    best_classes = improved
            else:
                s = _score(classes)
                if s > best_score_val:
                    best_score_val = s
                    best_classes = classes
    
    # Strategy 4: More aggressive local search on the best solution
    elapsed = time.time() - _START_TIME
    if best_classes is not None and elapsed < _TIME_LIMIT - 5:
        rng2 = random.Random(123)
        improved = _local_search(best_classes, rng2, max_iters=5000)
        if improved and _validate(improved):
            s = _score(improved)
            if s > best_score_val:
                best_score_val = s
                best_classes = improved
    
    if best_classes is not None:
        return best_classes
    
    # Fallback: simple valid construction
    return [[20 * j, j - 1] for j in range(1, 11)]
