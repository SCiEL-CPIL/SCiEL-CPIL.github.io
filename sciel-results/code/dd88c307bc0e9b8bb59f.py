"""Maximize 3-APs in a 4-AP-free subset of [1,256] via simulated annealing."""
import time
import random

N = 256
DEADLINE = 85.0  # safety margin before 90s kill


def _precompute():
    # For each element i (0-indexed, value i+1), precompute:
    #   three[i] = list of (d, other) pairs such that {i, i+d, i+2d} is a 3-AP
    #   four[i]  = list of (d, other) pairs such that {i, i+d, i+2d, i+3d} is a 4-AP
    # We store the "other" element that, together with i, completes the progression.
    # For a 3-AP {a, b, c} with b = a+d, c = a+2d:
    #   if we add a, we need b and c -> other = c (and b is determined)
    #   if we add c, we need a and b -> other = a
    #   if we add b, we need a and c -> other = a (and c)
    # Simpler: for each pair (i, j) with i<j, check if they form part of a 3-AP or 4-AP.
    # Actually, let's precompute for each element i, the set of other elements that
    # would create a 3-AP or 4-AP if both are present.
    
    # For 3-APs: {a, a+d, a+2d} all in A.
    # If we add element x, a new 3-AP is created if there exist d>0 such that:
    #   x = a: need a+d and a+2d in A  (two elements)
    #   x = a+d: need a and a+2d in A  (two elements)
    #   x = a+2d: need a and a+d in A  (two elements)
    #
    # For 4-APs: {a, a+d, a+2d, a+3d} all in A.
    # If we add element x, a new 4-AP is created if there exist d>0 such that:
    #   x = a: need a+d, a+2d, a+3d in A
    #   x = a+d: need a, a+2d, a+3d in A
    #   x = a+2d: need a, a+d, a+3d in A
    #   x = a+3d: need a, a+d, a+2d in A
    
    # Precompute: for each element i (0..255), lists of "partners"
    # three_partners[i] = list of frozensets of 2 elements that with i form a 3-AP
    # four_partners[i] = list of frozensets of 3 elements that with i form a 4-AP
    
    three_partners = [[] for _ in range(N)]
    four_partners = [[] for _ in range(N)]
    
    # 3-APs: a, a+d, a+2d where a>=1, a+2d<=256
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c > N:
                break
            # 3-AP: {a, b, c}
            # Adding a requires b,c; adding b requires a,c; adding c requires a,b
            three_partners[a - 1].append((b - 1, c - 1))
            three_partners[b - 1].append((a - 1, c - 1))
            three_partners[c - 1].append((a - 1, b - 1))
    
    # 4-APs: a, a+d, a+2d, a+3d where a>=1, a+3d<=256
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            # 4-AP: {a, b, c, e}
            four_partners[a - 1].append((b - 1, c - 1, e - 1))
            four_partners[b - 1].append((a - 1, c - 1, e - 1))
            four_partners[c - 1].append((a - 1, b - 1, e - 1))
            four_partners[e - 1].append((a - 1, b - 1, c - 1))
    
    return three_partners, four_partners


def _count_3aps(A):
    """Count distinct 3-APs in set A (0-indexed values)."""
    count = 0
    for a in A:
        for d in range(1, (N - a - 1) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if b in A and c in A:
                count += 1
    return count


def _has_4ap(A):
    """Check if A contains any nonconstant 4-AP."""
    for a in A:
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if b in A and c in A and e in A:
                return True
    return False


def _initial_greedy():
    """Greedy construction: add elements that maximize 3-APs without creating 4-APs."""
    A = set()
    for x in range(N):
        # Check if adding x creates a 4-AP
        creates_4ap = False
        for (p1, p2, p3) in _four_partners[x]:
            if p1 in A and p2 in A and p3 in A:
                creates_4ap = True
                break
        if creates_4ap:
            continue
        # Check if adding x creates any 4-AP (need to check all positions)
        # Actually the above checks if x is the first element. We need to check all positions.
        # Let me re-check: _four_partners[x] contains tuples of 3 elements that with x form a 4-AP.
        # So if all 3 are in A, adding x creates a 4-AP.
        if not creates_4ap:
            A.add(x)
    return A


def _count_3aps_fast(A, three_partners):
    """Count 3-APs by checking each element's partners."""
    count = 0
    for x in A:
        for (p1, p2) in three_partners[x]:
            if p1 in A and p2 in A:
                count += 1
    return count // 3  # Each 3-AP is counted 3 times (once per element)


def construct():
    start_time = time.time()
    random.seed(42)
    
    three_partners, four_partners = _precompute()
    
    # Initial solution via greedy
    A = _initial_greedy()
    if not A:
        A = {0, 1, 2, 4}
    
    # Verify no 4-AP
    if _has_4ap(A):
        # Fallback: remove elements that are part of 4-APs
        changed = True
        while changed and _has_4ap(A):
            changed = False
            for a in sorted(A):
                for d in range(1, (N - a) // 3 + 1):
                    b = a + d
                    c = a + 2 * d
                    e = a + 3 * d
                    if b in A and c in A and e in A:
                        A.discard(e)
                        changed = True
                        break
                if changed:
                    break
    
    best_A = set(A)
    best_score = _count_3aps_fast(A, three_partners)
    
    # Simulated annealing
    T = 50.0
    cooling = 0.9995
    min_T = 0.1
    
    while time.time() - start_time < DEADLINE:
        # Pick a random move: add, remove, or swap
        move_type = random.random()
        A_copy = set(A)
        
        if move_type < 0.4:
            # Add a random element
            candidates = [x for x in range(N) if x not in A]
            if not candidates:
                continue
            x = random.choice(candidates)
            # Check 4-AP constraint
            valid = True
            for (p1, p2, p3) in four_partners[x]:
                if p1 in A and p2 in A and p3 in A:
                    valid = False
                    break
            if valid:
                A_copy.add(x)
        elif move_type < 0.7:
            # Remove a random element
            if not A:
                continue
            x = random.choice(list(A))
            A_copy.discard(x)
        else:
            # Swap: remove one, add another
            if not A:
                continue
            x = random.choice(list(A))
            A_copy.discard(x)
            candidates = [y for y in range(N) if y not in A_copy]
            if not candidates:
                continue
            y = random.choice(candidates)
            # Check 4-AP constraint for y
            valid = True
            for (p1, p2, p3) in four_partners[y]:
                if p1 in A_copy and p2 in A_copy and p3 in A_copy:
                    valid = False
                    break
            if valid:
                A_copy.add(y)
            else:
                A_copy.add(x)  # revert removal
        
        if not A_copy:
            continue
        
        new_score = _count_3aps_fast(A_copy, three_partners)
        delta = new_score - _count_3aps_fast(A, three_partners)
        
        if delta >= 0 or random.random() < min(1.0, delta / T if T > 0 else 0):
            A = A_copy
            if new_score > best_score:
                best_score = new_score
                best_A = set(A)
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    # Return as sorted list of 1-indexed values
    return sorted(x + 1 for x in best_A)


if __name__ == "__main__":
    result = construct()
    print(f"Cardinality: {len(result)}")
    print(f"Result: {result}")
    # Verify
    A0 = set(x - 1 for x in result)
    print(f"3-APs: {_count_3aps(A0)}")
    print(f"Has 4-AP: {_has_4ap(A0)}")
