import itertools
import random
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))

def dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def is_isosceles(a, b, c):
    d_ab = dist_sq(a, b)
    d_ac = dist_sq(a, c)
    d_bc = dist_sq(b, c)
    return d_ab == d_ac or d_ab == d_bc or d_ac == d_bc

def is_isosceles_free(S):
    """Check if set S contains no isosceles triangles."""
    for a, b, c in itertools.combinations(S, 3):
        if is_isosceles(a, b, c):
            return False
    return True

def is_maximal(S):
    """Check if S is a maximal no-isosceles set."""
    S_set = set(S)
    for q in POINTS:
        if q in S_set:
            continue
        # Check if q can be added to S without creating an isosceles triangle
        # q can be added if for all pairs (a, b) in S, (q, a, b) is NOT isosceles
        can_add = True
        for a, b in itertools.combinations(S, 2):
            if is_isosceles(q, a, b):
                can_add = False
                break
        if can_add:
            return False
    return True

def greedy_maximal(order):
    """Construct a maximal set by adding points in the given order."""
    S = []
    for p in order:
        # Check if p can be added to S
        ok = True
        for a, b in itertools.combinations(S, 2):
            if is_isosceles(p, a, b):
                ok = False
                break
        if ok:
            S.append(p)
    return S

def minimize_maximal(S):
    """Try to remove points from maximal set S to get a smaller maximal set."""
    S = list(S)
    changed = True
    while changed:
        changed = False
        for i in range(len(S)):
            p = S[i]
            S_new = S[:i] + S[i+1:]
            if is_maximal(S_new):
                S = S_new
                changed = True
                break # Restart loop with new S
    return S

def construct():
    best_set = None
    best_size = float('inf')
    
    # Try multiple random orders to find diverse maximal sets
    for _ in range(50):
        random.shuffle(POINTS)
        S = greedy_maximal(POINTS)
        # Minimize
        S_min = minimize_maximal(S)
        if len(S_min) < best_size:
            best_size = len(S_min)
            best_set = S_min
            
    # Also try the standard row-major order as a baseline
    S = greedy_maximal(POINTS)
    S_min = minimize_maximal(S)
    if len(S_min) < best_size:
        best_size = len(S_min)
        best_set = S_min

    return best_set
