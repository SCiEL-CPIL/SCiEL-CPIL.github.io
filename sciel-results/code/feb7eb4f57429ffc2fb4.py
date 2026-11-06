import time
import random
import itertools

N = 40
START_TIME = time.time()
DEADLINE = 85  # Safety margin

def elapsed():
    return time.time() - START_TIME

def _check_validity(edges, cycles):
    """Verify the solution is valid."""
    edge_set = set()
    for a, b in edges:
        if a == b:
            return False
        edge_set.add((min(a, b), max(a, b)))
    
    for k, cycle in enumerate(cycles, start=3):
        if len(cycle) != k:
            return False
        if len(set(cycle)) != k:
            return False
        for v in cycle:
            if v < 0 or v >= N:
                return False
        for i in range(k):
            u = cycle[i]
            v = cycle[(i + 1) % k]
            if (min(u, v), max(u, v)) not in edge_set:
                return False
    return True

def get_covered_lengths(S):
    """
    Given a set of chord endpoints S (integers 2..38), 
    compute the set of cycle lengths covered by:
    1. Single chords: k+1, 41-k
    2. Pairs of chords: |u-v|+2
    3. Rim: 40
    """
    S = set(S)
    covered = set()
    covered.add(N)
    
    # Single chord lengths
    for k in S:
        covered.add(k + 1)
        covered.add(N - k + 1)
        
    # Pair chord lengths
    for u, v in itertools.combinations(S, 2):
        covered.add(abs(u - v) + 2)
        
    return covered

def is_pancyclic(S):
    """Check if S covers all lengths 3..40."""
    covered = get_covered_lengths(S)
    # We need 3..39 covered by chords, 40 by rim.
    # So we just check 3..39.
    for L in range(3, N):
        if L not in covered:
            return False
    return True

def find_chord_set(m):
    """
    Find a set of m chords that makes the graph pancyclic.
    Uses randomized local search.
    """
    candidates = list(range(2, N - 1))  # 2..38
    best_S = None
    
    # Multiple restarts
    for restart in range(20000):
        if elapsed() > DEADLINE:
            break
            
        # Initialize S
        S = set(random.sample(candidates, m))
        
        # Local search
        for step in range(100):
            if is_pancyclic(S):
                return S
            
            current_covered = get_covered_lengths(S)
            missing = set(range(3, N)) - current_covered
            
            if not missing:
                return S
                
            # Try to improve by swapping one chord
            best_swap = None
            best_gain = 0
            
            # Sample a few random swaps to keep it fast
            # Or iterate all if m is small? m=9, candidates=37. 9*37=333 checks.
            # 333 * 100 steps * 20000 restarts = 666 million ops. Might be slow in Python.
            # Let's optimize: only check swaps that cover a missing length.
            
            # Pre-compute potential gains?
            # Simpler: Just do a few random swaps per step.
            
            for _ in range(10):
                if not S: continue
                k_in = random.choice(list(S))
                k_out_candidates = [x for x in candidates if x not in S]
                if not k_out_candidates: continue
                k_out = random.choice(k_out_candidates)
                
                trial_S = (S - {k_in}) | {k_out}
                trial_covered = get_covered_lengths(trial_S)
                gain = len(trial_covered - current_covered)
                
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (k_in, k_out, trial_S)
                    
            if best_swap:
                S = best_swap[2]
            else:
                # Random move to escape local optimum
                if S:
                    k_in = random.choice(list(S))
                    k_out_candidates = [x for x in candidates if x not in S]
                    if k_out_candidates:
                        k_out = random.choice(k_out_candidates)
                        S = (S - {k_in}) | {k_out}
                        
        if is_pancyclic(S):
            return S

    return None

def build_solution(S):
    """
    Construct edges and cycles from a valid chord set S.
    """
    S = sorted(S)
    
    # Build edge set
    edge_set = set()
    # Rim
    for i in range(N):
        a, b = i, (i + 1) % N
        edge_set.add((min(a, b), max(a, b)))
    # Chords
    for k in S:
        a, b = 0, k
        edge_set.add((min(a, b), max(a, b)))
        
    edges = list(edge_set)
    
    # Determine which generator covers which length
    # We need to assign each L in 3..39 to a specific (type, params)
    length_map = {}
    
    # Rim
    length_map[N] = ('rim',)
    
    # Singles
    for k in S:
        l1 = k + 1
        l2 = N - k + 1
        if l1 not in length_map:
            length_map[l1] = ('chord_short', k)
        if l2 not in length_map:
            length_map[l2] = ('chord_long', k)
            
    # Pairs
    for u, v in itertools.combinations(S, 2):
        l = abs(u - v) + 2
        if l not in length_map:
            length_map[l] = ('chord_pair', min(u, v), max(u, v))
            
    # Verify all lengths 3..40 are mapped
    for L in range(3, N + 1):
        if L not in length_map:
            return None  # Should not happen if S is valid
            
    # Build cycles
    cycles = []
    for L in range(3, N + 1):
        typ = length_map[L]
        if typ[0] == 'rim':
            cycles.append(list(range(N)))
        elif typ[0] == 'chord_short':
            k = typ[1]
            # 0, 1, ..., k
            cycles.append(list(range(0, k + 1)))
        elif typ[0] == 'chord_long':
            k = typ[1]
            # 0, N-1, ..., k
            cycles.append([0] + list(range(N - 1, k - 1, -1)))
        elif typ[0] == 'chord_pair':
            u, v = typ[1], typ[2]
            # 0, u, u+1, ..., v
            cycles.append([0, u] + list(range(u + 1, v + 1)))
            
    return {'edges': edges, 'cycles': cycles}

def construct():
    # Try m=9 first (theoretical minimum)
    # If that fails, try m=10, etc.
    for m in range(9, 15):
        if elapsed() > DEADLINE - 5:
            break
            
        S = find_chord_set(m)
        if S:
            sol = build_solution(S)
            if sol and _check_validity(sol['edges'], sol['cycles']):
                return sol
                
    # Fallback to the original 18-chord construction if search fails
    # This ensures we return a valid result
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    for k in range(2, 19):
        edges.add((0, k))
    edges.add((19, 38))
    edge_list = list(edges)

    cycles = [None] * (N - 2)
    cycles[37] = list(range(N))
    for L in range(3, 20):
        cycles[L - 3] = list(range(L))
    cycles[20 - 3] = list(range(19, 39))
    cycles[21 - 3] = [0] + list(range(2, 20)) + [38, 39]
    cycles[22 - 3] = list(range(19, -1, -1)) + [39, 38]
    for L in range(23, 40):
        k = 41 - L
        cycles[L - 3] = [0] + list(range(39, k - 1, -1))
        
    return {'edges': edge_list, 'cycles': cycles}
