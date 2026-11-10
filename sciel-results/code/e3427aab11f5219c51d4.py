"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 16
NUM_PAIRS = N * (N - 1) // 2  # 120
MAX_COLOR = 119  # colors 0..119

# Precompute all 4-cycles: for every 4-subset {a,b,c,d}, the three cycle pairings
# A 4-cycle on vertices (v0,v1,v2,v3) in order has edges (v0,v1),(v1,v2),(v2,v3),(v3,v0)
# For a 4-subset sorted as a<b<c<d, the three distinct 4-cycles (up to reversal/rotation) are:
# Cycle 1: a-b-c-d-a  -> edges (a,b),(b,c),(c,d),(d,a)
# Cycle 2: a-b-d-c-a  -> edges (a,b),(b,d),(d,c),(c,a)
# Cycle 3: a-c-b-d-a  -> edges (a,c),(c,b),(b,d),(d,a)
# We need to map vertex pairs to indices in the flat array.

def pair_index(i, j):
    """Return the index in the flat array for pair (i,j) where i<j."""
    if i > j:
        i, j = j, i
    # Pairs are ordered lexicographically: (0,1),(0,2),...,(0,15),(1,2),...
    # Index = i*N - i*(i+1)//2 + (j - i - 1)
    return i * N - i * (i + 1) // 2 + (j - i - 1)

# Precompute the three cycles for each 4-subset
# Each cycle is a list of 4 edge indices
ALL_CYCLES = []  # list of tuples of 4 edge-indices

def build_cycles():
    from itertools import combinations
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = [pair_index(a,b), pair_index(b,c), pair_index(c,d), pair_index(d,a)]
        # Cycle 2: a-b-d-c-a
        e2 = [pair_index(a,b), pair_index(b,d), pair_index(d,c), pair_index(c,a)]
        # Cycle 3: a-c-b-d-a
        e3 = [pair_index(a,c), pair_index(c,b), pair_index(b,d), pair_index(d,a)]
        cycles.append(tuple(e1))
        cycles.append(tuple(e2))
        cycles.append(tuple(e3))
    return cycles

# Build at import time
ALL_CYCLES = build_cycles()
NUM_CYCLES = len(ALL_CYCLES)

def count_colors(coloring):
    """Return the number of distinct colors used."""
    return len(set(coloring))

def is_valid(coloring):
    """Check if no 4-cycle has four distinct edge colors."""
    for cyc in ALL_CYCLES:
        c0, c1, c2, c3 = coloring[cyc[0]], coloring[cyc[1]], coloring[cyc[2]], coloring[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def check_violation(coloring, cyc):
    """Check a single cycle for violation."""
    c0, c1, c2, c3 = coloring[cyc[0]], coloring[cyc[1]], coloring[cyc[2]], coloring[cyc[3]]
    return len({c0, c1, c2, c3}) == 4

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    best_coloring = None
    best_score = 0
    
    # Strategy: Use simulated annealing / local search
    # Start with a random assignment and try to improve
    
    rng = random.Random(42)
    
    # Initial: try a structured approach first
    # Assign colors based on some pattern to maximize diversity while avoiding 4-distinct cycles
    
    def random_coloring():
        return [rng.randint(0, MAX_COLOR) for _ in range(NUM_PAIRS)]
    
    def greedy_init():
        """Greedy initialization: assign random colors, fix violations by recoloring."""
        coloring = random_coloring()
        # Try to reduce violations
        for _ in range(200):
            violations = []
            for cyc in ALL_CYCLES:
                if check_violation(coloring, cyc):
                    violations.append(cyc)
            if not violations:
                break
            # Pick a random violated cycle and change one of its edges
            cyc = rng.choice(violations)
            idx = rng.choice(cyc)
            coloring[idx] = rng.randint(0, MAX_COLOR)
        return coloring
    
    def local_search(coloring, max_iters=50000):
        """Local search: randomly change colors and accept if score improves or maintains validity."""
        coloring = list(coloring)
        current_score = count_colors(coloring)
        current_valid = is_valid(coloring)
        
        # If not valid, we need to make it valid first
        if not current_valid:
            # Try to fix violations
            for _ in range(10000):
                violations = []
                for cyc in ALL_CYCLES:
                    if check_violation(coloring, cyc):
                        violations.append(cyc)
                        break
                if not violations:
                    current_valid = True
                    current_score = count_colors(coloring)
                    break
                cyc = violations[0]
                # Change a random edge in the violated cycle
                idx = rng.choice(cyc)
                coloring[idx] = rng.randint(0, MAX_COLOR)
        
        if not current_valid:
            return coloring, current_score
        
        for it in range(max_iters):
            # Pick a random edge to recolor
            idx = rng.randint(0, NUM_PAIRS - 1)
            old_color = coloring[idx]
            new_color = rng.randint(0, MAX_COLOR)
            if new_color == old_color:
                continue
            coloring[idx] = new_color
            
            # Check if still valid
            if is_valid(coloring):
                new_score = count_colors(coloring)
                if new_score >= current_score:
                    current_score = new_score
                else:
                    # Revert
                    coloring[idx] = old_color
            else:
                # Revert
                coloring[idx] = old_color
        
        return coloring, current_score
    
    # Try multiple initializations
    num_attempts = 10
    for attempt in range(num_attempts):
        if time.time() > deadline - 5.0:
            break
        
        if attempt == 0:
            # Structured initial: use a small set of colors in a pattern
            coloring = []
            for i in range(N):
                for j in range(i+1, N):
                    coloring.append((i + 2*j) % 20)
        else:
            coloring = greedy_init()
        
        coloring, score = local_search(coloring, max_iters=30000)
        
        if is_valid(coloring):
            if score > best_score:
                best_score = score
                best_coloring = list(coloring)
    
    # If we have a valid coloring, try more aggressive local search
    if best_coloring is not None:
        # Further improvement: try to increase color count by changing edges
        # that don't create violations
        for _ in range(5):
            if time.time() > deadline - 3.0:
                break
            improved = False
            for idx in range(NUM_PAIRS):
                if time.time() > deadline - 2.0:
                    break
                old_color = best_coloring[idx]
                # Try to assign a new color not currently used
                used_colors = set(best_coloring)
                for new_color in range(MAX_COLOR + 1):
                    if new_color in used_colors:
                        continue
                    best_coloring[idx] = new_color
                    if is_valid(best_coloring):
                        improved = True
                        break
                    else:
                        best_coloring[idx] = old_color
            if not improved:
                break
    
    if best_coloring is None:
        # Fallback: all zeros (valid but score 1)
        return [0] * NUM_PAIRS
    
    return best_coloring
# EVOLVE-BLOCK-END
