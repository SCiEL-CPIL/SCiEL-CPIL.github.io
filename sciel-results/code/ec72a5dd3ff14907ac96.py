"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute all 4-cycles (a,b,c,d) with a<b<c<d and the 6 edge indices
# A 4-cycle on {a,b,c,d} can be:
#   (a,b,c,d): edges ab, bc, cd, da
#   (a,b,d,c): edges ab, bd, dc, ca
#   (a,c,b,d): edges ac, cb, bd, da
# The 6 edges among 4 vertices are: ab, ac, ad, bc, bd, cd
# The 3 distinct 4-cycles (up to reversal) are:
#   Cycle 1: a-b-c-d-a  -> edges (ab, bc, cd, da)
#   Cycle 2: a-b-d-c-a  -> edges (ab, bd, dc, ca)
#   Cycle 3: a-c-b-d-a  -> edges (ac, cb, bd, da)

def _edge_idx(i, j):
    """Map pair (i,j) with i<j to index 0..119 in lex order."""
    # Lex order: (0,1),(0,2),...,(0,15),(1,2),(1,3),...
    # For i<j: index = i*(N-1) - i*(i-1)//2 + (j - i - 1)
    # Actually let's just compute directly:
    # Number of edges with first vertex < i: sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - i*(i-1)//2
    # Then among edges starting at i, the (j-i-1)-th one (0-indexed)
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Build edge index lookup
EDGE_IDX = {}
for i in range(N):
    for j in range(i + 1, N):
        EDGE_IDX[(i, j)] = _edge_idx(i, j)

# Precompute all 4-cycles: for each 4-vertex subset {a,b,c,d}, the 3 cycles
FOUR_CYCLES = []  # list of tuples of 4 edge indices
for a, b, c, d in combinations(range(N), 4):
    # Edges:
    e_ab = EDGE_IDX[(a, b)]
    e_ac = EDGE_IDX[(a, c)]
    e_ad = EDGE_IDX[(a, d)]
    e_bc = EDGE_IDX[(b, c)]
    e_bd = EDGE_IDX[(b, d)]
    e_cd = EDGE_IDX[(c, d)]
    
    # Cycle 1: a-b-c-d-a: edges ab, bc, cd, da(=ad)
    FOUR_CYCLES.append((e_ab, e_bc, e_cd, e_ad))
    # Cycle 2: a-b-d-c-a: edges ab, bd, dc(=cd), ca(=ac)
    FOUR_CYCLES.append((e_ab, e_bd, e_cd, e_ac))
    # Cycle 3: a-c-b-d-a: edges ac, cb(=bc), bd, da(=ad)
    FOUR_CYCLES.append((e_ac, e_bc, e_bd, e_ad))

NUM_CYCLES = len(FOUR_CYCLES)  # C(16,4)*3 = 1820*3 = 5460

def _count_colors(colors):
    return len(set(colors))

def _is_valid(colors):
    """Check no 4-cycle has 4 distinct edge colors."""
    for cycle in FOUR_CYCLES:
        c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def _violations(colors):
    """Return list of violating cycles."""
    vlist = []
    for cycle in FOUR_CYCLES:
        c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        if len({c0, c1, c2, c3}) == 4:
            vlist.append(cycle)
    return vlist

def _modular_init(seed_offset=0):
    """Generate a coloring using modular arithmetic."""
    colors = [0] * NUM_EDGES
    # Use c(i,j) = (i * j + seed_offset) % 19
    for i in range(N):
        for j in range(i + 1, N):
            colors[EDGE_IDX[(i, j)]] = (i * j + seed_offset) % 19
    return colors

def _local_search(colors, max_time=55.0):
    """Local search: try to fix violations by changing edge colors.
    Objective: maximize number of distinct colors while maintaining validity."""
    start_time = time.time()
    best_colors = list(colors)
    best_score = _count_colors(best_colors)
    
    # First, fix any violations
    colors = list(colors)
    for iteration in range(100000):
        if time.time() - start_time > max_time:
            break
        vlist = _violations(colors)
        if not vlist:
            break
        # Pick a random violation
        cycle = vlist[random.randint(0, len(vlist) - 1)]
        # Try changing one of the 4 edges to match another edge in the cycle
        edge_positions = [0, 1, 2, 3]
        random.shuffle(edge_positions)
        fixed = False
        for pos in edge_positions:
            ei = cycle[pos]
            # Try to set colors[ei] to one of the other three colors in this cycle
            other_colors = set()
            for p2 in range(4):
                if p2 != pos:
                    other_colors.add(colors[cycle[p2]])
            for new_c in other_colors:
                if new_c == colors[ei]:
                    continue
                old_c = colors[ei]
                colors[ei] = new_c
                # Quick check: does this create new violations?
                # For speed, just accept and recheck globally
                fixed = True
                break
            if fixed:
                break
        if not fixed:
            # Try a random color
            ei = cycle[random.randint(0, 3)]
            colors[ei] = random.randint(0, 119)
    
    # Now try to increase color diversity while maintaining validity
    if time.time() - start_time < max_time:
        current_score = _count_colors(colors)
        no_improve_count = 0
        for iteration in range(200000):
            if time.time() - start_time > max_time:
                break
            if no_improve_count > 5000:
                break
            
            # Pick a random edge
            ei = random.randint(0, NUM_EDGES - 1)
            old_c = colors[ei]
            
            # Try a new color (prefer unused colors to increase diversity)
            used = set(colors)
            if len(used) < 120:
                # Pick an unused color
                all_colors = set(range(120))
                unused = list(all_colors - used)
                new_c = random.choice(unused)
            else:
                new_c = random.randint(0, 119)
            
            if new_c == old_c:
                continue
            
            colors[ei] = new_c
            if _is_valid(colors):
                new_score = _count_colors(colors)
                if new_score > current_score:
                    current_score = new_score
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_colors = list(colors)
                else:
                    no_improve_count += 1
                    colors[ei] = old_c
            else:
                colors[ei] = old_c
                no_improve_count += 1
    
    return best_colors, best_score

def _try_multiple_starts(max_time=55.0):
    """Try multiple initial colorings and return the best valid one."""
    start_time = time.time()
    best_colors = None
    best_score = 0
    
    # Try different modular constructions
    for offset in range(20):
        if time.time() - start_time > max_time:
            break
        init_colors = _modular_init(seed_offset=offset)
        # Also try linear combinations
        init_colors2 = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                init_colors2[EDGE_IDX[(i, j)]] = (i + 3 * j + offset) % 19
        init_colors3 = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                init_colors3[EDGE_IDX[(i, j)]] = (2 * i + 5 * j + offset) % 19
        
        for init in [init_colors, init_colors2, init_colors3]:
            if time.time() - start_time > max_time:
                break
            result, score = _local_search(init, max_time=max_time - (time.time() - start_time))
            if score > best_score:
                best_score = score
                best_colors = result
    
    # If no valid result found, try a simple valid construction
    if best_colors is None or not _is_valid(best_colors):
        # Use recursive Gallai-style: partition into 2 groups of 8
        # Colors 0-9 for within-group, 10-18 for cross-group
        colors = [0] * NUM_EDGES
        group1 = list(range(8))
        group2 = list(range(8, 16))
        # Within group1: use colors 0..7
        for i, j in combinations(group1, 2):
            colors[EDGE_IDX[(i, j)]] = (i + j) % 8
        # Within group2: use colors 0..7
        for i, j in combinations(group2, 2):
            colors[EDGE_IDX[(i, j)]] = (i + j) % 8
        # Cross: use colors 8..18
        for i in group1:
            for j in group2:
                colors[EDGE_IDX[(i, j)]] = 8 + ((i + j) % 11)
        if _is_valid(colors):
            best_colors = colors
            best_score = _count_colors(colors)
        else:
            # Fallback: all same color
            best_colors = [0] * NUM_EDGES
            best_score = 1
    
    return best_colors

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    best_colors = None
    best_score = 0
    
    # Try multiple random seeds for local search
    random.seed(42)
    
    # Phase 1: Try modular constructions with local search
    for offset in range(10):
        if time.time() > deadline:
            break
        init_colors = _modular_init(seed_offset=offset)
        result, score = _local_search(init_colors, max_time=deadline - time.time())
        if score > best_score:
            best_score = score
            best_colors = result
    
    # Phase 2: Try linear combinations
    for a, b, offset in [(1, 3, 0), (2, 5, 0), (1, 5, 1), (3, 7, 2), (2, 3, 3)]:
        if time.time() > deadline:
            break
        init_colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                init_colors[EDGE_IDX[(i, j)]] = (a * i + b * j + offset) % 19
        result, score = _local_search(init_colors, max_time=deadline - time.time())
        if score > best_score:
            best_score = score
            best_colors = result
    
    # Phase 3: Randomized starts
    for trial in range(10):
        if time.time() > deadline:
            break
        init_colors = [random.randint(0, 19) for _ in range(NUM_EDGES)]
        result, score = _local_search(init_colors, max_time=deadline - time.time())
        if score > best_score:
            best_score = score
            best_colors = result
    
    # Final: ensure validity
    if best_colors is None:
        best_colors = [0] * NUM_EDGES
    
    # Final validation and fix
    if not _is_valid(best_colors):
        # Aggressive fix
        vlist = _violations(best_colors)
        for cycle in vlist:
            # Set one edge to match another
            best_colors[cycle[0]] = best_colors[cycle[1]]
        # Re-validate
        if not _is_valid(best_colors):
            # Nuclear option: reduce to a known valid coloring
            colors = [0] * NUM_EDGES
            group1 = list(range(8))
            group2 = list(range(8, 16))
            for i, j in combinations(group1, 2):
                colors[EDGE_IDX[(i, j)]] = (i + j) % 8
            for i, j in combinations(group2, 2):
                colors[EDGE_IDX[(i, j)]] = (i + j) % 8
            for i in group1:
                for j in group2:
                    colors[EDGE_IDX[(i, j)]] = 8 + ((i + j) % 11)
            best_colors = colors
    
    return best_colors
# EVOLVE-BLOCK-END
