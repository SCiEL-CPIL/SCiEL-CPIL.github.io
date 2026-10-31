import time
import random
import itertools

# Precomputation for the specific task C(19, 7, 4)
V = 19
K = 7
T = 4

ALL_T_SUBSETS = list(itertools.combinations(range(V), T))
T_SUBSET_INDEX = {s: i for i, s in enumerate(ALL_T_SUBSETS)}
NUM_T_SUBSETS = len(ALL_T_SUBSETS)

ALL_BLOCKS = list(itertools.combinations(range(V), K))
BLOCK_T_SUBSETS = {}
for b in ALL_BLOCKS:
    BLOCK_T_SUBSETS[b] = [T_SUBSET_INDEX[t] for t in itertools.combinations(b, T)]

# Precompute which blocks contain which t-subsets for fast greedy
T_SUBSET_TO_BLOCKS = [[] for _ in range(NUM_T_SUBSETS)]
for b in ALL_BLOCKS:
    for idx in BLOCK_T_SUBSETS[b]:
        T_SUBSET_TO_BLOCKS[idx].append(b)

def simple_greedy(v, k, t):
    """Fallback greedy for non-standard parameters."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        best_block = None
        max_cover = 0
        for b in itertools.combinations(range(v), k):
            cover_count = sum(1 for ts in itertools.combinations(b, t) if ts in uncovered)
            if cover_count > max_cover:
                max_cover = cover_count
                best_block = b
        blocks.append(best_block)
        for ts in itertools.combinations(best_block, t):
            uncovered.discard(ts)
    return [list(b) for b in blocks]

def greedy_cover_1974():
    """Optimized greedy for 19,7,4."""
    uncovered = set(range(NUM_T_SUBSETS))
    blocks = []
    while uncovered:
        # Pick a random uncovered t-subset to break symmetry
        t_idx = random.choice(list(uncovered))
        candidates = T_SUBSET_TO_BLOCKS[t_idx]
        
        best_block = None
        max_cover = 0
        for b in candidates:
            # Count how many uncovered t-subsets this block covers
            cover_count = 0
            for idx in BLOCK_T_SUBSETS[b]:
                if idx in uncovered:
                    cover_count += 1
            if cover_count > max_cover:
                max_cover = cover_count
                best_block = b
        
        blocks.append(best_block)
        for idx in BLOCK_T_SUBSETS[best_block]:
            uncovered.discard(idx)
        
    return blocks

def reduce_cover(blocks):
    """Remove redundant blocks iteratively."""
    changed = True
    while changed:
        changed = False
        cover_count = [0] * NUM_T_SUBSETS
        for b in blocks:
            for idx in BLOCK_T_SUBSETS[b]:
                cover_count[idx] += 1
        
        new_blocks = []
        for b in blocks:
            redundant = True
            for idx in BLOCK_T_SUBSETS[b]:
                if cover_count[idx] == 1:
                    redundant = False
                    break
            if not redundant:
                new_blocks.append(b)
            else:
                changed = True
        
        blocks = new_blocks
    return blocks

def construct(v, k, t):
    if (v, k, t) != (19, 7, 4):
        return simple_greedy(v, k, t)
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial Solution
    best_blocks = reduce_cover(greedy_cover_1974())
    current_blocks = best_blocks
    
    # Simulated Annealing
    T = 1.0
    min_T = 0.05
    
    while time.time() < deadline:
        # Generate Candidate
        candidate = current_blocks[:]
        
        if len(candidate) > 0:
            move_type = random.random()
            if move_type < 0.5:
                # Random Replace
                i = random.randint(0, len(candidate) - 1)
                new_block = random.choice(ALL_BLOCKS)
                candidate[i] = new_block
            else:
                # Point Swap
                i = random.randint(0, len(candidate) - 1)
                b = candidate[i]
                p = random.choice(b)
                # Pick a point not in b
                available_points = [x for x in range(V) if x not in b]
                if available_points:
                    q = random.choice(available_points)
                    new_block = tuple(sorted((set(b) - {p}) | {q}))
                    candidate[i] = new_block
        
        reduced_candidate = reduce_cover(candidate)
        
        delta = len(reduced_candidate) - len(current_blocks)
        
        # Acceptance Criterion
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            # Accept neutral moves sometimes to explore
            accept = random.random() < 0.1
        else:
            # Accept worse moves based on temperature
            if T > min_T:
                # Use exp(-delta/T)
                # delta is small (1, 2, 3...)
                accept = random.random() < pow(2.718, -delta / T)
        
        if accept:
            current_blocks = reduced_candidate
            if len(current_blocks) < len(best_blocks):
                best_blocks = current_blocks
        
        # Cool down
        T *= 0.99
        if T < min_T:
            T = 1.0
            
    return [list(b) for b in best_blocks]
