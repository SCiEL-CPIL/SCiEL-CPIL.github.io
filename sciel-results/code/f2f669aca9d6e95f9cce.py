import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def get_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_edges():
    """Return list of (i, j) for all edges in lex order."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def build_cycles():
    """
    For each 4-vertex subset, generate the 3 cycles.
    Each cycle is a list of 4 edge indices.
    """
    from itertools import combinations
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = [get_idx(a,b), get_idx(b,c), get_idx(c,d), get_idx(d,a)]
        # Cycle 2: a-b-d-c-a
        e2 = [get_idx(a,b), get_idx(b,d), get_idx(d,c), get_idx(c,a)]
        # Cycle 3: a-c-b-d-a
        e3 = [get_idx(a,c), get_idx(c,b), get_idx(b,d), get_idx(d,a)]
        cycles.append((e1, e2, e3))
    return cycles

def build_edge_to_cycles():
    """For each edge index, list of (cycle_idx, position_in_cycle)."""
    cycles = build_cycles()
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, (c1, c2, c3) in enumerate(cycles):
        for pos, ei in enumerate(c1):
            edge_to_cycles[ei].append((ci, 0, pos))
        for pos, ei in enumerate(c2):
            edge_to_cycles[ei].append((ci, 1, pos))
        for pos, ei in enumerate(c3):
            edge_to_cycles[ei].append((ci, 2, pos))
    return cycles, edge_to_cycles

def check_cycle_valid(colors, edge_indices):
    """Check that 4 edges don't all have distinct colors."""
    c = [colors[e] for e in edge_indices]
    return len(set(c)) < 4

def is_valid(colors, cycles):
    """Check all cycles are valid."""
    for (c1, c2, c3) in cycles:
        if not check_cycle_valid(colors, c1):
            return False
        if not check_cycle_valid(colors, c2):
            return False
        if not check_cycle_valid(colors, c3):
            return False
    return True

def count_distinct(colors):
    return len(set(colors))

def try_assign_color(colors, edge_idx, new_color, cycles, edge_to_cycles):
    """
    Try to set colors[edge_idx] = new_color.
    Check if this would violate any cycle.
    Returns True if valid, False otherwise.
    """
    # Check all cycles containing this edge
    for (ci, which, pos) in edge_to_cycles[edge_idx]:
        c1, c2, c3 = cycles[ci]
        if which == 0:
            cycle = c1
        elif which == 1:
            cycle = c2
        else:
            cycle = c3
        # Simulate the change
        old_color = colors[edge_idx]
        colors[edge_idx] = new_color
        if not check_cycle_valid(colors, cycle):
            colors[edge_idx] = old_color
            return False
        colors[edge_idx] = old_color
    return True

def greedy_fill(colors, cycles, edge_to_cycles, deadline):
    """
    Greedily introduce new colors by trying to assign them to edges.
    Returns the set of used colors.
    """
    used_colors = set(colors)
    
    for new_color in range(MAX_COLOR):
        if time.time() > deadline:
            break
        if new_color in used_colors:
            continue
        
        # Try to assign this color to some edge
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        for ei in edge_order:
            if time.time() > deadline:
                break
            if colors[ei] == new_color:
                continue
            if try_assign_color(colors, ei, new_color, cycles, edge_to_cycles):
                colors[ei] = new_color
                used_colors.add(new_color)
                break
    
    return used_colors

def local_search(colors, cycles, edge_to_cycles, deadline):
    """
    Local search to find more unused colors.
    Tries to recolor edges to introduce new colors.
    Returns the set of used colors.
    """
    used_colors = set(colors)
    
    # Multiple passes with different strategies
    max_passes = 50
    for pass_num in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        for ei in edge_order:
            if time.time() > deadline:
                break
            
            current_color = colors[ei]
            # Try to change to an unused color
            unused = [c for c in range(MAX_COLOR) if c not in used_colors]
            random.shuffle(unused)
            
            for new_color in unused:
                if try_assign_color(colors, ei, new_color, cycles, edge_to_cycles):
                    colors[ei] = new_color
                    used_colors.add(new_color)
                    improved = True
                    break
        
        if not improved:
            break
    
    return used_colors

def multi_restart_search(cycles, edge_to_cycles, deadline):
    """
    Multiple restarts with different initial colorings.
    Returns the best coloring found.
    """
    best_colors = None
    best_count = 0
    
    # Budget for multiple restarts
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initial strategies
        if restart == 0:
            # Start with all zeros
            colors = [0] * NUM_EDGES
        elif restart == 1:
            # Start with a few base colors
            colors = [0] * NUM_EDGES
            # Assign some edges to color 1
            for i in range(0, NUM_EDGES, 7):
                colors[i] = 1
        elif restart == 2:
            # Random initialization with few colors
            colors = [random.randint(0, 2) for _ in range(NUM_EDGES)]
        elif restart == 3:
            # Block-based initialization
            colors = [0] * NUM_EDGES
            for i in range(0, NUM_EDGES, 4):
                colors[i] = (i // 4) % 3
        else:
            # More random
            colors = [random.randint(0, 4) for _ in range(NUM_EDGES)]
        
        # Validate and fix if needed
        if not is_valid(colors, cycles):
            # Fall back to all zeros
            colors = [0] * NUM_EDGES
        
        # Greedy fill
        used_colors = greedy_fill(colors, cycles, edge_to_cycles, deadline)
        
        # Local search
        used_colors = local_search(colors, cycles, edge_to_cycles, deadline)
        
        # Track best
        current_count = len(used_colors)
        if current_count > best_count:
            best_count = current_count
            best_colors = colors[:]
    
    if best_colors is None:
        best_colors = [0] * NUM_EDGES
    
    return best_colors

def construct():
    deadline = time.time() + 85  # 85s deadline, 5s safety margin
    random.seed(42)
    
    edges = build_edges()
    cycles, edge_to_cycles = build_edge_to_cycles()
    
    # Use multi-restart search
    best_colors = multi_restart_search(cycles, edge_to_cycles, deadline)
    
    return best_colors

# Pre-compute for efficiency
_cycles, _edge_to_cycles = build_edge_to_cycles()
