import itertools
import time
import random

N = 16
GRID = [(i, j) for i in range(N) for j in range(N)]
GRID_SET = set(GRID)
INDEX_MAP = {p: i for i, p in enumerate(GRID)}

# Precompute blockers
# Blockers[q] is a list of (idx_a, idx_b) such that {q, a, b} is isosceles
# We store indices for faster set membership if we use a bitmask or set of indices.
# Let's use a set of indices for S.

def _precompute_blockers():
    blockers = [[] for _ in range(len(GRID))]
    # Iterate all triples
    # To speed up, iterate q, then a, then b?
    # Or just iterate all combinations of 3.
    # 2.7M iterations.
    for q_idx, a_idx, b_idx in itertools.combinations(range(len(GRID)), 3):
        q = GRID[q_idx]
        a = GRID[a_idx]
        b = GRID[b_idx]
        d_qa = (q[0]-a[0])**2 + (q[1]-a[1])**2
        d_qb = (q[0]-b[0])**2 + (q[1]-b[1])**2
        d_ab = (a[0]-b[0])**2 + (a[1]-b[1])**2
        if d_qa == d_qb or d_qa == d_ab or d_qb == d_ab:
            # q is blocked by pair (a,b)
            blockers[q_idx].append((a_idx, b_idx))
            # Symmetry: a is blocked by (q,b), b is blocked by (q,a)
            # But we only need to check if q can be added to S.
            # So we only need Blockers[q].
            # However, when checking if S is maximal, we check all q not in S.
            # So we need Blockers for all q.
            # The triple {q,a,b} is isosceles.
            # This means:
            # 1. If q is outside S, and {a,b} subset S, q is blocked.
            # 2. If a is outside S, and {q,b} subset S, a is blocked.
            # 3. If b is outside S, and {q,a} subset S, b is blocked.
            # So we should add to all three.
            blockers[a_idx].append((q_idx, b_idx))
            blockers[b_idx].append((q_idx, a_idx))
    return blockers

BLOCKERS = _precompute_blockers()

def _can_add_idx(q_idx, S_set):
    """S_set is a set of indices."""
    for a_idx, b_idx in BLOCKERS[q_idx]:
        if a_idx in S_set and b_idx in S_set:
            return False
    return True

def _is_maximal_idx(S_set):
    """S_set is a set of indices."""
    for q_idx in range(len(GRID)):
        if q_idx in S_set:
            continue
        if _can_add_idx(q_idx, S_set):
            return False
    return True

def _greedy_construct_idx(order_indices):
    S_set = set()
    for p_idx in order_indices:
        if _can_add_idx(p_idx, S_set):
            S_set.add(p_idx)
    return S_set

def _minimize_idx(S_set):
    S_list = list(S_set)
    improved = True
    while improved:
        improved = False
        random.shuffle(S_list)
        for i in range(len(S_list)):
            p_idx = S_list[i]
            S_trial = set(S_list[:i] + S_list[i+1:])
            if _is_maximal_idx(S_trial):
                S_set = S_trial
                improved = True
                break
    return S_set

def construct():
    start_time = time.time()
    deadline = start_time + 300
    
    # Convert best set to indices
    best_S_set = None
    best_size = float('inf')
    
    # Initial greedy
    S_init = _greedy_construct_idx(list(range(len(GRID))))
    S_min = _minimize_idx(S_init)
    best_S_set = S_min
    best_size = len(S_min)
    
    # Randomized restarts
    while time.time() < deadline:
        random.shuffle(GRID) # This shuffles tuples, we need indices
        # Better: shuffle indices
        indices = list(range(len(GRID)))
        random.shuffle(indices)
        S_greedy = _greedy_construct_idx(indices)
        S_min = _minimize_idx(S_greedy)
        
        if len(S_min) < best_size:
            best_size = len(S_min)
            best_S_set = S_min
            
        # 2-move removal on best
        if best_size > 12: # Threshold to try 2-move
            S_best = set(best_S_set)
            improved_2 = True
            while improved_2 and time.time() < deadline:
                improved_2 = False
                S_list = list(S_best)
                random.shuffle(S_list)
                for i in range(len(S_list)):
                    for j in range(i+1, len(S_list)):
                        S_trial = set(S_list[:i] + S_list[i+1:j] + S_list[j+1:])
                        if _is_maximal_idx(S_trial):
                            S_best = S_trial
                            improved_2 = True
                            break
                    if improved_2:
                        break
            if len(S_best) < best_size:
                best_size = len(S_best)
                best_S_set = S_best

    # Convert back to tuples
    return [GRID[i] for i in best_S_set]
