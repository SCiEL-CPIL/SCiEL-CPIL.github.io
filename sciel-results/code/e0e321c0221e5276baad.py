"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import random
import time

# Precomputation
NUM_POINTS = N * N
POINTS = [(x, y) for x in range(N) for y in range(N)]

# Map point index to coordinate
# Index = x * N + y

def _dist2_idx(i, j):
    x1, y1 = i // N, i % N
    x2, y2 = j // N, j % N
    return (x1 - x2) ** 2 + (y1 - y2) ** 2

# Precompute pair indices
# We need a way to map a pair (i, j) with i < j to a unique integer index for bitset operations
# Total pairs = NUM_POINTS * (NUM_POINTS - 1) / 2
# For N=16, NUM_POINTS=256. Pairs = 256*255/2 = 32640.
# This fits in a Python integer bitset.

pair_to_idx = {}
idx_to_pair = []
idx = 0
for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        pair_to_idx[(i, j)] = idx
        idx_to_pair.append((i, j))
        idx += 1

TOTAL_PAIRS = idx

# Precompute bad_pairs_bitset for each point
# bad_pairs_bitset[q] is an integer where bit k is 1 if pair k forms an isosceles triangle with q
bad_pairs_bitset = [0] * NUM_POINTS

for q in range(NUM_POINTS):
    bitmask = 0
    for pair_idx, (a, b) in enumerate(idx_to_pair):
        if a == q or b == q:
            continue
        da = _dist2_idx(q, a)
        db = _dist2_idx(q, b)
        dab = _dist2_idx(a, b)
        if da == db or da == dab or db == dab:
            bitmask |= (1 << pair_idx)
    bad_pairs_bitset[q] = bitmask

def _get_pairs_bitset(S_set):
    """Compute the bitset of pairs present in S_set."""
    bitmask = 0
    S_list = list(S_set)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            a, b = S_list[i], S_list[j]
            if a > b:
                a, b = b, a
            bitmask |= (1 << pair_to_idx[(a, b)])
    return bitmask

def _can_add(q, S_bitset):
    """Check if q can be added to the set represented by S_bitset."""
    return (S_bitset & bad_pairs_bitset[q]) == 0

def _is_maximal(S_set, S_bitset):
    """Check if S is maximal."""
    for q in range(NUM_POINTS):
        if q not in S_set:
            if _can_add(q, S_bitset):
                return False
    return True

def _greedy_construct(random_order):
    """Construct a maximal set using greedy approach with given order."""
    S_set = set()
    S_bitset = 0
    for p in random_order:
        if _can_add(p, S_bitset):
            # Add p
            S_set.add(p)
            # Update S_bitset
            for a in S_set:
                if a == p:
                    continue
                a_idx, b_idx = a, p
                if a_idx > b_idx:
                    a_idx, b_idx = b_idx, a_idx
                S_bitset |= (1 << pair_to_idx[(a_idx, b_idx)])
    return S_set, S_bitset

def _remove_point(S_set, S_bitset, p):
    """Remove point p from S_set and update S_bitset. Returns new S_set, S_bitset."""
    new_S_set = set(S_set)
    new_S_set.remove(p)
    new_S_bitset = S_bitset
    for a in new_S_set:
        a_idx, b_idx = a, p
        if a_idx > b_idx:
            a_idx, b_idx = b_idx, a_idx
        new_S_bitset &= ~(1 << pair_to_idx[(a_idx, b_idx)])
    return new_S_set, new_S_bitset

def construct():
    """
    Find a minimal maximal no-isosceles set in [0, N)^2.
    
    Strategy: Local Search with Bitset Optimizations.
    1. Generate initial maximal set.
    2. Iteratively try to remove points (if maximality is preserved).
    3. If stuck, perform a 'kick' (remove random points, rebuild) to escape local minima.
    """
    start_time = time.time()
    timeout = 290.0  # Leave 10s buffer
    
    best_S_set = None
    best_S_bitset = None
    best_size = float('inf')
    
    random.seed(42)
    
    # Try multiple restarts
    for restart in range(20):
        if time.time() - start_time > timeout:
            break
            
        # 1. Initialize
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        S_set, S_bitset = _greedy_construct(order)
        
        if len(S_set) < best_size:
            best_size = len(S_set)
            best_S_set = set(S_set)
            best_S_bitset = S_bitset
            
        # 2. Local Search
        local_time_limit = 5.0
        local_start = time.time()
        
        while time.time() - local_start < local_time_limit:
            if time.time() - start_time > timeout:
                break
                
            # Try to remove a random point
            removed = False
            candidates = list(S_set)
            random.shuffle(candidates)
            
            for p in candidates:
                new_S_set, new_S_bitset = _remove_point(S_set, S_bitset, p)
                if _is_maximal(new_S_set, new_S_bitset):
                    # Accept move
                    S_set = new_S_set
                    S_bitset = new_S_bitset
                    if len(S_set) < best_size:
                        best_size = len(S_set)
                        best_S_set = set(S_set)
                        best_S_bitset = S_bitset
                    removed = True
                    break
            
            if not removed:
                # Stuck. Perform a kick.
                # Remove 2-4 random points, then greedily add back to make maximal
                kick_size = random.randint(2, 4)
                kick_points = random.sample(list(S_set), min(kick_size, len(S_set)))
                
                temp_S_set = set(S_set)
                temp_S_bitset = S_bitset
                for p in kick_points:
                    temp_S_set, temp_S_bitset = _remove_point(temp_S_set, temp_S_bitset, p)
                
                # Rebuild maximality
                # We need to add points to temp_S_set until it's maximal.
                # Greedy add in random order
                remaining = [q for q in range(NUM_POINTS) if q not in temp_S_set]
                random.shuffle(remaining)
                
                for q in remaining:
                    if _can_add(q, temp_S_bitset):
                        temp_S_set.add(q)
                        for a in temp_S_set:
                            if a == q:
                                continue
                            a_idx, b_idx = a, q
                            if a_idx > b_idx:
                                a_idx, b_idx = b_idx, a_idx
                            temp_S_bitset |= (1 << pair_to_idx[(a_idx, b_idx)])
                
                S_set = temp_S_set
                S_bitset = temp_S_bitset
                
                if len(S_set) < best_size:
                    best_size = len(S_set)
                    best_S_set = set(S_set)
                    best_S_bitset = S_bitset
                    
    if best_S_set is None:
        # Fallback: return a valid maximal set
        order = list(range(NUM_POINTS))
        S_set, S_bitset = _greedy_construct(order)
        best_S_set = S_set
        
    # Convert indices back to (x, y)
    return [POINTS[i] for i in best_S_set]
# EVOLVE-BLOCK-END
