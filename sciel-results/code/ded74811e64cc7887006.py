"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter

def _is_cap_set(vectors):
    """Check if a set of vectors in {0,1,2}^n is a cap set (no 3-term AP)."""
    n = len(vectors[0]) if vectors else 0
    # Use the property: a set S is a cap set iff for all x,y,z in S,
    # (x+y+z) mod 3 != 0 for all distinct x,y,z.
    # More efficiently: for each pair (x,y), check if z = (2*? - ?) ... 
    # Actually, 3-term AP in Z_3^n: x, y, z with x+z=2y.
    # So for each y, and each x != y, z = 2y - x (mod 3). Check if z is in set.
    vec_set = set(vectors)
    for y in vectors:
        for x in vectors:
            if x == y:
                continue
            # z = 2*y - x mod 3
            z = tuple((2*yi - xi) % 3 for xi, yi in zip(x, y))
            if z in vec_set and z != x and z != y:
                # Found a 3-term AP: x, y, z
                return False
    return True

def _has_ap(vectors, new_vec):
    """Check if adding new_vec to vectors creates a 3-term AP.
    Only need to check APs involving new_vec.
    For new_vec to be part of a 3-term AP, it must be:
    - The middle: exists x,z in vectors s.t. x+z = 2*new_vec
    - An endpoint: exists y in vectors s.t. the other endpoint z = 2*y - new_vec is in vectors (and z != new_vec, z != y)
    """
    vec_set = set(vectors)
    n = len(new_vec)
    
    # Case 1: new_vec is the middle element.
    # Need x, z in vec_set such that x + z = 2*new_vec (mod 3)
    # i.e., z = 2*new_vec - x (mod 3)
    for x in vec_set:
        z = tuple((2*new_vec[i] - x[i]) % 3 for i in range(n))
        if z in vec_set and z != x:
            # x, new_vec, z form an AP if they are distinct
            if x != new_vec and z != new_vec:
                return True
    
    # Case 2: new_vec is an endpoint.
    # AP: new_vec, y, z with y = (new_vec + z)/2, i.e., 2y = new_vec + z
    # So z = 2y - new_vec. Need y, z in vec_set, z != y, z != new_vec, y != new_vec.
    for y in vec_set:
        if y == new_vec:
            continue
        z = tuple((2*y[i] - new_vec[i]) % 3 for i in range(n))
        if z in vec_set and z != y and z != new_vec:
            return True
    
    return False

def priority(vector, n):
    """
    Greedy cap set construction with randomized restarts.
    We build a cap set by greedily adding vectors in order of priority.
    To find a good priority ordering, we use multiple random restarts with
    different random tie-breakers and weightings.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Generate all vectors in {0,1,2}^n
    all_vectors = []
    for i in range(3**n):
        v = []
        x = i
        for _ in range(n):
            v.append(x % 3)
            x //= 3
        all_vectors.append(tuple(v))
    
    best_cap_set_size = 0
    best_cap_set = set()
    
    # Try multiple random orderings
    rng = random.Random(42)
    
    # First, try a deterministic "balanced" ordering
    # Sort by sum of coordinates (to balance the 0,1,2 distribution)
    # This tends to avoid getting stuck in {0,1}^n subcubes
    balanced_order = sorted(all_vectors, key=lambda v: (sum(v), v))
    
    # Try greedy with this ordering
    cap_set = set()
    for v in balanced_order:
        if not _has_ap(list(cap_set), v):
            cap_set.add(v)
    best_cap_set_size = len(cap_set)
    best_cap_set = set(cap_set)
    
    # Now try random restarts with different random orderings
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Create a randomized ordering with some structure
        # Mix of random and structured priorities
        if attempts % 3 == 1:
            # Pure random
            order = all_vectors[:]
            rng.shuffle(order)
        elif attempts % 3 == 2:
            # Weighted random: prioritize vectors with sum close to n (balanced)
            def weight(v):
                s = sum(v)
                # Prefer sums near n (middle)
                return (s - n)**2 + rng.random() * 0.5
            order = sorted(all_vectors, key=weight)
        else:
            # Local search style: start from current best, perturb
            # Use the best cap set found so far as a guide
            # Assign higher priority to vectors that are "far" from existing cap set
            # in some sense
            if best_cap_set:
                def distance_weight(v):
                    # Count how many vectors in best_cap_set are close to v
                    # (differ in few coordinates)
                    close_count = 0
                    for cv in best_cap_set:
                        diff = sum(1 for a, b in zip(v, cv) if a != b)
                        if diff <= 1:
                            close_count += 1
                    # Prefer vectors far from existing cap set
                    return close_count + rng.random() * 0.1
                order = sorted(all_vectors, key=distance_weight)
            else:
                order = all_vectors[:]
                rng.shuffle(order)
        
        # Greedy construction
        cap_set = set()
        for v in order:
            if not _has_ap(list(cap_set), v):
                cap_set.add(v)
        
        if len(cap_set) > best_cap_set_size:
            best_cap_set_size = len(cap_set)
            best_cap_set = set(cap_set)
    
    return float(best_cap_set_size)
# EVOLVE-BLOCK-END
