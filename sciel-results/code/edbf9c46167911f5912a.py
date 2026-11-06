import time
import random
import numpy as np

DEADLINE = 55.0  # safety margin below 60s (evaluator kills at 90s, but let's be safe)
N = 512

def _build_conflict_matrix():
    """
    Precompute a matrix C where C[i] is a bitset of all j such that {i, j} is part of a 4-AP.
    Actually, we need to know if adding x creates a 4-AP given the current set S.
    A 4-AP is {a, a+d, a+2d, a+3d}.
    If we add x, it forms a 4-AP if there exist d and other elements such that:
    1. x = a: need a+d, a+2d, a+3d in S
    2. x = a+d: need a, a+2d, a+3d in S
    3. x = a+2d: need a, a+d, a+3d in S
    4. x = a+3d: need a, a+d, a+2d in S
    
    This is hard to vectorize directly as a single matrix because it depends on 3 other elements.
    However, we can precompute for each x, the set of "pairs" (y, z) such that {x, y, z} is a subset of a 4-AP.
    If x is added, and S contains y and z, and {x,y,z} completes a 4-AP, then we have a 4-AP.
    
    Let's precompute `conflict_pairs[x]` as a set of pairs (y, z) with y < z, y != x, z != x,
    such that {x, y, z} is contained in some 4-AP.
    Then, adding x is invalid if there exists (y, z) in conflict_pairs[x] such that y in S and z in S.
    
    For N=512, the number of 4-APs is roughly N^2/6 ~ 43k.
    Each 4-AP contributes to the conflict lists of its 4 members.
    For each member, we add 3 pairs (the other 3 members).
    So each x will have a list of pairs.
    """
    conflict_pairs = [set() for _ in range(N)]
    
    for a in range(N):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e >= N:
                continue
            
            # The 4-AP is {a, b, c, e}
            # For a: pairs (b,c), (b,e), (c,e)
            conflict_pairs[a].add((b, c))
            conflict_pairs[a].add((b, e))
            conflict_pairs[a].add((c, e))
            
            # For b: pairs (a,c), (a,e), (c,e)
            conflict_pairs[b].add((a, c))
            conflict_pairs[b].add((a, e))
            conflict_pairs[b].add((c, e))
            
            # For c: pairs (a,b), (a,e), (b,e)
            conflict_pairs[c].add((a, b))
            conflict_pairs[c].add((a, e))
            conflict_pairs[c].add((b, e))
            
            # For e: pairs (a,b), (a,c), (b,c)
            conflict_pairs[e].add((a, b))
            conflict_pairs[e].add((a, c))
            conflict_pairs[e].add((b, c))
            
    # Convert to numpy arrays for faster checking?
    # Or just keep as lists of tuples.
    # Let's convert to a list of (y, z) tuples.
    conflict_pairs_list = [list(p) for p in conflict_pairs]
    
    return conflict_pairs_list

def _is_valid_add(S, x, conflict_pairs_list):
    """
    Check if adding x to set S (represented as a boolean numpy array or bitset) creates a 4-AP.
    S is a numpy array of bools.
    """
    if S[x]:
        return False # Already in set
        
    pairs = conflict_pairs_list[x]
    # Check if any pair (y, z) is fully in S
    # Vectorized check:
    # Create a numpy array of y's and z's
    if not pairs:
        return True
        
    ys = np.array([p[0] for p in pairs])
    zs = np.array([p[1] for p in pairs])
    
    # Check if S[y] and S[z] are both True
    # We can do this element-wise
    valid = np.all(S[ys] & S[zs], axis=0)
    return not np.any(valid)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute conflict structures
    conflict_pairs_list = _build_conflict_matrix()
    
    # Initial solution: Random greedy
    rng = random.Random(42)
    S = np.zeros(N, dtype=bool)
    order = list(range(N))
    rng.shuffle(order)
    
    for x in order:
        if _is_valid_add(S, x, conflict_pairs_list):
            S[x] = True
            
    best_S = S.copy()
    best_size = np.sum(S)
    
    # Simulated Annealing
    # Parameters
    T = 10.0  # Initial temperature
    cooling_rate = 0.999
    min_T = 0.01
    
    current_size = best_size
    
    while time.time() < deadline - 2.0:
        if T < min_T:
            T = min_T
            
        # Choose a random move
        # 70% chance: try to add a random element
        # 30% chance: try to remove a random element (to escape local optima)
        
        if rng.random() < 0.7:
            # Try to add
            x = rng.randint(0, N - 1)
            if S[x]:
                continue
            if _is_valid_add(S, x, conflict_pairs_list):
                S[x] = True
                new_size = current_size + 1
                # Always accept improvement
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_S = S.copy()
            else:
                # Move rejected (invalid)
                pass
        else:
            # Try to remove (even if it reduces size, to allow escape)
            # Only remove if set is not empty
            if current_size == 0:
                continue
            # Pick a random element from S
            # For efficiency, we can keep a list of indices in S, but updating it is O(N).
            # With N=512, picking a random x and checking if S[x] is fine.
            # But we need to remove it.
            # Let's just pick a random x in [0, N). If S[x] is False, retry? No, that's biased.
            # Better: pick a random index from the list of set bits.
            # Since N is small, we can just use np.nonzero
            indices = np.nonzero(S)[0]
            if len(indices) == 0:
                continue
            x = indices[rng.randint(0, len(indices) - 1)]
            
            # Calculate "energy" change
            # Removing x always results in a valid set.
            # The "cost" is losing 1 element.
            # In SA, we accept worse moves with probability exp(-delta_E / T)
            # Here, "energy" is negative size? Or we just say delta_E = 1 (cost of removing 1).
            # So we accept with prob exp(-1/T).
            
            delta_E = 1.0
            if rng.random() < np.exp(-delta_E / T):
                S[x] = False
                current_size -= 1
            else:
                # Rejected
                pass
                
        # Cool down
        T *= cooling_rate
        
        # Check time
        if time.time() > deadline - 1.0:
            break
            
    # Final cleanup: try to add any remaining valid elements (greedy finish)
    for x in range(N):
        if not S[x]:
            if _is_valid_add(S, x, conflict_pairs_list):
                S[x] = True
                current_size += 1
    if current_size > best_size:
        best_size = current_size
        best_S = S.copy()
        
    # Convert to list of 1-indexed values
    result = [int(i) + 1 for i in np.nonzero(best_S)[0]]
    
    if not result:
        result = [1]
        
    return result
