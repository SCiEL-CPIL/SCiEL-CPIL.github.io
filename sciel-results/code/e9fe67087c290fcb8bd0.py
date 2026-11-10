# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Strategy:
    1. Construct a large initial set S (the complement of the Nikodym set N) using a
       Hermitian-like curve (Unital) in F_49^2. This ensures that no line contains too
       many points of S, making it easier to satisfy the Nikodym condition.
    2. Refine S using a greedy algorithm to maximize |S|.
       The condition for S to be valid is that for every point x in F_49^2,
       there exists a line L through x such that L \ {x} is disjoint from S.
       Equivalently, S is not a blocking set for the lines through any x.
       
       We maintain a count of valid lines for each point.
       A line L through x is valid if L \ {x} is disjoint from S.
       
       Greedy Step:
       - Try to add a point p to S.
       - This invalidates the line (x, p) for all x != p.
       - So valid_count[x] decreases by 1 for all x != p.
       - If min(valid_count) > 0, keep p. Else, revert.
       
       To be faster, we can use the fact that we only need to check if the minimum
       valid count drops to 0.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    p_prime = 7
    q = 49
    
    # Field F_49 = F_7[alpha] with alpha^2 = 3
    # Elements are (a, b) representing a + b*alpha
    # Multiplication: (a, b) * (c, d) = (ac + 3bd, ad + bc)
    
    def field_mul(a, b):
        # a, b are tuples (a1, a2), (b1, b2)
        # Returns (a1*b1 + 3*a2*b2, a1*b2 + a2*b1) mod 7
        return ((a[0]*b[0] + 3*a[1]*b[2]) % 7, (a[0]*b[1] + a[1]*b[0]) % 7)
        
    # Wait, indices are 0..2400.
    # Point i -> (xa, xb, ya, yb)
    # Let's map points to indices for fast lookup.
    
    # Precompute field multiplication table for speed
    # Map (a, b) to index a*7+b
    # mult_table[i, j] = k such that elem_i * elem_j = elem_k
    mult_table = np.zeros((49, 49), dtype=np.int32)
    for i in range(49):
        a1 = i // 7
        a2 = i % 7
        for j in range(49):
            b1 = j // 7
            b2 = j % 7
            c1 = (a1*b1 + 3*a2*b2) % 7
            c2 = (a1*b2 + a2*b1) % 7
            mult_table[i, j] = c1*7 + c2

    # Inverse table
    inv_table = np.zeros(49, dtype=np.int32)
    for i in range(1, 49):
        # Find j such that i * j = 1 (index of 1 is 1*7+0 = 7? No. 1 + 0*alpha = 1. Index 1*7+0 = 7? 
        # Wait. a + b*alpha. 1 = 1 + 0*alpha. Index = 1*7 + 0 = 7.
        # 0 = 0 + 0*alpha. Index 0.
        for j in range(1, 49):
            if mult_table[i, j] == 7:
                inv_table[i] = j
                break

    # Points in F_49^2
    # Index = ((xa*7 + xb)*7 + ya)*7 + yb
    # xa, xb, ya, yb in 0..6
    
    # We will work with indices 0..2400.
    
    # Precompute lines?
    # A line is defined by (slope m, intercept c) or vertical.
    # We need to update valid_count for all points on lines through p.
    # Actually, as derived before, adding p to S decreases valid_count[x] by 1 for ALL x != p.
    # Why?
    # valid_count[x] = number of lines L through x such that L \ {x} is disjoint from S.
    # If we add p to S:
    #   For any line L through x:
    #   If p not in L, status unchanged.
    #   If p in L and p != x:
    #     Old: L \ {x} disjoint from S_old.
    #     New: L \ {x} disjoint from S_new = S_old U {p}.
    #     Since p in L and p != x, p is in L \ {x}.
    #     So L \ {x} is NOT disjoint from S_new.
    #     So L becomes invalid.
    #   How many lines through x contain p?
    #   Exactly 1 (the line connecting x and p).
    #   So valid_count[x] decreases by 1 for all x != p.
    
    # So the update is global: valid_count -= 1, then valid_count[p] += 1.
    # Check min(valid_count) > 0.
    
    # Initial S:
    # We want a large initial S to start with.
    # If we start with S = empty, valid_count = 50.
    # We can add points until min(valid_count) == 0.
    # The maximum size of S is bounded by the condition that no point is blocked.
    # If we use a random order, we might get a decent size.
    # But we can do better with a structured initial set.
    
    # Let's try to construct a large "independent" set first?
    # No, the condition is not independence.
    
    # Let's just use the greedy algorithm from empty S, but with a good ordering.
    # Or, we can start with a known valid set and try to add more.
    
    # Let's try the greedy from empty, but process points in a specific order.
    # Points in F_49^2.
    # Order: random?
    
    rng = np.random.RandomState(123)
    candidates = np.arange(2401)
    rng.shuffle(candidates)
    
    valid_count = np.full(2401, 50, dtype=np.int32)
    in_S = np.zeros(2401, dtype=bool)
    
    # Greedy Phase 1: Add points
    for p_idx in candidates:
        if time.time() > deadline:
            break
            
        valid_count -= 1
        valid_count[p_idx] += 1
        
        if np.min(valid_count) > 0:
            in_S[p_idx] = True
        else:
            valid_count += 1
            valid_count[p_idx] -= 1
    
    # Now we have a maximal S with respect to this order.
    # Can we improve?
    # Try to swap points?
    # Or try different orderings?
    
    # Let's try a second pass with a different random seed or order.
    # But we are limited by time.
    
    # Let's try to "shake" the set.
    # Remove a point, try to add others?
    # This is complex.
    
    # Alternative:
    # The greedy algorithm is O(N^2) if done naively?
    # No, the update is O(N) (vectorized).
    # 2401 iterations * O(2401) operations = ~6M operations.
    # This is very fast in numpy.
    
    # Let's run multiple rounds with different random orders and keep the best S.
    # Since the update is fast, we can run many rounds.
    
    best_S_size = np.sum(in_S)
    best_in_S = in_S.copy()
    
    for seed in range(100):
        if time.time() > deadline:
            break
            
        rng = np.random.RandomState(seed)
        candidates = np.arange(2401)
        rng.shuffle(candidates)
        
        valid_count = np.full(2401, 50, dtype=np.int32)
        in_S = np.zeros(2401, dtype=bool)
        
        for p_idx in candidates:
            valid_count -= 1
            valid_count[p_idx] += 1
            
            if np.min(valid_count) > 0:
                in_S[p_idx] = True
            else:
                valid_count += 1
                valid_count[p_idx] -= 1
        
        size = np.sum(in_S)
        if size > best_S_size:
            best_S_size = size
            best_in_S = in_S.copy()
            
    # Use best_in_S
    in_S = best_in_S
    
    # Construct N (complement of S)
    n_points = np.sum(~in_S)
    
    result_list = []
    for i in range(2401):
        if not in_S[i]:
            yb = i % 7
            ya = (i // 7) % 7
            xb = (i // 49) % 7
            xa = (i // 343) % 7
            result_list.append([
                [xa, xb],
                [ya, yb]
            ])
            
    if len(result_list) == 0:
        return np.empty((0, 2, 2), dtype=np.int64)
        
    result = np.array(result_list, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
