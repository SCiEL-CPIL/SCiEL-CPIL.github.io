import time
import sys
import numpy as np

_START_TIME = time.time()
DEADLINE = _START_TIME + 310.0

# Cache for priority sets
_PRIORITY_CACHE = {}

def _int_to_tuple(idx, n):
    v = []
    x = idx
    for _ in range(n):
        v.append(x % 3)
        x //= 3
    return tuple(v)

def _tuple_to_int(v):
    idx = 0
    for i, d in enumerate(reversed(v)):
        idx += d * (3 ** i)
    return idx

def _reflection_index(v_idx, a_idx, n):
    """
    Compute the index of w where w_i = (2*v_i - a_i) mod 3.
    This is the third point in the arithmetic progression (a, v, w) in F_3^n.
    """
    w_idx = 0
    power = 1
    vi = v_idx
    ai = a_idx
    for _ in range(n):
        vd = vi % 3
        ad = ai % 3
        wd = (2 * vd - ad) % 3
        w_idx += wd * power
        vi //= 3
        ai //= 3
        power *= 3
    return w_idx

def _generate_all_vectors(n):
    """Generate all vectors in {0,1,2}^n as a list of tuples and a numpy array of indices."""
    total = 3 ** n
    vectors = []
    for i in range(total):
        vectors.append(_int_to_tuple(i, n))
    indices = np.arange(total)
    return vectors, indices

def _compute_structural_scores(n, vectors, indices):
    """
    Compute a static structural score for each vector.
    We use a heuristic based on the "balance" of digits.
    In F_3^n, a cap set avoids lines x+y+z=0.
    Vectors with many 0s or 1s might be more likely to form lines with other sparse vectors.
    However, a better static heuristic is often related to the "weight" or specific digit distributions.
    
    Let's try a score based on the count of 1s and 2s.
    Actually, a known good heuristic for greedy cap sets is to order by the number of non-zero entries,
    but that often fails. 
    
    Instead, let's use a "conflict potential" score.
    For each vector v, we can estimate how many vectors u it would conflict with if added early.
    Conflict if u, v, w are in a line. w = 2v-u.
    This is hard to compute statically for all pairs.
    
    Alternative: Use a random projection or a simple feature-based score.
    Let's stick to a robust static ordering: 
    1. Sort by the sum of digits (Hamming weight from 0).
    2. Shuffle within groups.
    
    But the prompt asks for a DIFFERENT strategy. 
    The current program uses random greedy + local search.
    The new strategy: **Deterministic Greedy with Dynamic Conflict Avoidance using a Bitmask/Count approach**.
    
    Instead of a full local search with O(N) updates, we will use a **Greedy Set Packing** approach 
    with a **Priority Queue** of "safety" scores.
    
    Actually, the most effective different strategy for cap sets is often **Randomized Greedy with a Specific Heuristic**.
    Let's implement a **Frequency-Driven Greedy**:
    1. Start with an empty set S.
    2. Maintain a count array `forbidden_count` of size 3^n.
    3. Initially, all vectors are candidates.
    4. Select the vector v that minimizes the "conflict cost".
       Conflict cost of v = sum of forbidden_count[u] for all u that would be forbidden if v is added?
       No, that's expensive.
       
    Let's try a **Simple Heuristic Greedy**:
    Order vectors by a static score derived from a small reference cap set or random projection.
    Then greedily add if valid.
    
    To make it "structurally different", we will implement a **Beam Search** style greedy.
    Maintain a set of K best partial sets? No, too complex.
    
    Let's go with **Dynamic Priority Greedy**:
    1. Precompute a random linear map (projection) to 2D or 1D.
    2. Use this to create an initial ordering.
    3. During greedy selection, re-evaluate the priority of remaining candidates based on the current set's "shadow".
    
    Actually, the simplest structural change that often works better than random+LS is **Pure Greedy with a Smart Static Order**.
    The current program spends 200s on LS. If the greedy is bad, LS might not fix it in time.
    A better static order can yield a better seed.
    
    Heuristic: Sort by the number of 1s (ascending), then number of 2s (ascending).
    Why? Vectors with more 0s are "sparser". In F_3, 0 is the identity.
    Lines often involve 0, 1, 2.
    If we pick 0-vector first, it forbids 2v for all v.
    
    Let's try a **Randomized Static Order with a Twist**:
    Generate 10 different random orders.
    For each order, run a fast greedy.
    Keep the best.
    Then, instead of complex LS, do a **Greedy Repair**:
    Try to add any remaining vector that fits.
    
    Wait, the current program already does multiple greedy seeds and LS.
    
    New Strategy: **Simulated Annealing with a Better Move Generator**.
    The current LS is "add if possible, else remove random". This is weak.
    A better LS: **Swap Move**.
    Remove v, add u.
    Check if S \ {v} U {u} is a cap set.
    This requires checking if u conflicts with any a in S\{v}.
    This is O(|S|).
    
    Let's implement **Swap-Based Local Search**.
    1. Greedy seed.
    2. Local Search:
       - Maintain `forbidden_count`.
       - At each step, pick a random v in S to remove.
       - Update `forbidden_count` (decrement).
       - Pick a random u not in S to add.
       - Check if `forbidden_count[u]` (after removal) is 0.
       - If yes, add u (increment), accept.
       - If no, reject (restore state).
       - Use Simulated Annealing to accept worse moves? No, we want max size.
       - Actually, "Remove v, Add u" keeps size constant. We need "Add u" to increase size.
       - So: 
         1. Try to add random u. If valid, do it.
         2. If stuck, remove random v.
         3. Try to add random u. If valid, do it. (Size back to original, but different set).
         4. Repeat.
         
    This is effectively what the current program does, but the "remove random" is unguided.
    
    **Key Improvement**: Remove the vector that is "most responsible" for blocking others.
    How to measure? Count how many available vectors are blocked by v.
    This is expensive to compute exactly.
    
    Alternative Structural Change: **Use a Different Representation**.
    Represent the set as a bitset (numpy array of bools).
    Use vectorized operations to check conflicts?
    Conflict check: For each v in S, compute 2v - a for all a in S.
    This is O(|S|^2 * n).
    
    Let's try a **Deterministic Greedy with a Strong Heuristic**:
    The heuristic: **Maximize the minimum distance to existing points in a random projection**.
    
    1. Generate random weights W (n x 1).
    2. Project all vectors to 1D scores.
    3. Sort by score.
    4. Greedy add:
       - Maintain a set of "blocked" vectors.
       - Iterate through the sorted list.
       - If vector is not blocked, add it and block its reflections.
       
    This is fast and deterministic given the weights.
    Run this with multiple random weight vectors.
    
    This is materially different from "Random Shuffle + LS".
    It is "Projected Greedy".
    
    Let's implement this.
    """
    pass

def _projected_greedy(n, vectors, indices, weights, time_limit):
    """
    Greedy cap set construction using a linear projection for ordering.
    weights: numpy array of shape (n,)
    """
    start = time.time()
    total = 3 ** n
    
    # Compute projection scores: score[i] = dot(vectors[i], weights)
    # vectors is a list of tuples. Convert to numpy for speed.
    # For n <= 10, total <= 59049.
    # Creating a numpy array of shape (total, n) is fast.
    vec_array = np.array(vectors, dtype=np.int8)
    scores = vec_array @ weights
    
    # Sort indices by score descending
    sorted_indices = indices[np.argsort(-scores)]
    
    S = set()
    forbidden = set()
    
    for idx in sorted_indices:
        if time.time() - start > time_limit:
            break
        if idx in forbidden:
            continue
        
        S.add(idx)
        v = vectors[idx]
        
        # Update forbidden set
        for a_idx in S:
            if a_idx == idx:
                continue
            w_idx = _reflection_index(idx, a_idx, n)
            forbidden.add(w_idx)
            
    return S

def _local_search_swap(n, initial_set, vectors, seed, time_limit):
    """
    Local search with swap moves to improve the cap set.
    """
    start = time.time()
    total = 3 ** n
    rng = random.Random(seed)
    
    S = set(initial_set)
    
    # Initialize forbidden_count
    forbidden_count = np.zeros(total, dtype=np.int32)
    
    # Rebuild forbidden_count for the initial set
    # O(|S|^2)
    for v in S:
        for a in S:
            if v == a:
                continue
            w = _reflection_index(v, a, n)
            forbidden_count[w] += 1
            
    # List of available vectors (forbidden_count == 0)
    # We will maintain a list of "candidates" to speed up addition.
    # Initially, candidates = [i for i in range(total) if i not in S and forbidden_count[i] == 0]
    # But updating this list is hard.
    
    # Strategy:
    # 1. Try to add a random vector from the "available" pool.
    #    To find available vectors efficiently, we can keep a list of all indices not in S.
    #    And check forbidden_count.
    
    all_indices = list(range(total))
    
    while time.time() - start < time_limit:
        # 1. Try to add
        added = False
        # Try a few random candidates from the full space, filtering out S
        # This is O(1) per check, but might waste time if S is large.
        # Better: Keep a list of "not in S" indices? 
        # As S grows, "not in S" shrinks.
        # Let's just sample from all_indices and check.
        
        for _ in range(50):
            v = rng.choice(all_indices)
            if v in S:
                continue
            if forbidden_count[v] == 0:
                # Add v
                S.add(v)
                for a in S:
                    if a == v:
                        continue
                    w = _reflection_index(v, a, n)
                    forbidden_count[w] += 1
                added = True
                break
        
        if added:
            continue
            
        # 2. If stuck, perform a "shake": remove a random element
        if len(S) > 0:
            victim = rng.choice(list(S))
            S.remove(victim)
            for a in S:
                w = _reflection_index(victim, a, n)
                forbidden_count[w] -= 1
                
            # Optional: Immediately try to add a different vector
            # The loop will do this in the next iteration.
            
    return S

def priority(vector, n):
    if n not in _PRIORITY_CACHE:
        start_search = time.time()
        total = 3 ** n
        
        # Generate vectors
        vectors, indices = _generate_all_vectors(n)
        
        best_set = set()
        best_size = 0
        
        # Phase 1: Projected Greedy with multiple random projections
        # This is structurally different from random shuffle.
        # It uses the geometry of the space to order vectors.
        
        num_projections = 5
        time_per_projection = 20.0
        
        for i in range(num_projections):
            if time.time() - start_search > 100.0: # 100s budget for greedy
                break
            rng = np.random.RandomState(i)
            weights = rng.randn(n)
            S_proj = _projected_greedy(n, vectors, indices, weights, time_per_projection)
            if len(S_proj) > best_size:
                best_size = len(S_proj)
                best_set = S_proj
                
        # Phase 2: Local Search with Swap/Remove
        # Use the best set from Phase 1
        # Budget: remaining time - 20s safety
        
        remaining = 300.0 - (time.time() - start_search)
        if remaining > 50.0:
            # Run LS on the best set
            # Also run LS on a few other good sets from Phase 1?
            # For now, just the best one to allow deeper search.
            
            S_ls = _local_search_swap(n, best_set, vectors, seed=42, time_limit=remaining - 20.0)
            
            if len(S_ls) > best_size:
                best_size = len(S_ls)
                best_set = S_ls
        
        # Convert best_set (indices) to set of tuples
        best_tuples = set()
        for idx in best_set:
            best_tuples.add(vectors[idx])
        
        _PRIORITY_CACHE[n] = best_tuples
    
    return 1.0 if vector in _PRIORITY_CACHE[n] else 0.0

# Need to import random for the local search
import random
