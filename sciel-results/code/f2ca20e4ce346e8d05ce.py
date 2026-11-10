# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

START_TIME = time.time()
TIME_LIMIT = 590  # Safety margin


def kissing_number11() -> np.ndarray:
    """
    Returns a 2-D array of shape (k, 11) with integer entries (int64) such that:
      - second dimension is 11
      - k >= 2
      - no point is the zero vector
      - min squared pairwise distance >= max squared norm (i.e., points are 'kissing' in the strong sense)
    
    Strategy:
    1. Start with all weight-2 vectors (norm^2 = 2). There are C(11,2)*4 = 220.
       Any distinct pair has dist^2 >= 2. Since max norm^2 = 2, this is valid.
    2. Attempt to add weight-4 vectors (norm^2 = 4) using a greedy approach.
       If we add weight-4 vectors, the max norm^2 becomes 4.
       Therefore, we require min dist^2 >= 4 for ALL pairs.
       
       Pairs of weight-2 vectors: min dist^2 = 2. This FAILS the >= 4 requirement.
       So we cannot simply mix weight-2 and weight-4 sets if we keep all weight-2s.
       
       Wait, the condition is: min_{u,v in S} ||u-v||^2 >= max_{u in S} ||u||^2.
       If S contains any weight-4 vector, max_norm_sq = 4.
       Then ALL pairs in S must have dist^2 >= 4.
       But two weight-2 vectors can have dist^2 = 2.
       So if S contains two weight-2 vectors with dist^2=2, and one weight-4 vector, it is INVALID.
       
       Therefore, to use weight-4 vectors, we must ensure that NO pair of weight-2 vectors in S has dist^2 < 4.
       Actually, we must ensure NO pair in S has dist^2 < 4.
       
       This suggests we should either:
       A) Use only weight-2 vectors. (k=220)
       B) Use only weight-4 vectors. (k = ?)
       C) Use a mix, but carefully prune weight-2 vectors so that all remaining pairs have dist^2 >= 4.
       
       Let's analyze Option B: Weight-4 vectors only.
       Pool size: C(11,4) * 2^4 = 330 * 16 = 5280.
       Max norm^2 = 4.
       Need min dist^2 >= 4.
       dist^2(u,v) = 8 - 2 dot(u,v).
       Need 8 - 2 dot >= 4 => dot <= 2.
       So we need a subset of the 5280 weight-4 vectors where no two have dot product > 2.
       Dot product > 2 means dot = 3 or 4.
       dot = 4 implies u=v (same vector). We pick distinct vectors, so this is avoided.
       dot = 3 implies they share 3 indices with matching signs, and differ on the 4th (or 3 shared, 1 unique each? No, weight 4).
       If u and v share 3 indices, let the shared indices be I (|I|=3).
       u has non-zeros on I ∪ {a}, v has non-zeros on I ∪ {b}, a≠b.
       dot(u,v) = sum_{i in I} u_i v_i.
       Since u_i, v_i are ±1, the sum is in {-3, -1, 1, 3}.
       So dot=3 is possible if u_i = v_i for all i in I.
       So the constraint is: No two vectors in the set can share 3 indices with identical signs on those indices.
       
       This is a combinatorial packing problem.
       Let's try a greedy approach to find a large subset of weight-4 vectors satisfying dot <= 2.
       
       Option C: Mix.
       If we include weight-4 vectors, max_norm^2=4.
       We need all pairs (u,v) in S to have dist^2 >= 4.
       For two weight-2 vectors u,v: dist^2 = 4 - 2 dot(u,v).
       Need 4 - 2 dot >= 4 => dot <= 0.
       So if we mix, we can only include weight-2 vectors that are orthogonal to each other?
       Or more generally, the set of weight-2 vectors included must be a subset of a maximal orthogonal set?
       Max size of an orthogonal set of weight-2 vectors in 11D?
       Each vector uses 2 coordinates. Orthogonal means disjoint supports or opposite signs on shared support?
       Wait, dot=0.
       If supports are disjoint, dot=0.
       If supports share 1 index, dot = ±1. Not 0.
       So orthogonal weight-2 vectors must have disjoint supports.
       Max number of disjoint pairs in 11D is floor(11/2) = 5.
       So we can have at most 5 weight-2 vectors if we require mutual orthogonality?
       No, we require dist^2 >= 4 for ALL pairs in S.
       If u,v are weight-2, dist^2 >= 4 implies dot <= 0.
       Since dot is integer, dot <= 0.
       If they share 1 index, dot is ±1. So dot=1 is bad. dot=-1 is ok (dist^2=6).
       If they share 0 indices, dot=0 (dist^2=4).
       So we can include weight-2 vectors as long as no two have dot=1.
       Dot=1 means they share 1 index with same sign.
       This is a "conflict".
       
       However, if we have weight-4 vectors in S, we also need dist(u,v) >= 4 for u (weight 2) and v (weight 4).
       dist^2 = 2 + 4 - 2 dot = 6 - 2 dot.
       Need 6 - 2 dot >= 4 => dot <= 1.
       So weight-2 and weight-4 vectors can have dot at most 1.
       
       This mixing seems complex and might yield less than just a large pure weight-4 set, because the weight-4 pool is much larger (5280 vs 220).
       A greedy selection from 5280 might yield several hundred vectors.
       220 is the current baseline.
       If we can get > 220 from weight-4, that's better.
       
       Let's implement the greedy selection for weight-4 vectors.
       Constraint: dot(u,v) <= 2 for all u,v in S.
       
       Algorithm:
       1. Generate all 5280 weight-4 vectors.
       2. Shuffle them.
       3. Iterate through the shuffled list.
       4. For each vector v, check if dot(v, u) <= 2 for all u in current S.
       5. If yes, add v to S.
       
       Optimization:
       Checking dot(v, u) for all u in S is O(|S| * 11).
       If |S| grows to 500, this is 5500 ops per candidate.
       5280 candidates * 5500 ~ 29 million ops. Fast enough in Python with NumPy?
       Vectorized dot product: S is (M, 11), v is (11,). S @ v gives (M,).
       Check if max(S @ v) <= 2.
       This is very fast.
       
       We can try multiple random shuffles to find a better packing.
       
       Let's also consider weight-3 vectors?
       Norm^2 = 3.
       Pool: C(11,3) * 2^3 = 165 * 8 = 1320.
       Max norm^2 = 3.
       Need dist^2 >= 3.
       dist^2 = 6 - 2 dot.
       6 - 2 dot >= 3 => 2 dot <= 3 => dot <= 1.5 => dot <= 1.
       Constraint: dot(u,v) <= 1.
       This is tighter than weight-4 (dot<=2).
       Likely yields smaller sets than weight-4.
       
       Weight-5 vectors?
       Norm^2 = 5.
       Pool: C(11,5) * 2^5 = 462 * 32 = 14784.
       Max norm^2 = 5.
       Need dist^2 >= 5.
       dist^2 = 10 - 2 dot >= 5 => 2 dot <= 5 => dot <= 2.5 => dot <= 2.
       Constraint: dot(u,v) <= 2.
       Pool is larger (14784).
       This might yield even more points!
       
       Let's generalize:
       For weight-w vectors:
       Norm^2 = w.
       Pool size: C(11,w) * 2^w.
       Constraint: dist^2 >= w => 2w - 2 dot >= w => w - 2 dot >= 0 => dot <= w/2.
       Since dot is integer, dot <= floor(w/2).
       
       Let's check w=2: dot <= 1. (We found min dist^2=2, max norm^2=2. 2>=2 ok. Dot<=1 is correct?
       dist^2 = 4 - 2 dot. Need >= 2. 4-2dot >= 2 => 2dot <= 2 => dot <= 1. Correct.)
       
       w=3: dot <= 1.
       w=4: dot <= 2.
       w=5: dot <= 2.
       w=6: dot <= 3.
       w=7: dot <= 3.
       w=8: dot <= 4.
       
       Larger w means larger pool and looser constraint (relative to pool density?).
       However, higher w vectors are "denser" and might have more conflicts?
       Actually, the constraint dot <= floor(w/2) allows more overlap.
       For w=4, dot<=2 allows sharing 2 indices with matching signs (dot=2).
       For w=5, dot<=2 allows sharing 2 indices with matching signs.
       
       Let's try w=4 and w=5 and w=6 with greedy.
       w=4 pool: 5280.
       w=5 pool: 14784.
       w=6 pool: C(11,6)*64 = 462*64 = 29568.
       
       Time limit 600s.
       Greedy is O(N * M * D).
       For w=6, N=29568. If M grows to 1000, 29568 * 1000 * 11 ~ 325M ops.
       With NumPy vectorization, this might be okay, but Python loop overhead is significant.
       We can use a batch approach or just limit the number of restarts/iterations.
       
       Let's start with w=4 as it's a good balance.
       If time permits, try w=5.
       
       We will implement a function `greedy_select(pool, max_dot)` that returns a valid subset.
       We will run it on w=4 pool.
       
       To improve performance, we can precompute the dot product matrix?
       5280x5280 matrix is ~224MB. Acceptable.
       Then greedy selection is just checking rows.
       
       Let's do that.
       
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    d = 11
    
    def generate_pool(w):
        pool = []
        for idx in itertools.combinations(range(d), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = np.zeros(d, dtype=np.int64)
                for i, s in zip(idx, signs):
                    v[i] = s
                pool.append(v)
        return np.array(pool)
    
    def greedy_select(pool, max_dot, time_limit):
        n = len(pool)
        if n == 0:
            return np.empty((0, d), dtype=np.int64)
        
        # Precompute dot product matrix
        # Memory: n*n*8 bytes. For n=5280, ~224MB. OK.
        # For n=14784, ~1.7GB. Too much.
        # So for larger pools, we must compute on the fly or use a different strategy.
        # Let's stick to w=4 for now (n=5280).
        
        # Dot product matrix D where D[i,j] = dot(pool[i], pool[j])
        D = pool @ pool.T
        
        # We want to select a subset S such that for all i,j in S, D[i,j] <= max_dot.
        # Note D[i,i] = w.
        
        best_set = []
        
        # Try a few random orderings
        np.random.seed(42)
        
        for restart in range(10):
            if time.time() > time_limit:
                break
                
            order = np.random.permutation(n)
            selected = []
            selected_indices = []
            
            for idx in order:
                if time.time() > time_limit:
                    break
                
                # Check if idx is compatible with all selected
                # D[idx, selected_indices] <= max_dot
                if len(selected_indices) == 0:
                    compatible = True
                else:
                    # Vectorized check
                    dots = D[idx, selected_indices]
                    compatible = np.all(dots <= max_dot)
                
                if compatible:
                    selected.append(idx)
                    selected_indices.append(idx)
            
            if len(selected) > len(best_set):
                best_set = selected
        
        if len(best_set) == 0:
            return np.empty((0, d), dtype=np.int64)
        
        return pool[best_set]
    
    # Try weight 4
    # max_dot for w=4 is floor(4/2) = 2
    pool4 = generate_pool(4)
    result4 = greedy_select(pool4, 2, deadline)
    k4 = len(result4)
    
    # If result4 is better than 220, use it.
    # Otherwise, fall back to weight 2.
    
    if k4 > 220:
        arr = result4
    else:
        # Generate weight 2 vectors
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for si in (-1, 1):
                    for sj in (-1, 1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        points.append(v)
        arr = np.array(points, dtype=np.int64)
        
    arr = np.around(arr).astype(np.int64)
    return arr


# EVOLVE-BLOCK-END
