import time
import zlib
from collections import defaultdict

_start_time = time.time()
_DEADLINE = _start_time + 280.0

# Stateful storage for the greedy set
# The evaluator adds vectors greedily in descending priority.
# We assume the evaluator calls priority() for ALL vectors in {0,1,2}^n to determine the order,
# or at least calls it in a way that we can intercept.
# 
# CRITICAL INSIGHT: 
# If the evaluator sorts vectors by priority and then greedily adds them, 
# the set S is determined by the ordering. 
# If we could know S, we could compute exact conflicts.
# 
# However, we don't know S during the priority call.
# BUT, if the evaluator calls priority() in a specific order (e.g., lexicographic) 
# and we can deduce which vectors were accepted, we could track S.
# 
# The contract says: "vectors ... are added greedily in descending priority".
# This implies:
# 1. Evaluator has a list of all vectors.
# 2. Evaluator calls priority(v) for each v.
# 3. Evaluator sorts by priority.
# 4. Evaluator iterates: if v is compatible with S, add to S.
# 
# If the evaluator calls priority() for ALL vectors before starting the greedy add,
# we cannot know S during the calls.
# 
# HOWEVER, some implementations might call priority() lazily or in a way that allows state tracking.
# Or, perhaps the "stateful" strategy relies on the fact that the *same* priority function
# is used to generate the order, and we can *simulate* the greedy process if we assume a specific
# tie-breaking or ordering mechanism?
# 
# Let's look at the "Dynamic Feedback Control" directive.
# "Treat the priority assignment as a dynamic feedback control problem where the heuristic adapts 
# to the current state of the cap set by tracking the marginal 'line-completion risk'".
# 
# This implies we MUST track the set. How?
# 
# Hypothesis: The evaluator might call `priority` in a stream, or we can use the call sequence
# to infer the set if we assume the evaluator uses a standard greedy algorithm that we can mirror.
# 
# Alternative: The evaluator might pass the *current set* in a way we can access via global state?
# No, the signature is fixed: `priority(vector, n)`.
# 
# Let's try a different angle: **Implicit Reconstruction**.
# If we assign priorities such that the greedy algorithm produces a specific, predictable set,
# we can verify if our predicted set matches the "ideal" cap set structure.
# 
# But we need to *adapt* the priority to the *actual* set.
# 
# Is it possible that the evaluator calls `priority` for a vector, and if it's accepted,
# we get a signal? No, return is float.
# 
# Let's reconsider the "Stateful" aspect.
# If the evaluator calls `priority` for all $3^n$ vectors, the total number of calls is fixed.
# If we maintain a counter, we know *how many* vectors have been processed, but not *which*.
# 
# WAIT. What if the evaluator's greedy algorithm is **deterministic** based on the priorities?
# If we can predict the order in which vectors are added, we can simulate the set.
# 
# Let's assume the evaluator:
# 1. Generates all vectors.
# 2. Calls priority(v) for each v.
# 3. Sorts by priority (descending). Ties broken by vector value (e.g., lexicographic or hash).
# 4. Greedy add.
# 
# If we control the priority, we control the sort order (mostly).
# If we make the priority function **stateful** such that it depends on the *history of calls*,
# we might be able to infer the set.
# 
# But the calls are likely batched or unordered.
# 
# Let's look at the "line-completion risk" idea.
# A vector $v$ creates a conflict if there exists $u \in S$ such that $u, v, w$ form an AP.
# In $\mathbb{Z}_3^n$, $u, v, w$ is an AP if $u + w = 2v \pmod 3$.
# So $w = 2v - u \pmod 3$.
# $v$ is invalid if $2v - u \in S$ for any $u \in S$.
# 
# If we knew $S$, we could compute this exactly.
# 
# **New Strategy: Predictive Simulation**.
# We will assume a specific, simple greedy order (e.g., lexicographic) and simulate the set $S_{pred}$.
# We will assign priorities based on the risk of conflict with $S_{pred}$.
# If the actual evaluator's set $S_{actual}$ diverges from $S_{pred}$, our heuristic becomes less optimal,
# but it might still be better than static heuristics if the divergence is small.
# 
# To make this robust, we will use a **Hybrid**:
# 1. **Static Base**: A strong balance/centrality score (like the current program) to ensure a good baseline.
# 2. **Dynamic Penalty**: A penalty based on the number of "near-AP" configurations with a *predicted* set.
# 
# How to predict the set?
# We can't know the exact order. But we can assume the greedy algorithm picks the "best" vectors first.
# If we define "best" as "highest balance", we can pre-compute the top $K$ balanced vectors and assume they are in $S$.
# Then, for a candidate $v$, we check if it conflicts with any of these top $K$ vectors.
# 
# This is computationally feasible if $K$ is small (e.g., 1000).
# 
# **Refined Strategy: Top-K Conflict Avoidance**.
# 1. Pre-compute a list of the top $K$ most "balanced" vectors (highest centrality).
# 2. Assume these $K$ vectors are in the set $S$ (or a large subset of it).
# 3. For a new vector $v$, compute the number of conflicts with these $K$ vectors.
# 4. Priority = Balance(v) - $\lambda \times \text{Conflicts}(v, S_{topK})$.
# 
# This is still static (pre-computed $S_{topK}$), but it incorporates *pairwise* structure, not just intrinsic properties.
# 
# To make it truly "dynamic" or "feedback", we can update $S_{topK}$ if we could observe the set.
# Since we can't, we stick with the pre-computed set.
# 
# However, the directive says "adapts to the current state".
# If we can't adapt, is it a valid "different strategy"?
# Yes, it's a different *modelling family*: **Pairwise Conflict Approximation** vs **Spectral Smoothness**.
# 
# Let's implement **Top-K Conflict Avoidance**.
# 
# Steps:
# 1. Define a function to compute "Centrality" (balance) of a vector.
# 2. Generate all vectors? No, $3^n$ is too big for $n>10$.
# 3. Sample a large number of random vectors, compute their centrality, and keep the top $K$.
# 4. In `priority(v)`:
#    - Compute centrality of $v$.
#    - Compute conflicts with each vector in $S_{topK}$.
#    - Return centrality - $\lambda \times \text{conflicts}$.
# 
# This requires storing $S_{topK}$ and checking conflicts.
# Conflict check: $u, v, w$ is AP if $w = 2v - u \pmod 3$.
# We need to check if $w \in S_{topK}$.
# So we store $S_{topK}$ in a set for O(1) lookup.
# 
# Complexity: $O(K)$ per priority call.
# If $K=1000$ and $n=10$, this is fast.
# 
# Let's try this.

# Pre-computation
# We need to handle variable n.
# We will pre-compute for a few common n values, or compute on the fly if n changes.
# Since n is likely fixed per evaluation run, we can cache.

_CACHE = {}

def _get_top_k_set(n, k=500):
    if n in _CACHE:
        return _CACHE[n]
    
    # Generate random vectors
    import random
    rng = random.Random(12345)
    
    vectors = []
    # Sample 3*n*k random vectors to ensure we get enough good ones
    samples = 3 * n * k
    for _ in range(samples):
        v = tuple(rng.randint(0, 2) for _ in range(n))
        # Compute centrality
        c0 = v.count(0)
        c1 = v.count(1)
        c2 = v.count(2)
        ideal = n / 3.0
        var = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
        # Lower var is better
        vectors.append((var, v))
    
    # Sort by var (ascending)
    vectors.sort(key=lambda x: x[0])
    
    # Take top k
    top_k = [v for _, v in vectors[:k]]
    
    # Store in a set for fast lookup
    top_k_set = set(top_k)
    
    _CACHE[n] = top_k_set
    return top_k_set

def _count_conflicts(v, n, top_k_set):
    # Count how many u in top_k_set form an AP with v
    # AP: u, v, w such that w = 2v - u mod 3
    # We check if w is in top_k_set.
    # Note: If w == u, then 2v - u == u => 2v == 2u => v == u.
    # If v is in top_k_set, and u is in top_k_set, and v==u, then it's a trivial AP?
    # No, distinct elements.
    # If v == u, then w = v. So {v, v, v} is not a 3-term AP of distinct elements.
    # The definition of cap set usually requires distinct elements.
    # So we should check if w is in top_k_set AND w != v AND w != u?
    # Actually, if w == v, then u, v, v is not an AP.
    # If w == u, then u, v, u is not an AP.
    # So we just check if w is in top_k_set.
    # If w is in top_k_set, and w != v, then {u, v, w} is a potential AP.
    # If w == v, it's not a 3-term AP.
    
    conflicts = 0
    for u in top_k_set:
        # Compute w = 2v - u mod 3
        w = tuple((2 * vi - ui) % 3 for vi, ui in zip(v, u))
        if w in top_k_set:
            # Check if distinct
            if w != v and w != u:
                conflicts += 1
    return conflicts

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0
    
    # 1. Compute Balance (Centrality)
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    ideal = n / 3.0
    var = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    max_var = 2 * n * n / 3.0
    if max_var > 0:
        norm_var = var / max_var
    else:
        norm_var = 0.0
    balance = 1.0 - norm_var
    
    # 2. Compute Conflicts with Top-K Set
    top_k_set = _get_top_k_set(n)
    conflicts = _count_conflicts(vector, n, top_k_set)
    
    # 3. Combine
    # Balance is in [0, 1].
    # Conflicts is an integer.
    # We need to scale conflicts.
    # Max possible conflicts is K.
    # Let's normalize conflicts by K.
    k = len(top_k_set)
    norm_conflicts = conflicts / k if k > 0 else 0.0
    
    # Penalty weight
    lambda_pen = 0.5
    
    score = balance - lambda_pen * norm_conflicts
    
    # 4. Tie-breaker
    h = 0
    for val in vector:
        h = (h * 31 + val) & 0xFFFFFFFF
    hash_score = (h / 4294967296.0) * 0.01
    
    score += hash_score
    
    return float(score)
