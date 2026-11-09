# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Instead of a static heuristic based on vector features, we use a dynamic, state-aware greedy strategy.
# The priority of a vector is determined by how many conflicts (3-term APs) it would create with the *currently selected* set.
# This requires maintaining the current set of selected vectors and checking for conflicts on the fly.
# However, the `priority` function is called for each vector in a pre-sorted list? No, the task says:
# "vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set"
# This implies the evaluator sorts ALL vectors by priority first, then iterates through them, adding if valid.
# If that's the case, a dynamic state-aware strategy inside `priority` is impossible because `priority` is called
# during the sorting phase, before any vectors are added.
#
# Let's re-read carefully: "vectors ... are added greedily in descending priority"
# This usually means:
# 1. Compute priority for all vectors.
# 2. Sort vectors by priority (descending).
# 3. Iterate through sorted vectors.
# 4. If adding vector keeps set a cap set, add it.
#
# If this is the case, `priority` cannot know the current state of the set. It must be a static function of the vector.
# BUT, inspiration 3 mentions "dynamic, state-aware greedy strategy that penalizes vectors whose addition would create new 3-term arithmetic progressions with already-sel".
# This suggests that maybe the evaluator calls `priority` dynamically? Or maybe the "greedy" part is implemented by the candidate?
# No, the interface is just `priority(vector, n) -> float`. The evaluator handles the greedy selection.
# If the evaluator sorts once, then a static heuristic is all we have.
#
# However, there is another interpretation: The evaluator might provide the current set? No, the signature is fixed.
#
# Let's look at the "Inspiration 3" again. "dynamic, state-aware...". If the evaluator doesn't pass state, how can it be state-aware?
# Perhaps the "state" is implicit in the order of calls? No, sorting is batch.
#
# Wait, maybe the evaluator does NOT sort all vectors at once?
# "vectors ... are added greedily in descending priority"
# If it's a true greedy algorithm, it might pick the best available vector at each step.
# If so, `priority` might be called repeatedly, and the set of *available* vectors shrinks.
# But `priority` doesn't take the current set as an argument.
#
# Let's assume the standard FunSearch contract:
# 1. Generate all vectors.
# 2. Call `priority` for each.
# 3. Sort by priority.
# 4. Greedy add.
#
# If this is the case, a static heuristic is the only option. The current program uses a "balanced counts" heuristic.
# A materially different strategy for a static heuristic:
# Instead of balancing counts, we can use a **hash-based** or **chaotic** ordering to break ties in a way that might accidentally avoid dense conflict clusters.
# Or, we can use a **different structural feature**.
#
# Known good cap sets often resemble linear subspaces or cosets.
# The set {0,2}^n is a cap set of size 2^n.
# The set of vectors with sum 0 mod 3 is a cap set?
# Let's check: In F_3^n, a hyperplane is a cap set?
# No, a hyperplane contains lines.
#
# Let's try a heuristic based on the **sum of coordinates mod 3**.
# Vectors with the same sum mod 3 might form a cap set?
# If v1 + v2 + v3 = 0 mod 3 (coordinate-wise), and sum(v1)=s1, sum(v2)=s2, sum(v3)=s3.
# Then s1+s2+s3 = 0 mod 3.
# This doesn't prevent APs within the same sum class.
#
# Let's try a **randomized** static ordering.
# Use a deterministic hash of the vector to assign a random-looking priority.
# This might explore a different part of the solution space than the "balanced" heuristic.
# Sometimes, "random" orderings perform well in greedy set packing because they avoid the "clumping" effect of structured heuristics.
#
# We will use a high-quality hash function (like a simple polynomial hash or a cryptographic hash if available, but let's stick to simple math for speed) to generate a float in [0, 1).
# We can combine this with a weak structural bias to guide the search slightly.
#
# Strategy:
# Score = 0.9 * random_hash(vector) + 0.1 * structural_bias(vector)
#
# The random part ensures a diverse ordering.
# The structural bias can be the "balanced counts" score, but with low weight.
#
# Let's implement a fast, deterministic hash for tuples of ints.
# We can use `zlib.crc32` or just a simple polynomial rolling hash.
# `hash()` is randomized per process, which is bad for reproducibility across runs if the evaluator runs multiple times, but within a single run it's fine. However, the evaluator might run the program multiple times?
# "Keep the function names... Each evaluation must finish within 300s."
# Usually, the evaluator runs the program once per evaluation.
# To be safe, use a deterministic hash.
#
# Let's use a simple polynomial hash:
# h = 0
# for v in vector:
#     h = (h * 31 + v) % (2**32)
#
# Then map to [0, 1).
#
# We will also add a small bias towards vectors with fewer 1s, as they are generally safer.
#
# Let's refine the "materially different" aspect.
# Current: Purely structural, deterministic, based on counts.
# New: Primarily random (deterministic hash), with a slight structural bias.
# This is a different design family: "Randomized Greedy" vs "Heuristic Greedy".

def priority(vector, n):
    # Deterministic hash
    h = 0
    for v in vector:
        h = (h * 31 + v) % (2**32)
    
    # Normalize to [0, 1)
    rand_score = h / (2**32)
    
    # Structural bias: Prefer fewer 1s
    c1 = sum(1 for v in vector if v == 1)
    # Max c1 is n. Min is 0.
    # We want to prioritize low c1.
    # bias = -c1 / n
    bias = -c1 / max(1, n)
    
    # Combine
    # Weight random highly to ensure diversity
    final_score = 0.95 * rand_score + 0.05 * bias
    
    return final_score
# EVOLVE-BLOCK-END
