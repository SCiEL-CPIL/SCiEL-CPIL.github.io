"""
Cap set construction via greedy insertion with a strong, fast heuristic priority.

The evaluator calls priority(vector, n) repeatedly (presumably to order candidates)
and then constructs a cap set greedily in descending priority order, checking the
cap-set constraint at each step. The score is the final cap set size.

Key insight for improvement:
- A vector in {0,1,2}^n is part of a cap set iff no three distinct elements form an affine line, i.e., there do not exist x, y, z in the set (not necessarily distinct in the triple, but x != y != z with x + y + z = 0 mod 3 component-wise) that satisfy x + y + z ≡ 0 (mod 3) with x, y, z distinct and forming a line.
- More precisely: a set S ⊆ {0,1,2}^n is a cap set if it contains no three distinct points x, y, z such that x + y + z ≡ 0 (mod 3) component-wise. Equivalently, no x, y, z with x ≠ y ≠ z and x + y + z ≡ 0.

For the greedy algorithm to build a large cap set, we want to prioritize vectors that are "less likely to conflict" with previously added vectors. A good heuristic: prioritize vectors that have fewer potential partners that would form a line with an already-added vector.

However, since priority(vector, n) is called independently per vector (stateless), we need a static heuristic that correlates with low conflict probability.

Strong static heuristic:
- Vectors with more "balanced" symbol distribution (close to 1/3 each of 0,1,2) tend to be in larger cap sets? Not necessarily.
- Alternative: vectors that are "far" from the origin in some sense, or have high entropy.

Actually, a well-known fact: the largest cap sets in {0,1,2}^n are related to error-correcting codes. A simple but effective heuristic is to prioritize vectors by their Hamming weight relative to a reference, or by a pseudo-random hash that breaks symmetries.

But the seed returns 0.0 for all, meaning lexicographic order. We can do much better.

Better idea: Use a deterministic hash-based priority that correlates with vectors being "spread out". For example, map each vector to a scalar via a linear projection with pseudo-random coefficients, then prioritize by that scalar. This breaks the lexicographic bias and may lead to larger cap sets.

Even better: We know that the all-zeros vector is a common member of cap sets. But we can't condition on state.

Let's think differently. The greedy algorithm adds vectors in descending priority. If we assign priorities such that vectors that are "orthogonal" in some sense are preferred, we might get larger sets.

A practical approach: Use a fixed pseudo-random priority based on a hash of the vector. This is stateless, fast, and breaks symmetries. Different hash functions will yield different cap set sizes; we can pick one that works well empirically.

However, we need a single deterministic function. Let's use a simple hash: interpret the vector as a base-3 number, then use a multiplicative hash.

Actually, let's use a more principled heuristic: prioritize vectors by the number of "forbidden pairs" they would create. But we don't know the current set.

Since we can't be stateful, let's use a heuristic that is known to work well in practice for cap sets: prioritize vectors with higher "diversity" in their symbols. For example, vectors with all three symbols present (when n >= 3) are more "central" and might be less likely to form lines with other diverse vectors? This is not well-founded.

Alternative: Use a linear score with fixed coefficients. For each coordinate i, assign a weight w_i, and compute score = sum(v_i * w_i). Choose w_i to be pseudo-random but fixed. This gives a total order that is not lexicographic.

Let's implement: 
- Precompute a fixed list of weights for coordinates (using a deterministic pseudo-random sequence).
- priority(vector, n) = sum(vector[i] * weight[i] for i in range(n))

This is O(n) per call, fast, and deterministic. Different weight vectors yield different greedy outcomes. We can choose weights that are "good" by construction.

To make it even better, we can use two layers: primary sort by a hash, secondary by weight. But priority must return a single float.

Let's go with: priority = hash_value where hash_value is derived from a strong hash of the vector interpreted as base-3.

We'll use a simple but effective hash: 
  h = 0
  for v in vector:
      h = (h * 31 + v) & 0xFFFFFFFF
  return float(h)

This gives a pseudo-random ordering. Empirically, random orderings often yield larger cap sets than lexicographic because they avoid systematic conflicts.

But wait: the evaluator says "vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set". So the evaluator iterates over all 3^n vectors, sorted by priority descending, and adds each if it doesn't create a line.

So our priority function just needs to define an ordering. A good ordering is one that spreads out the selected vectors.

Let's use a deterministic pseudo-random priority based on a high-quality hash.

We'll use Python's built-in hash? No, hash is not stable across runs for tuples. We need a stable hash.

Implement a simple stable hash:

def _stable_hash(vec):
    h = 0
    for v in vec:
        h = (h * 131 + v) & 0xFFFFFFFF
    return h

Then priority = float(_stable_hash(vec))

This is O(n), fast, and deterministic.

To improve further, we can use a hash that is more uniformly distributed. Let's use a larger multiplier and mask.

Actually, let's use a better approach: map the vector to an integer in base 3, then use a multiplicative hash.

def _to_int(vec):
    val = 0
    for v in vec:
        val = val * 3 + v
    return val

Then priority = float((val * 2654435761) & 0xFFFFFFFF)

This is a common multiplicative hash.

Let's implement that.

We also need to handle the deadline. But the priority function is called many times (3^n times), and each call must be fast. Our O(n) hash is fine.

The 300s limit is for the entire evaluation, which includes the greedy construction. Our priority function should be as fast as possible.

Let's implement the multiplicative hash.
"""

def priority(vector, n):
    """
    Return a deterministic pseudo-random priority for the vector.
    
    The vector is interpreted as a base-3 number, then hashed with a multiplicative hash
    to produce a uniform-ish ordering that breaks lexicographic bias.
    """
    # Convert vector to a base-3 integer
    val = 0
    for v in vector:
        val = val * 3 + v
    
    # Multiplicative hash to spread out the values
    # 2654435761 is a large odd constant (Knuth's multiplicative hash constant)
    h = (val * 2654435761) & 0xFFFFFFFF
    
    # Return as float for the interface
    return float(h)
