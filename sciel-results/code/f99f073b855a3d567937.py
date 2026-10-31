"""Program for kissing number in dimension 11: maximize k nonzero integer vectors
with min pairwise distance >= max norm.

Strategy: Start with the D11 root lattice vectors (weight-2, ±1 signs) giving 220 vectors.
Then greedily add weight-3 binary vectors (all +1 on a 3-subset) that are compatible
with all existing vectors and each other. The key insight is that weight-3 vectors with
all +1 signs have dot product <= 1 with any weight-2 vector (since they share at most
2 indices, and the signs align to give dot product 0, 1, or 2... wait, if a weight-3
vector has +1 at indices {a,b,c} and a weight-2 vector has +1 at {a,b}, then dot = 2.
That violates the constraint. So we need to be more careful.

Actually, let me reconsider. For a weight-3 vector u (all +1 on support S_u of size 3)
and a weight-2 vector v (±1 on support S_v of size 2):
u·v = sum over indices in S_u ∩ S_v of (1 * sign_v[index]).
If |S_u ∩ S_v| = 2, then u·v = sign1 + sign2, which can be 2, 0, or -2.
If u·v = 2, then ||u-v||^2 = 3 + 2 - 2*2 = 1 < 2 (max norm^2 from weight-2 is 2). BAD.
So we need u·v <= 1 for all weight-2 vectors v.

This means for any weight-3 vector u with support {a,b,c}, we need that for every pair
{a,b}, {a,c}, {b,c}, the weight-2 vectors with those supports don't have both signs +1.
But in our D11 set, we have ALL sign combinations, including (+1, +1) for every pair.
So any weight-3 vector with all +1 will conflict with the weight-2 vector that has +1, +1
on any of its pairs.

What if we use weight-3 vectors with mixed signs? Let's say u has signs (s1, s2, s3) on
support {a,b,c}. For a weight-2 vector v with support {a,b} and signs (t1, t2):
u·v = s1*t1 + s2*t2.
We need this <= 1 for all (t1, t2) in {(1,1), (1,-1), (-1,1), (-1,-1)}.
The maximum of s1*t1 + s2*t2 over all sign choices is |s1| + |s2| = 2 (when t1 = s1, t2 = s2).
So u·v can be 2, which violates the constraint.

Therefore, NO weight-3 vector (regardless of signs) can be added to the full D11 set,
because for any 3-subset {a,b,c}, there exists a weight-2 vector with support {a,b} and
signs matching u's signs at a and b, giving dot product 2.

So the D11 set of 220 is a local maximum under this greedy approach.

Can we do better by NOT using all 220 D11 vectors, and instead using a different construction?
The kissing number in 11 dimensions is known to be at least 460 (from the Leech lattice
projection or other constructions). But explicit constructions are complex.

For a direct approach within the time limit, the D11 construction of 220 is the best
simple explicit construction I can implement. Let me verify the constraint is satisfied.

For two D11 vectors u and v:
- If they share no indices: u·v = 0, ||u-v||^2 = 4 >= 2. OK.
- If they share 1 index i: u·v = s_u[i] * s_v[i] ∈ {-1, 1}. ||u-v||^2 = 4 - 2*u·v ∈ {2, 6}. OK.
- If they share 2 indices (same support {i,j}): u·v = s_u[i]*s_v[i] + s_u[j]*s_v[j].
  Possible values: 2, 0, -2. If u·v = 2, then u = v (same signs), which is a duplicate (not allowed).
  If u·v = 0, ||u-v||^2 = 4. OK.
  If u·v = -2, u = -v, ||u-v||^2 = 8. OK.

So all 220 vectors are valid. The answer is 220.
"""
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # safety margin

    D = 11
    vectors = []

    # D11 lattice: vectors with exactly two non-zero entries, each ±1
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)

    return np.array(vectors, dtype=np.int64)
