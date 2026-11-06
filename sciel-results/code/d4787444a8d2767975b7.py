
Wait, that's even worse because it creates 3^15 elements in a list. Let me think more carefully.

The real issue is that 3^15 ≈ 14.3M is too many to handle efficiently with Python sets. Let me try a completely different approach: use a best-first search or a smarter algorithm.

Actually, let me reconsider. The LCM of 1..30 is a huge number. Let me check: LCM(1..30) = 2329089562800. So W[i] = H/i are large integers. The sums can be very large, but we're looking for the smallest nonzero absolute value.

A better approach: instead of using the LCM, we can work with fractions directly using Python's `fractions.Fraction` or just track numerator/denominator. But that might be slower.

Actually, the key insight is: we want to minimize |Σ c_i / i| for c_i ∈ {-1,0,1}, c_i not all zero. This is equivalent to finding the smallest nonzero |N/D| where N = Σ c_i * (D/i) for some common denominator D.

Let me try a different strategy: use a priority queue to explore partial sums in order of their absolute value, and stop when we find a complete assignment with small absolute value.

