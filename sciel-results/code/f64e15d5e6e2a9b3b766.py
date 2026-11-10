"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    MOD = 1009
    N = MOD - 1  # 1008

    # Precompute modular inverses for 1 to N
    # inv[x] is the unique y in 1..N such that x*y % MOD == 1
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, MOD - 2, MOD)

    # Identify self-inverse elements (x where inv[x] == x)
    # These are x such that x^2 ≡ 1 (mod 1009), i.e., x ≡ 1 or x ≡ -1 ≡ 1008
    self_inverse = set()
    for x in range(1, N + 1):
        if inv[x] == x:
            self_inverse.add(x)

    best_start = 1
    best_length = 0

    # Two-pointer / sliding window approach:
    # For a fixed start s, find the maximum end e such that [s, e] is valid.
    # An interval [s, e] is valid iff for every x in [s, e]:
    #   1. x is not self-inverse (i.e., inv[x] != x)
    #   2. inv[x] is not in [s, e] (i.e., inv[x] < s or inv[x] > e)
    #
    # Note: condition 2 with x=y is covered by condition 1 (if inv[x]==x, then inv[x] is in [s,e] when x is in [s,e]).
    # So we just need: for all x in [s, e], inv[x] < s or inv[x] > e.

    e = 0  # current right boundary

    for s in range(1, N + 1):
        if time.time() > deadline:
            break

        # If s is self-inverse, no valid interval can start at s
        if s in self_inverse:
            continue

        # Ensure e >= s
        if e < s:
            e = s

        # Expand e as far as possible while [s, e] remains valid
        # We need to check: for all x in [s, e], inv[x] < s or inv[x] > e
        # As we extend e, we only need to check the new element e+1.
        # But we also need to ensure that for all existing x in [s, e], inv[x] > e still holds
        # or inv[x] < s. Since e is increasing, the condition inv[x] > e becomes stricter.
        # So we need a different approach.

        # Alternative: for a fixed s, the maximum valid e is the minimum over all x in [s, e]
        # of the value such that inv[x] is not in [s, e].
        # Specifically, for each x, the constraint is: if inv[x] > x, then e < inv[x] (since inv[x] > e needed if inv[x] >= s).
        # Wait, let me reconsider.

        # For a fixed s, an interval [s, e] is valid iff:
        # For all x in [s, e]: inv[x] < s or inv[x] > e
        #
        # This means: for all x in [s, e], if inv[x] >= s, then inv[x] > e.
        # Equivalently: e < min{inv[x] : x in [s, e], inv[x] >= s}
        #
        # Let's define for each x: if inv[x] >= s, then x imposes a constraint e < inv[x].
        # So the maximum e for start s is:
        # e_max(s) = min{inv[x] - 1 : x in [s, e_max], inv[x] >= s}
        # But this is implicit.

        # Better approach: iterate e from s upward, and maintain the minimum "blocking" inverse.
        # For a fixed s, as we increase e, we track the minimum value of inv[x] for x in [s, e] where inv[x] >= s.
        # If this minimum is <= e, then the interval is invalid.

        # Let's use a cleaner two-pointer:
        # For fixed s, start with e = s - 1, and extend e while valid.
        # To check if adding next_e = e + 1 keeps validity:
        #   We need inv[next_e] < s or inv[next_e] > next_e.
        #   But also, for all existing x in [s, e], we need inv[x] < s or inv[x] > next_e.
        #   Since inv[x] was either < s or > e (from previous validity), and next_e > e,
        #   if inv[x] > e, it's possible that inv[x] <= next_e.
        #   So we need to track the minimum inv[x] among x in [s, e] with inv[x] >= s.

        # Let's precompute or maintain this.
        # For efficiency, let's just do the direct check but with early termination.

        # Actually, the original approach in the seed program is:
        #   if inv[next_e] < s or inv[next_e] > next_e:
        #       e = next_e
        #   else:
        #       break
        # This is INCORRECT because it doesn't check that existing elements' inverses are still outside [s, next_e].

        # Let me verify with a small example.
        # MOD = 7, N = 6
        # inv: 1->1, 2->4, 3->5, 4->2, 5->3, 6->6
        # Self-inverse: 1, 6
        # Start s=2:
        #   e starts at 2. Check next_e=3: inv[3]=5. 5 < 2? No. 5 > 3? Yes. So e=3.
        #   Check next_e=4: inv[4]=2. 2 < 2? No. 2 > 4? No. So break.
        #   Interval [2,3]. Check: x=2, inv[2]=4 not in [2,3]. x=3, inv[3]=5 not in [2,3]. Valid.
        #   But what about [2,4]? x=4, inv[4]=2 in [2,4]. Invalid. Correct.
        #
        # Another example: s=3, MOD=13, N=12
        # inv: 1->1, 2->7, 3->9, 4->10, 5->8, 6->11, 7->2, 8->5, 9->3, 10->4, 11->6, 12->12
        # Self-inverse: 1, 12
        # s=3:
        #   e=3. next_e=4: inv[4]=10. 10<3? No. 10>4? Yes. e=4.
        #   next_e=5: inv[5]=8. 8<3? No. 8>5? Yes. e=5.
        #   next_e=6: inv[6]=11. 11<3? No. 11>6? Yes. e=6.
        #   next_e=7: inv[7]=2. 2<3? Yes. e=7.
        #   next_e=8: inv[8]=5. 5<3? No. 5>8? No. Break.
        #   Interval [3,7]. Check: x=3, inv=9 not in [3,7]. x=4, inv=10 not in. x=5, inv=8 not in. x=6, inv=11 not in. x=7, inv=2 not in. Valid.
        #   But wait, is [3,8] valid? x=8, inv[8]=5 in [3,8]. Invalid. Correct.
        #
        # Now let's find a case where the simple check fails.
        # We need: inv[next_e] < s or inv[next_e] > next_e is true, but some existing x has inv[x] in [s, next_e].
        # For that, we need inv[x] > e (old) but inv[x] <= next_e (new). So inv[x] = next_e.
        # And inv[next_e] < s or inv[next_e] > next_e.
        # Example: MOD=17, N=16
        # inv: 1->1, 2->9, 3->6, 4->13, 5->7, 6->3, 7->5, 8->15, 9->2, 10->12, 11->14, 12->10, 13->4, 14->11, 15->8, 16->16
        # Self-inverse: 1, 16
        # s=2:
        #   e=2. next_e=3: inv[3]=6. 6<2? No. 6>3? Yes. e=3.
        #   next_e=4: inv[4]=13. 13<2? No. 13>4? Yes. e=4.
        #   next_e=5: inv[5]=7. 7<2? No. 7>5? Yes. e=5.
        #   next_e=6: inv[6]=3. 3<2? No. 3>6? No. Break.
        #   Interval [2,5]. Check: x=2,inv=9 not in. x=3,inv=6 not in. x=4,inv=13 not in. x=5,inv=7 not in. Valid.
        #
        # Let me try to construct a counterexample.
        # We need x in [s, e] with inv[x] = e+1, and inv[e+1] < s or inv[e+1] > e+1.
        # Let's say inv[x] = e+1, so x = inv[e+1].
        # So we need inv[inv[e+1]] = e+1, which is always true.
        # And inv[e+1] < s or inv[e+1] > e+1.
        # If inv[e+1] < s, then x = inv[e+1] < s, so x is not in [s, e]. Contradiction.
        # If inv[e+1] > e+1, then x = inv[e+1] > e+1, so x is not in [s, e]. Contradiction.
        # So the simple check IS correct!

        # Proof: Suppose [s, e] is valid. Consider adding next_e = e+1.
        # The only new pair to check is (x, next_e) for x in [s, e] and (next_e, next_e).
        # (next_e, next_e): inv[next_e] == next_e? If so, invalid. The check inv[next_e] > next_e handles this (false if equal).
        # (x, next_e): x * next_e ≡ 1 iff x = inv[next_e].
        # So we need inv[next_e] not in [s, next_e].
        # If inv[next_e] < s: not in [s, next_e]. OK.
        # If inv[next_e] > next_e: not in [s, next_e]. OK.
        # If s <= inv[next_e] <= next_e: then inv[next_e] is in [s, next_e], so the pair (inv[next_e], next_e) is in the interval, invalid.
        # So the condition is exactly: inv[next_e] < s or inv[next_e] > next_e.
        # And we don't need to recheck existing pairs because they were already valid and the interval only grew.
        # Wait, existing pairs are (x1, x2) with x1, x2 in [s, e]. They are still in [s, next_e], so they're still valid.
        # New pairs are (x, next_e) for x in [s, e] and (next_e, next_e).
        # (x, next_e) is invalid iff x = inv[next_e].
        # So we just need inv[next_e] not in [s, next_e].
        # This is equivalent to inv[next_e] < s or inv[next_e] > next_e.
        # So the original check is correct!

        # Great, so the seed program's logic is correct. Let me just run it efficiently.
        # The seed program uses a two-pointer where e is non-decreasing as s increases.
        # But is e non-decreasing? If s increases, the constraint inv[x] < s becomes looser, so e should be able to stay the same or increase.
        # Actually, when s increases by 1, some elements that were blocked by inv[x] < s might now be allowed.
        # But e is the maximum valid end. As s increases, the interval [s, e] shrinks from the left, so e might be able to increase.
        # But it's not guaranteed that e is non-decreasing. Let me think.
        # Actually, for s2 > s1, the interval [s2, e2] is a subset of [s1, e2] if e2 >= s2.
        # The constraints for [s2, e] are weaker than for [s1, e] because the condition inv[x] < s is easier to satisfy.
        # So if [s1, e1] is valid, then [s2, e1] is valid (for s2 > s1, e1 >= s2).
        # So e2 >= e1. Yes, e is non-decreasing.

        # So the two-pointer is valid. Let me just use it.

        # But wait, the seed program resets e to s if e < s. And it doesn't reset e when s increases.
        # So e is carried over. This should be correct.

        # Let me just run the seed program's logic.
        # But I want to make sure it's efficient. N=1008, so the total work is O(N) for the two-pointer.

        # Let me just implement it cleanly.
        e = 0
        for s in range(1, N + 1):
            if time.time() > deadline:
                break

            if s in self_inverse:
                continue

            if e < s:
                e = s

            while e < N:
                next_e = e + 1
                if inv[next_e] < s or inv[next_e] > next_e:
                    e = next_e
                else:
                    break

            length = e - s + 1
            if length > best_length:
                best_length = length
                best_start = s

    return [best_start, best_length]
# EVOLVE-BLOCK-END
