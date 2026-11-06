import time

def construct():
    start_time = time.time()
    deadline = start_time + 45.0

    MOD = 1009
    N = 1008

    # Precompute modular inverses via Fermat's little theorem
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # For each position i, compute the minimum valid start if i is the right endpoint
    # of a valid interval. If i is self-inverse, no interval ending at i is valid.
    # Otherwise, if inv[i] = y, then any valid interval [s, i] must have s > y (so y < s)
    # or y > i (impossible since y = inv[i] and we check). Actually:
    #   We need: for all x in [s, i], inv[x] not in [s, i].
    #   For x = i: need inv[i] not in [s, i]. Since inv[i] = y:
    #     if y < s: OK
    #     if s <= y <= i: NOT OK
    #   So for x = i alone, we need s > y (if y <= i) or s >= 1 (if y > i, which can't happen for y=inv[i] in [1,N]).
    #   Actually inv[i] is always in [1, 1008] for i in [1, 1008] since 1009 is prime.
    #   So y = inv[i] is in [1, 1008]. If y == i, self-inverse, impossible.
    #   If y != i, we need s > y OR y > i. But y could be > i or y < i.
    #   If y > i: then y is not in [s, i] for any s <= i, so no constraint from x=i.
    #   If y < i: then we need s > y, i.e., s >= y + 1.
    #   If y == i: self-inverse, impossible.
    #
    # But we also need to check ALL x in [s, i], not just x = i.
    # The backward pass: for right endpoint r, the minimum valid start is
    #   max over all x in [s, r] of (constraint from x)
    # where constraint from x is:
    #   if inv[x] == x: impossible (mark as N+1)
    #   elif inv[x] < x: need s > inv[x], i.e., s >= inv[x] + 1
    #   else (inv[x] > x): no constraint (inv[x] > x means inv[x] > any s <= x, so inv[x] not in [s, r] as long as inv[x] > r... wait no)
    #
    # Wait, I need to be more careful. For x in [s, r], we need inv[x] not in [s, r].
    # inv[x] not in [s, r] means inv[x] < s OR inv[x] > r.
    #
    # For a fixed r and x in [s, r]:
    #   If inv[x] > r: always OK (regardless of s)
    #   If inv[x] < s: OK
    #   If s <= inv[x] <= r: NOT OK
    #
    # So the constraint is: for all x in [s, r], either inv[x] < s or inv[x] > r.
    #
    # Backward pass approach:
    # For each r from N down to 1, maintain the maximum "lower bound" on s.
    # When we add position r to the window (going backward), we need to check:
    #   For x = r: if inv[r] <= r, then we need s > inv[r], so s >= inv[r]+1.
    #              if inv[r] > r, no constraint.
    #              if inv[r] == r, impossible.
    #   For existing x in [s, r-1]: their constraints were already accounted for.
    #     But wait, the constraint for x depends on r! If inv[x] was previously > r-1 but now r increased...
    #     No, r is the right endpoint. If we're doing backward pass from N to 1, r decreases.
    #
    # Let me reconsider. Forward pass: for each right endpoint r (increasing), find min valid s.
    #
    # Actually, let's think about it differently. Define for each position i:
    #   If i is self-inverse (inv[i] == i): conflict_bound[i] = i + 1 (any interval containing i is invalid)
    #   Else if inv[i] < i: conflict_bound[i] = inv[i] + 1 (if i is in [s, r], need s > inv[i])
    #   Else (inv[i] > i): conflict_bound[i] = 1 (no constraint on s from i alone, since inv[i] > i >= s)
    #     Wait, but inv[i] > i doesn't mean inv[i] > r. If r >= inv[i], then inv[i] is in [s, r] if s <= inv[i].
    #     So for x = i with inv[i] > i: if r >= inv[i], we need s > inv[i], i.e., s >= inv[i]+1.
    #     If r < inv[i], no constraint.
    #
    # This is getting complicated because the constraint depends on both s and r.
    #
    # Better approach: For a fixed interval [s, r], it's valid iff for all x in [s, r]:
    #   inv[x] < s OR inv[x] > r
    #
    # Equivalently: for all x in [s, r] with inv[x] in [s, r], we have a violation.
    # So [s, r] is valid iff there is no x in [s, r] such that inv[x] is also in [s, r].
    #
    # Key insight: if x is in [s, r] and inv[x] is in [s, r], then both x and inv[x] are in [s, r].
    # Since inv[inv[x]] = x, this is symmetric.
    #
    # For a fixed r, what is the maximum valid s? (We want to minimize s to maximize length.)
    # [s, r] is valid iff no x in [s, r] has inv[x] in [s, r].
    #
    # Let's define: for each position i, let lo[i] = min(i, inv[i]) and hi[i] = max(i, inv[i]).
    # A pair (i, inv[i]) with i != inv[i] creates a conflict if both i and inv[i] are in [s, r],
    # i.e., s <= lo[i] and hi[i] <= r.
    # Self-inverse: i = inv[i], conflict if s <= i <= r.
    #
    # So [s, r] is valid iff:
    #   For all i in [s, r] with i != inv[i]: NOT (s <= lo[i] AND hi[i] <= r)
    #   For all i in [s, r] with i == inv[i]: NOT (s <= i AND i <= r) [but i is in [s,r] by assumption, so always violated]
    #
    # So self-inverse elements make any interval containing them invalid.
    #
    # For non-self-inverse i: conflict if s <= lo[i] and hi[i] <= r.
    #   Since i is in [s, r], we have s <= i <= r.
    #   If inv[i] < i: lo[i] = inv[i], hi[i] = i. Conflict if s <= inv[i] (and i <= r is true since i in [s,r]).
    #     So need s > inv[i].
    #   If inv[i] > i: lo[i] = i, hi[i] = inv[i]. Conflict if hi[i] <= r, i.e., inv[i] <= r.
    #     If inv[i] <= r: need NOT (s <= i), but s <= i is always true since i in [s,r]. So conflict!
    #     If inv[i] > r: no conflict from i.
    #
    # So for fixed r, the constraint from each i in [s, r] is:
    #   If i is self-inverse: interval invalid.
    #   If inv[i] < i: need s > inv[i].
    #   If inv[i] > i and inv[i] <= r: interval invalid (since s <= i <= r and inv[i] <= r, so both in [s,r]).
    #   If inv[i] > i and inv[i] > r: no constraint.
    #
    # For fixed r, let's find the minimum valid s:
    #   s must be > max{inv[i] : i in [s, r], inv[i] < i}
    #   AND no i in [s, r] is self-inverse
    #   AND no i in [s, r] has inv[i] > i and inv[i] <= r
    #
    # The third condition: if there exists i in [s, r] with inv[i] > i and inv[i] <= r, then [s, r] is invalid.
    #   This means: if there's a pair (i, inv[i]) with i < inv[i] <= r and i >= s, then invalid.
    #   Equivalently: if there's a pair with both elements in [s, r], invalid.
    #
    # This is still circular (s appears on both sides). Let me use a different strategy.
    #
    # Strategy: For each right endpoint r, binary search or scan for the minimum valid s.
    # But that's O(n^2) in the worst case.
    #
    # Better: Use the fact that if [s, r] is valid, then [s', r] is valid for s' > s.
    # And if [s, r] is invalid, [s, r'] is invalid for r' >= r (adding more elements can only add conflicts).
    # Wait, that's not right. [s, r] invalid doesn't mean [s, r'] invalid for r' > r.
    # Actually it does: if [s, r] has a conflict pair (x, inv[x]) both in [s, r], then [s, r'] also contains them.
    # So validity is monotone in r (decreasing) and monotone in s (increasing).
    #
    # Two-pointer: maintain left pointer s. For each r (increasing), advance s until [s, r] is valid.
    # When we add r to the window, check if r creates a conflict:
    #   - If inv[r] is in [s, r-1]: conflict, advance s past inv[r].
    #   - If r is self-inverse: conflict, advance s past r.
    #   - If inv[r] > r: no immediate conflict (inv[r] not in window yet).
    #   - But also: when s advances, some previous elements' constraints might change.
    #     Specifically, an element i in the window with inv[i] > i: if inv[i] was > r (no conflict),
    #     but now... no, r only increases, so inv[i] > r stays true or becomes false (if r catches up to inv[i]).
    #     Wait, if inv[i] > i and r increases past inv[i], then inv[i] <= r, and since i is in [s, r]
    #     (assuming s <= i), we get a conflict. But i might have been pushed out by s advancing.
    #
    # The two-pointer should work if we handle it correctly:
    #   For each r:
    #     While [s, r] is invalid:
    #       Find the conflict. The conflict is either:
    #         - r is self-inverse: s = r + 1
    #         - inv[r] in [s, r-1]: s = max(s, inv[r] + 1)
    #         - Some i in [s, r-1] has inv[i] in [s, r]: 
    #           This can happen if inv[i] = r (i.e., inv[r] = i, so inv[i] = r). 
    #           But that's the same as inv[r] in [s, r-1] case.
    #           Or inv[i] < i and inv[i] in [s, r-1]: but this was already handled when i was added.
    #           Or inv[i] > i and inv[i] = r: same as above.
    #           Or inv[i] > i and inv[i] in [s, r-1]: this means both i and inv[i] are in [s, r-1],
    #             which would have been caught when inv[i] was added (since inv[i] > i, inv[i] was added after i).
    #
    # So the only new conflict when adding r is:
    #   - r self-inverse
    #   - inv[r] in [s, r-1] (which covers both inv[r] < r and inv[r] > r cases... wait, inv[r] > r means inv[r] not in [s, r-1])
    #
    # Actually if inv[r] > r, then inv[r] is not in [s, r-1], so no conflict from r's perspective.
    # But r might be the inv of some existing element i in [s, r-1]. That is, inv[i] = r.
    # Since inv[inv[i]] = i, this means inv[r] = i. So inv[r] = i < r, which is the case inv[r] in [s, r-1].
    # So it's covered!
    #
    # Therefore, the two-pointer is:
    #   s = 1
    #   for r in 1..N:
    #     if r is self-inverse: s = r + 1; continue
    #     if inv[r] < r and s <= inv[r] <= r-1: s = inv[r] + 1
    #     # Now [s, r] should be valid
    #     update best
    #
    # But wait, when s advances, we might create new "valid" status for previous elements.
    # No, advancing s only removes elements from the left, which can only remove conflicts.
    # So if [s, r-1] was valid, and we advance s to s', [s', r-1] is still valid.
    # Then we add r: if no new conflict, [s', r] is valid.
    #
    # The only issue: when s advances past some element, that element is no longer in the window,
    # so its constraint is gone. This is fine.
    #
    # Let me also handle the case where inv[r] < s: no conflict.
    # And inv[r] > r: no conflict (as argued above).
    #
    # So the simple two-pointer should work! Let me also verify with the backward pass idea.
    #
    # Backward pass (as suggested in the directive):
    #   For each position i, compute the "conflict bound" cb[i]:
    #     If i is self-inverse: cb[i] = i + 1 (any interval containing i is invalid, so min start > i)
    #     Else if inv[i] < i: cb[i] = inv[i] + 1 (if i is in interval, start must be > inv[i])
    #     Else (inv[i] > i): cb[i] = 1 (no constraint from i when i is the rightmost... but this isn't quite right)
    #
    # The issue with the backward pass is that the constraint from i depends on whether inv[i] is also in the interval.
    # For inv[i] > i: the constraint only activates if inv[i] <= r (right endpoint).
    #
    # I think the two-pointer is cleaner and correct. Let me implement it carefully.

    best_start = 1
    best_len = 0

    s = 1  # left pointer (start of current window)

    for r in range(1, N + 1):
        if time.time() > deadline:
            break

        if inv[r] == r:
            # Self-inverse: cannot be in any valid interval
            s = r + 1
            continue

        if inv[r] < r and s <= inv[r]:
            # inv[r] is in the current window [s, r-1], advance s past it
            s = inv[r] + 1

        # Now [s, r] should be valid
        current_len = r - s + 1
        if current_len > 0 and current_len > best_len:
            best_len = current_len
            best_start = s

    # Ensure we return a valid non-empty interval
    if best_len <= 0:
        for i in range(1, N + 1):
            if inv[i] != i:
                best_start = i
                best_len = 1
                break

    return [best_start, best_len]
