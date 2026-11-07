import time
import random


# ---- incremental (fast) verification state ----
# dp[k] is a bitset: bit s is set iff some k-subset sums to s.
# The rule is violated exactly when two DIFFERENT cardinalities share a sum.

class Verifier:
    __slots__ = ("n", "dp", "acc", "valid", "max_k")

    def __init__(self, n):
        self.n = n
        self.dp = [0] * (n + 1)
        self.dp[0] = 1
        self.acc = 1          # union of all dp[k] so far
        self.valid = True     # no cross-cardinality collision yet
        self.max_k = 0        # highest cardinality with non-zero dp[k]

    def add(self, x):
        if not self.valid:
            return False
        n = self.n
        prev = self.dp
        ndp = list(prev)
        
        # We only need to update k from max_k + 1 down to 1
        # ndp[k] = prev[k] | (prev[k-1] << x)
        # For k > max_k + 1, prev[k] is 0 and prev[k-1] is 0 (if k-1 > max_k)
        # Actually, prev[k-1] is non-zero only if k-1 <= max_k.
        # So we only need to update k in [1, max_k + 1].
        
        new_max_k = self.max_k + 1
        if new_max_k > n:
            new_max_k = n
            
        # Update dp in reverse order to use previous state
        for k in range(new_max_k, 0, -1):
            ndp[k] = prev[k] | (prev[k - 1] << x)
            
        # Check for collisions
        # A collision occurs if ndp[k] shares any bits with acc (which contains sums from other cardinalities)
        # Note: acc currently contains sums from k=0..max_k.
        # The new sums in ndp[k] for k <= max_k might collide with existing sums in acc (from other k's).
        # The new sums in ndp[new_max_k] might collide with existing sums in acc.
        
        # We must check all k in [1, new_max_k] against acc.
        # However, acc is the union of all *previous* dp[j] for j != k.
        # Wait, the definition of collision is: sum S is in dp[k1] and dp[k2] for k1 != k2.
        # If we add x, new subsets are formed.
        # If a new subset of size k has sum S, and S was already in dp[j] for j != k, it's a collision.
        # S could be in dp[j] from before, or S could be in dp[j] formed by the same addition x?
        # No, dp[j] is updated simultaneously.
        # Let's stick to the simple check:
        # For each k in 1..new_max_k, check if ndp[k] & acc is non-zero.
        # But acc needs to be the union of *all other* cardinalities.
        # The previous acc was union of dp[0..max_k].
        # If we update dp[k] for k<=max_k, the new ndp[k] might collide with ndp[j] for j!=k.
        # The previous check `ndp[k] & acc` where acc is union of *previous* dp's is sufficient to detect collision with *old* subsets.
        # But we also need to detect collision between *new* subsets of different sizes.
        # Example: x=1. New 1-subset {1} sum 1. New 2-subset {1, ?} ... wait, we only add one element at a time.
        # If we add x, we create new subsets containing x.
        # A new k-subset containing x has sum S.
        # A new j-subset containing x has sum T.
        # If S == T, and k != j, that's a collision.
        # But S = sum(old k-1) + x, T = sum(old j-1) + x.
        # S == T implies sum(old k-1) == sum(old j-1).
        # This means the old (k-1)-subset and old (j-1)-subset had the same sum.
        # If k-1 != j-1, this was already a collision in the previous state!
        # Since the previous state was valid, no such collision existed.
        # Therefore, collisions can only occur between:
        # 1. New k-subset (with x) and Old j-subset (without x).
        # 2. New k-subset (with x) and New j-subset (with x) is impossible as shown above (would imply old collision).
        # 3. Old k-subset (without x) and Old j-subset (without x) is impossible (previous state valid).
        
        # So we only need to check if any new sum (in ndp[k] but not in prev[k]) collides with acc.
        # Or simpler: check if ndp[k] & acc is non-zero?
        # acc contains all old sums.
        # If ndp[k] has a bit set, it's either an old sum (in prev[k]) or a new sum.
        # If it's an old sum, it was in acc. If it collided, it would have been caught before?
        # No, if sum S was in prev[k], it was in acc. If it was also in prev[j] for j!=k, previous state invalid.
        # So prev[k] & acc == 0 for all k? No, acc is union of ALL prev[j].
        # So prev[k] is a subset of acc.
        # If we check ndp[k] & acc, we will definitely find bits from prev[k].
        # We need to check if (ndp[k] & ~prev[k]) & acc is non-zero?
        # i.e., do any *new* sums collide with *any* old sums?
        
        # Let's refine the check:
        # new_sums_k = ndp[k] & ~prev[k]
        # if new_sums_k & acc: return False
        
        # This is much faster because we only check the *new* bits.
        
        for k in range(1, new_max_k + 1):
            new_sums = ndp[k] & ~prev[k]
            if new_sums & self.acc:
                self.valid = False
                return False
                
        # Update acc
        # acc = union of all ndp[k]
        # We can just do acc |= ndp[k] for all k
        # But we can be smarter? No, just OR them.
        for k in range(1, new_max_k + 1):
            self.acc |= ndp[k]
            
        self.dp = ndp
        self.max_k = new_max_k
        return True


def _check(arr):
    """Independent full verification (used before returning)."""
    n = len(arr)
    if n < 1 or n > 60:
        return False
    seen = set()
    for v in arr:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        if v < 1 or v > 256:
            return False
        if v in seen:
            return False
        seen.add(v)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True


def _try_add(arr, used, ver, v):
    if v in used:
        return False
    arr.append(v)
    used.add(v)
    ok = ver.add(v)
    if not ok:
        arr.pop()
        used.discard(v)
    return ok


def _greedy(deadline, rng, order=None):
    """Deterministic/rule-based build: largest-valid-first, small gap."""
    cur = []
    used = set()
    ver = Verifier(60)
    if order is None:
        order = list(range(256, 0, -1))
        if rng is not None:
            rng.shuffle(order)

    added = True
    while added and len(cur) < 60 and time.monotonic() < deadline:
        added = False
        for v in order:
            if _try_add(cur, used, ver, v):
                added = True
                break

    # refine: try inserting mid-range values to grow further
    extra = list(range(1, 257))
    if rng is not None:
        rng.shuffle(extra)
    for v in extra:
        if time.monotonic() > deadline:
            break
        if _try_add(cur, used, ver, v):
            pass
    return cur


def _local_search(arr, deadline):
    """Try swapping elements in arr to see if we can add more elements."""
    if len(arr) >= 60:
        return arr

    cur = list(arr)
    used = set(cur)
    ver = Verifier(60)
    for v in cur:
        ver.add(v)

    # --- Pass A: 1-for-2 (remove one, add two) ---
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(cur)
        for i in range(n):
            if time.monotonic() > deadline:
                break
            removed = cur[i]
            cur.pop(i)
            used.discard(removed)

            ver2 = Verifier(60)
            for v in cur:
                ver2.add(v)

            added_count = 0
            for v in range(256, 0, -1):
                if v not in used and _try_add(cur, used, ver2, v):
                    added_count += 1
                    if added_count == 2:
                        break
                if time.monotonic() > deadline:
                    break

            if added_count >= 2:
                improved = True
            else:
                # Revert: pop added elements, restore removed
                for _ in range(added_count):
                    cur.pop()
                cur.insert(i, removed)
                used.add(removed)

    # --- Pass B: 1-for-1 swap (replace one element to unblock additions) ---
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(cur)
        for i in range(n):
            if time.monotonic() > deadline:
                break
            old = cur[i]
            for new in range(256, 0, -1):
                if new == old or new in used:
                    continue
                # Try swap
                cur[i] = new
                used.discard(old)
                used.add(new)
                ver2 = Verifier(60)
                ok = True
                for v in cur:
                    if not ver2.add(v):
                        ok = False
                        break
                if not ok:
                    cur[i] = old
                    used.discard(new)
                    used.add(old)
                    continue
                # After swap, try to add one more element
                added = False
                for v in range(256, 0, -1):
                    if v not in used and _try_add(cur, used, ver2, v):
                        added = True
                        break
                if added:
                    improved = True
                    break  # restart outer loop with new state
                else:
                    # revert swap
                    cur[i] = old
                    used.discard(new)
                    used.add(old)
        if improved:
            break  # restart while with updated cur

    # --- Pass C: remove-then-re-add polish ---
    # For each element, try removing it and re-adding a different one,
    # then attempt further additions.
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(cur)
        for i in range(n):
            if time.monotonic() > deadline:
                break
            old = cur[i]
            cur.pop(i)
            used.discard(old)
            ver2 = Verifier(60)
            for v in cur:
                ver2.add(v)
            # Try re-adding a different element at this position
            found = False
            for new in range(256, 0, -1):
                if new == old or new in used:
                    continue
                if _try_add(cur, used, ver2, new):
                    found = True
                    break
            if found:
                improved = True
                break
            else:
                # Restore
                cur.insert(i, old)
                used.add(old)

    return cur


def construct():
    t0 = time.monotonic()
    deadline = t0 + 82.0  # explicit safety margin under 90s wall-clock kill
    best = None
    best_len = 0

    def consider(arr):
        nonlocal best, best_len
        if _check(arr) and len(arr) > best_len:
            best = list(arr)
            best_len = len(arr)
            return True
        return False

    # 1) deterministic greedy: largest-valid-first (structure-derived answer)
    g0 = _greedy(deadline, None)
    consider(g0)
    
    # 2) local search on the greedy result
    if best is not None:
        ls_result = _local_search(best, deadline)
        consider(ls_result)

    # 3) structured restarts: random order, keep global best
    rng = random.Random(2024)
    for _ in range(600):
        if time.monotonic() > deadline:
            break
        g = _greedy(deadline, rng)
        if len(g) > best_len:
            consider(g)
            # Apply local search to the new best
            if best is not None:
                ls_result = _local_search(best, deadline)
                consider(ls_result)
        if best_len >= 60:
            break

    if best is None:
        best = list(range(226, 257))
        if not _check(best):  # shouldn't happen
            best = list(range(1, 31))
    best = [int(v) for v in best]
    return best
