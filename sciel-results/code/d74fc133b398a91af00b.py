<<<<<<< SEARCH
"""Combined program: fast DP-based validity + geometric core + diverse greedy search."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s limit
    n = 512
    target_size = 48

    def _build_dp(arr):
        """dp[k] = bitset of all subset sums of size k from arr."""
        m = len(arr)
        if m == 0:
            return [1]
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in arr:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k - 1] << val)
        return dp

    def _is_valid_incremental(existing, x):
        """Check if adding x to existing keeps it valid. Returns True if valid."""
        if not existing:
            return True
        m = len(existing)
        total_sum = sum(existing)
        dp = _build_dp(existing)

        # Check 1: x is not the mean of any subset of existing of size >= 2
        for k in range(2, m + 1):
            target = k * x
            if target > total_sum:
                break
            if (dp[k] >> target) & 1:
                return False

        # Check 2: no element a in existing is the mean of a subset of existing∪{x}
        # of size >= 2 that includes x. Such subset = x + t elements from existing,
        # total size t+1 >= 2 so t >= 1. Mean = a => sum(t elements) = (t+1)*a - x.
        for a in existing:
            for t in range(1, m + 1):
                target = (t + 1) * a - x
                if target < 0:
                    continue
                if target > total_sum:
                    break
                if (dp[t] >> target) & 1:
                    return False
        return True

    def _greedy_from_order(order, selected_start=None):
        """Build a valid set by greedily adding elements in the given order."""
        if selected_start is None:
            selected = []
            selected_set = set()
        else:
            selected = list(selected_start)
            selected_set = set(selected)
        for x in order:
            if x in selected_set:
                continue
            if _is_valid_incremental(selected, x):
                selected.append(x)
                selected_set.add(x)
                if len(selected) >= target_size:
                    break
        return selected
=======
"""Combined program: fast DP-based validity + geometric core + diverse greedy search."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s limit
    n = 512
    target_size = 48

    def _build_dp(arr):
        """dp[k] = bitset of all subset sums of size k from arr."""
        m = len(arr)
        if m == 0:
            return [1]
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in arr:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k - 1] << val)
        return dp

    def _check_x_not_mean(existing, dp, total_sum, x):
        """Check x is not mean of any subset of existing of size >= 2."""
        m = len(existing)
        for k in range(2, m + 1):
            target = k * x
            if target > total_sum:
                break
            if (dp[k] >> target) & 1:
                return False
        return True

    def _check_existing_not_mean(existing, dp, total_sum, x):
        """Check no element in existing is mean of subset of existing∪{x} incl x."""
        m = len(existing)
        for a in existing:
            for t in range(1, m + 1):
                target = (t + 1) * a - x
                if target < 0:
                    continue
                if target > total_sum:
                    break
                if (dp[t] >> target) & 1:
                    return False
        return True

    def _is_valid_incremental(existing, x, dp=None, total_sum=None):
        """Check if adding x to existing keeps it valid. Returns True if valid."""
        if not existing:
            return True
        m = len(existing)
        if dp is None:
            dp = _build_dp(existing)
        if total_sum is None:
            total_sum = sum(existing)

        if not _check_x_not_mean(existing, dp, total_sum, x):
            return False
        if not _check_existing_not_mean(existing, dp, total_sum, x):
            return False
        return True

    def _greedy_from_order(order, selected_start=None):
        """Build a valid set by greedily adding elements in the given order."""
        if selected_start is None:
            selected = []
            selected_set = set()
        else:
            selected = list(selected_start)
            selected_set = set(selected)

        # Cache DP for the current selected set
        dp_cache = _build_dp(selected) if selected else [1]
        sum_cache = sum(selected)

        for x in order:
            if x in selected_set:
                continue
            if _is_valid_incremental(selected, x, dp_cache, sum_cache):
                selected.append(x)
                selected_set.add(x)
                dp_cache = _build_dp(selected)
                sum_cache += x
                if len(selected) >= target_size:
                    break
        return selected
>>>>>>> REPLACE

Now let me update the local search to also use cached DP, and add more strategies:

