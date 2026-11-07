def _quick_check(S):
    """Quick sufficient check: min sum * 2 > max sum guarantees validity."""
    if not S:
        return True
    min_sum = min(S)
    max_sum = sum(S)
    return 2 * min_sum > max_sum
