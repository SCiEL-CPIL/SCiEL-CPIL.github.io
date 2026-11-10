from fractions import Fraction
from itertools import combinations


def _is_subsum_free(selected, candidate):
    """Check that no reciprocal of a selected element equals the sum of
    reciprocals of a nonempty subset of the OTHER selected elements.
    'candidate' is the element to add; we verify the full set
    selected + [candidate].
    """
    S = selected + [candidate]
    n = len(S)
    recip = {x: Fraction(1, x) for x in S}

    for i in range(n):
        target = recip[S[i]]
        others = [S[j] for j in range(n) if j != i]
        # check all nonempty subsets of 'others'
        m = len(others)
        for mask in range(1, 1 << m):
            total = Fraction(0, 1)
            for k in range(m):
                if (mask >> k) & 1:
                    total += recip[others[k]]
                    if total > target:
                        break
            if total == target:
                return False
    return True


def construct():
    # Greedy: consider candidates in ascending order; add if the set stays
    # reciprocal-subsum-free. Smaller elements are 'targets' that are harder
    # to form from larger elements (which have smaller reciprocals) if the
    # larger elements are not present. By adding small elements first, we
    # protect them from being formed by sums of larger elements (which are
    # added later and might be rejected if they cause conflicts).
    selected = []
    for cand in range(1, 25):
        if _is_subsum_free(selected, cand):
            selected.append(cand)
    # Fallback: guarantee a valid nonempty set
    if not selected:
        return [1]
    return list(selected)
