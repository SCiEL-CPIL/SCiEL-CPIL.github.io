"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction
    deadline = time.time() + 85.0
    denoms = list(range(2, 49))
    n = len(denoms)
    recip = [Fraction(1, d) for d in denoms]

    def greedy():
        selected, total = [], Fraction(0)
        for i, d in enumerate(denoms):
            if total + recip[i] < 1:
                selected.append(i)
                total += recip[i]
        return selected

    def exact_sum(indices):
        s = Fraction(0)
        for j in indices:
            s += recip[j]
        return s

    def local_search(initial_indices, time_limit):
        """Enhanced local search: add, remove, swap, and double-swap moves."""
        cur = set(initial_indices)
        cur_sum = exact_sum(list(cur))
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try all single additions
            for add in range(n):
                if add not in cur:
                    new_sum = cur_sum + recip[add]
                    if new_sum < 1 and new_sum > cur_sum:
                        cur.add(add)
                        cur_sum = new_sum
                        improved = True
                        break
            if improved:
                continue
            # Try all single removals (if removing improves deficit, i.e., sum gets closer to 1)
            # Actually removal decreases sum, which increases deficit, so we don't want removals
            # unless we're doing a swap
            # Try all swaps: remove one, add one
            for rem in list(cur):
                for add in range(n):
                    if add == rem or add in cur:
                        continue
                    new_sum = cur_sum - recip[rem] + recip[add]
                    if new_sum < 1 and new_sum > cur_sum:
                        cur.discard(rem)
                        cur.add(add)
                        cur_sum = new_sum
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # Try double swaps: remove two, add two
            cur_list = list(cur)
            for i1 in range(len(cur_list)):
                for i2 in range(i1 + 1, len(cur_list)):
                    rem1, rem2 = cur_list[i1], cur_list[i2]
                    base_sum = cur_sum - recip[rem1] - recip[rem2]
                    for add1 in range(n):
                        if add1 == rem1 or add1 == rem2 or add1 in cur:
                            continue
                        s1 = base_sum + recip[add1]
                        for add2 in range(n):
                            if add2 == rem1 or add2 == rem2 or add2 == add1 or add2 in cur:
                                continue
                            new_sum = s1 + recip[add2]
                            if new_sum < 1 and new_sum > cur_sum:
                                cur.discard(rem1)
                                cur.discard(rem2)
                                cur.add(add1)
                                cur.add(add2)
                                cur_sum = new_sum
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return list(cur), cur_sum

    try:
        import numpy as np
        vals = np.array([1.0 / d for d in denoms])
        nA = 23
        nB = n - nA
        A = vals[:nA]
        B = vals[nA:]

        sizeA = 1 << nA
        masksA = np.arange(sizeA, dtype=np.int32)
        sumsA = np.zeros(sizeA, dtype=np.float64)
        for i in range(nA):
            sumsA += A[i] * ((masksA >> i) & 1)
        order = np.argsort(sumsA)
        sumsA_sorted = sumsA[order]
        masksA_sorted = masksA[order]

        sizeB = 1 << nB
        masksB = np.arange(sizeB, dtype=np.int32)
        sumsB = np.zeros(sizeB, dtype=np.float64)
        for i in range(nB):
            sumsB += B[i] * ((masksB >> i) & 1)

        targets = 1.0 - sumsB
        idx = np.searchsorted(sumsA_sorted, targets, side='right') - 1
        valid = idx >= 0
        cand_idx = np.where(valid, idx, 0)
        cand_sum = np.where(valid, sumsB + sumsA_sorted[cand_idx], -np.inf)

        # Get more top candidates
        K = min(5000, sizeB)
        topB = np.argpartition(cand_sum, -K)[-K:]
        topB = topB[np.argsort(cand_sum[topB])[::-1]]

        best = None
        best_frac = None
        best_indices = None
        for bi in topB:
            if time.time() > deadline:
                break
            ai = int(cand_idx[bi])
            maskA = int(masksA_sorted[ai])
            maskB = int(masksB[bi])
            indices = [i for i in range(nA) if (maskA >> i) & 1]
            indices += [nA + i for i in range(nB) if (maskB >> i) & 1]
            if not indices:
                continue
            s = exact_sum(indices)
            if s < 1 and (best_frac is None or s > best_frac):
                best_frac = s
                best = indices
                best_indices = list(indices)

        if best_indices is not None:
            # Enhanced local search with more iterations
            search_deadline = min(deadline, time.time() + 15.0)
            best_indices, best_frac = local_search(best_indices, search_deadline)
            best = best_indices
            return best
    except Exception:
        pass
    return greedy()
# EVOLVE-BLOCK-END
