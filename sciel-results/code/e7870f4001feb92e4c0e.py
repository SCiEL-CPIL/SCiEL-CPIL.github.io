<<<<<<< SEARCH
import time
from math import lcm
from fractions import Fraction
=======
import time
from math import lcm
from fractions import Fraction
from itertools import combinations
>>>>>>> REPLACE
<<<<<<< SEARCH
    # ---- branch and bound with tighter bound ----
    def sum_compat_avail(sel, avail):
        """Sum of recip[i] for i in avail that are compatible with all in sel."""
        s = 0.0
        m = avail
        while m:
            lb = m & (-m)
            i = lb.bit_length() - 1
            # Check if i is compatible with all in sel
            ok = True
            for t in sel:
                if (conflict_mask[t] >> i) & 1:
                    ok = False
                    break
            if ok:
                s += recip[i]
            m ^= lb
        return s

    def sum_compat_avail_frac(sel, avail):
        """Exact fraction version for tight bounds."""
        s = Fraction(0)
        m = avail
        while m:
            lb = m & (-m)
            i = lb.bit_length() - 1
            ok = True
            for t in sel:
                if (conflict_mask[t] >> i) & 1:
                    ok = False
                    break
            if ok:
                s += recip_frac[i]
            m ^= lb
        return s

    nodes = [0]

    def bnb(sel, avail, cur, cur_frac):
        nonlocal best_val, best_val_frac, best_sel
        if time.time() > deadline:
            return
        # Tight bound: only count elements compatible with current selection
        upper_float = cur + sum_compat_avail(sel, avail)
        if upper_float <= best_val - 1e-9:
            return
        # If close, use exact fraction bound
        if upper_float <= best_val + 1e-6:
            upper_frac = cur_frac + sum_compat_avail_frac(sel, avail)
            if upper_frac <= best_val_frac:
                return
        # Find next candidate in order that is in avail and compatible with sel
        for i in order:
            if (avail >> i) & 1:
                # Check compatibility with current selection
                ok = True
                for t in sel:
                    if (conflict_mask[t] >> i) & 1:
                        ok = False
                        break
                if ok:
                    break
        else:
            if cur_frac > best_val_frac:
                best_val = cur
                best_val_frac = cur_frac
                best_sel = list(sel)
            return
        nodes[0] += 1
        # Take branch
        sel.append(i)
        new_avail = avail & ~(conflict_mask[i]) & ~(1 << i)
        bnb(sel, new_avail, cur + recip[i], cur_frac + recip_frac[i])
        sel.pop()
        # Skip branch
        bnb(sel, avail & ~(1 << i), cur, cur_frac)

    bnb([], full_mask, 0.0, Fraction(0))

    # ensure nonempty
    if not best_sel:
        best_sel = [order[0]]
    return [nums[i] for i in best_sel]
=======
    # Precompute compatible sets for each i (bitsets of j compatible with i)
    compat_mask = [0] * N
    for i in range(N):
        compat_mask[i] = full_mask & ~(1 << i) & ~conflict_mask[i]

    # ---- branch and bound with LP relaxation bound ----
    def lp_bound(sel, avail):
        """LP relaxation bound: greedy coloring of the conflict graph restricted to avail.
        For each color class (independent set), we can take at most all of them.
        Upper bound = cur + sum over color classes of max weight in that class."""
        # Build list of candidates
        cands = []
        m = avail
        while m:
            lb = m & (-m)
            i = lb.bit_length() - 1
            # Check if i is compatible with all in sel
            ok = True
            for t in sel:
                if (conflict_mask[t] >> i) & 1:
                    ok = False
                    break
            if ok:
                cands.append(i)
            m ^= lb
        if not cands:
            return 0.0
        # Greedy coloring: assign colors to cands
        # Color class c contains nodes with color c
        # Upper bound: for each color class, we can select at most all nodes in it
        # But actually, in an independent set, we can select at most one node per color class
        # Wait, no: a color class IS an independent set, so we can select ALL nodes in a color class
        # The bound is: sum over all nodes (trivial) or use a better approach
        # Actually for MWIS, the LP bound via coloring gives:
        # Partition into independent sets I_1, ..., I_k
        # Then opt <= sum_i max_{v in I_i} w(v)  ... NO
        # Actually: opt <= sum_i sum_{v in I_i} w(v) = total sum (trivial)
        # The correct bound: since each color class is independent, we can take all of one class
        # But we can mix across classes as long as no conflicts
        # Better approach: use the fact that in the conflict graph restricted to cands,
        # the independence number is at most the number of colors needed
        # For weighted case, a simple bound: sort by weight, use first-fit coloring
        # Then bound = sum of weights of nodes that could potentially be in an independent set
        # Actually the simplest useful bound: just sum all compatible weights (what we had)
        # Let me try a different approach: sequential bound
        # Process cands in order of decreasing weight
        # At each step, the remaining bound is the sum of weights of unprocessed nodes
        # that are compatible with the current partial solution
        s = 0.0
        m2 = avail
        # Filter to only compatible with sel
        for t in sel:
            m2 &= ~conflict_mask[t]
        while m2:
            lb = m2 & (-m2)
            i = lb.bit_length() - 1
            s += recip[i]
            m2 ^= lb
        return s

    def tight_bound(sel, avail):
        """Compute a tighter upper bound using greedy sequential processing."""
        # Start with all elements compatible with sel
        m = avail
        for t in sel:
            m &= ~conflict_mask[t]
        # Greedy: process in order of decreasing weight
        # At each step, we either take the node or skip it
        # The bound is: for the next node in order, the bound is w(i) + bound(rest compatible with i)
        # or bound(rest without i)
        # We take the max of these two
        # This is essentially a memoized B&B bound, but computed greedily
        # For efficiency, just use the sum of all compatible weights as base
        # and improve with a simple coloring argument
        # Simple improvement: if two nodes conflict, we can't take both
        # So bound <= sum - min(w1, w2) for each conflicting pair... too complex
        # Just return the sum for now
        s = 0.0
        while m:
            lb = m & (-m)
            i = lb.bit_length() - 1
            s += recip[i]
            m ^= lb
        return s

    nodes = [0]
    best_bound_pruned = [0]

    def bnb(sel, avail, cur, cur_frac):
        nonlocal best_val, best_val_frac, best_sel
        if time.time() > deadline:
            return
        # Compute bound
        m = avail
        for t in sel:
            m &= ~conflict_mask[t]
        # Quick sum bound
        bound_sum = 0.0
        m2 = m
        while m2:
            lb = m2 & (-m2)
            i = lb.bit_length() - 1
            bound_sum += recip[i]
            m2 ^= lb
        upper_float = cur + bound_sum
        if upper_float <= best_val - 1e-9:
            best_bound_pruned[0] += 1
            return
        # If close, use exact fraction
        if upper_float <= best_val + 1e-6:
            bound_frac = Fraction(0)
            m3 = m
            while m3:
                lb = m3 & (-m3)
                i = lb.bit_length() - 1
                bound_frac += recip_frac[i]
                m3 ^= lb
            upper_frac = cur_frac + bound_frac
            if upper_frac <= best_val_frac:
                best_bound_pruned[0] += 1
                return
        # Find next candidate
        next_i = -1
        for i in order:
            if (avail >> i) & 1:
                ok = True
                for t in sel:
                    if (conflict_mask[t] >> i) & 1:
                        ok = False
                        break
                if ok:
                    next_i = i
                    break
        if next_i == -1:
            if cur_frac > best_val_frac:
                best_val = cur
                best_val_frac = cur_frac
                best_sel = list(sel)
            return
        nodes[0] += 1
        # Take branch
        sel.append(next_i)
        new_avail = avail & ~(conflict_mask[next_i]) & ~(1 << next_i)
        bnb(sel, new_avail, cur + recip[next_i], cur_frac + recip_frac[next_i])
        sel.pop()
        # Skip branch
        bnb(sel, avail & ~(1 << next_i), cur, cur_frac)

    bnb([], full_mask, 0.0, Fraction(0))

    # Final local search polish
    best_sel, best_val = local_search(best_sel)

    # Ensure nonempty
    if not best_sel:
        best_sel = [order[0]]
    return [nums[i] for i in best_sel]
>>>>>>> REPLACE

Wait, I realize the bound improvement I'm making is essentially the same as before. Let me think more carefully about what would actually help. The key issue is that the B&B is exploring too many branches because the bound is weak. Let me try a different approach: use a better initial solution and a smarter branching strategy.

Actually, looking at the inspirations, one scored 1.1167 using fractions, math, random, time. Let me try combining the B&B with random restarts and a better local search.

