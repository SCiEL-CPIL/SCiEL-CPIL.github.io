"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _all_block_sums(seq):
    """Return set of all nonempty consecutive block sums."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _try_extend(values, sums, candidates, max_depth=0):
    """Try to extend values by picking from candidates. Return best extension."""
    if not candidates:
        return values, sums
    best = (values, sums)
    for x in candidates:
        # Compute new tails
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            new_sums = sums | set(tails)
            new_values = values + [x]
            # Recursively try to extend further
            remaining = [c for c in candidates if c > x]
            extended, _ = _try_extend(new_values, new_sums, remaining)
            if len(extended) > len(best[0]):
                best = (extended, new_sums)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Strategy 1: Greedy smallest (original)
    def greedy_smallest():
        values, sums = [], set()
        for x in range(1, 257):
            if time.time() > deadline:
                break
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values
    
    # Strategy 2: Greedy largest
    def greedy_largest():
        values, sums = [], set()
        for x in range(256, 0, -1):
            if time.time() > deadline:
                break
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return sorted(values)
    
    # Strategy 3: Greedy with lookahead - try each candidate, pick one that allows longest extension
    def greedy_lookahead():
        values, sums = [], set()
        remaining = list(range(1, 257))
        while remaining and time.time() < deadline:
            best_candidate = None
            best_length = 0
            best_ext = None
            best_ext_sums = None
            for x in remaining:
                tails, total = [x], x
                for a in reversed(values):
                    total += a
                    tails.append(total)
                if not sums.intersection(tails):
                    new_sums = sums | set(tails)
                    new_values = values + [x]
                    # Quick lookahead: try greedy extension from here
                    ext, _ = _try_extend(new_values, new_sums, [c for c in remaining if c > x], max_depth=3)
                    if len(ext) > best_length:
                        best_length = len(ext)
                        best_candidate = x
                        best_ext = ext
                        best_ext_sums = new_sums
            if best_candidate is None:
                break
            values = best_ext
            sums = best_ext_sums
            remaining = [c for c in remaining if c > values[-1]]
        return values
    
    # Run all strategies, keep the best
    best_result = []
    
    for strategy in [greedy_smallest, greedy_largest, greedy_lookahead]:
        if time.time() > deadline - 5:
            break
        try:
            result = strategy()
            if len(result) > len(best_result):
                best_result = result
        except Exception:
            pass
    
    return best_result if best_result else greedy_smallest()
# EVOLVE-BLOCK-END
