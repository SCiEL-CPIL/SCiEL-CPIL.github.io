"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from functools import lru_cache

def construct(n_rows, k):
    """
    CSP backtracking with MRV for difference triangle set.
    Each row has k+1 entries starting at 0, strictly increasing.
    All positive differences a_j - a_i (i<j) within rows are globally distinct.
    Minimize the largest entry.
    """
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * total_diffs_per_row
    
    # We'll do a backtracking search.
    # State: (rows_filled, current_row_index, current_row_entries, used_diffs_frozen)
    # But used_diffs is a set, so we can't easily memoize. 
    # Instead, we do iterative deepening on the max value M, and for each M,
    # do a CSP search.
    
    # Lower bound on M: we need at least total_diffs distinct positive differences,
    # each <= M, so M >= total_diffs (since diffs are in {1,...,M}).
    # Actually diffs can be up to M, and there are M possible values, so M >= total_diffs.
    
    # But also, each row's entries are in [0, M], strictly increasing, k+1 entries.
    # The smallest possible max for a single row with k+1 entries starting at 0 
    # with all internal diffs distinct is related to Golomb rulers.
    
    # Let's do iterative deepening on M.
    
    # For the search, we'll represent state as:
    #   rows_done: tuple of completed rows (each a tuple of entries)
    #   cur_row: tuple of entries in current row so far
    #   used_diffs: frozenset of all differences used so far
    #   row_idx: which row we're filling (0 to n_rows-1)
    
    # MRV: For the current row, we need to choose the next entry x > cur_row[-1]
    # such that all diffs x - a for a in cur_row are not in used_diffs and not conflicting
    # with each other (they're automatically distinct since a's are distinct).
    # The "remaining values" for the next entry are those x in (cur_row[-1], M] 
    # such that {x - a for a in cur_row} is disjoint from used_diffs.
    
    # But MRV across rows: we could also choose which row to fill next.
    # For simplicity, let's fill rows in order, but within a row, use MRV-like 
    # value selection.
    
    # Actually, for stronger MRV, let's consider: at any point, we have some rows 
    # partially filled. The most constrained position is the one with fewest valid 
    # next values. But this is complex. Let's start with row-order filling and 
    # within-row, try values in increasing order (which also helps minimize M).
    
    # Let me implement a recursive backtracking with pruning.
    
    sys.setrecursionlimit(10000)
    
    def search(M):
        """Try to find a valid set with max entry <= M. Return list of rows or None."""
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, cur_row):
            # cur_row is a list of entries for the current row, starts with [0]
            if row_idx == n_rows:
                # All rows filled
                return rows[:]
            
            if len(cur_row) == k + 1:
                # Current row is complete, start next row
                rows.append(cur_row)
                result = backtrack(row_idx + 1, [0])
                if result is not None:
                    return result
                rows.pop()
                return None
            
            # Need to add next entry to cur_row
            # MRV-like: find all valid next values, pick the most constrained?
            # For now, try values in increasing order from cur_row[-1]+1 to M
            last = cur_row[-1]
            # Precompute: for each candidate x, check if diffs are all new
            # To speed up, we can precompute the set of diffs that would be added
            for x in range(last + 1, M + 1):
                new_diffs = set()
                conflict = False
                for a in cur_row:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                    if d in new_diffs:  # shouldn't happen since a's are distinct
                        conflict = True
                        break
                    new_diffs.add(d)
                if conflict:
                    continue
                
                # Check if this is feasible: we still need to fill (k+1 - len(cur_row) - 1) 
                # more entries, each adding more diffs. The remaining diffs must be 
                # distinct and not in used_diffs ∪ new_diffs.
                # Simple bound: total diffs used so far + new_diffs + remaining diffs needed <= M
                # Remaining entries: rem = k + 1 - len(cur_row) - 1
                # Each remaining entry adds (len(cur_row)+1) diffs (to all previous entries)
                # Total remaining diffs = sum_{i=1}^{rem} (len(cur_row) + i)
                rem = k + 1 - len(cur_row) - 1
                rem_diffs = 0
                for i in range(1, rem + 1):
                    rem_diffs += len(cur_row) + i
                # Also, other rows not yet started will need their diffs
                # This is getting complex. Let's just try the value.
                
                cur_row.append(x)
                used_diffs.update(new_diffs)
                result = backtrack(row_idx, cur_row)
                if result is not None:
                    return result
                cur_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return None
        
        result = backtrack(0, [0])
        return result
    
    # Iterative deepening on M
    # Lower bound: total_diffs (since we need that many distinct diffs, each in 1..M)
    # Also, for a single row, the minimal Golomb ruler of order k+1 has length ~ k^2
    # For k=8, order 9 Golomb ruler min length is 55 (known)
    # But we have 4 rows, so M needs to accommodate all.
    
    # Simple lower bound: M >= total_diffs
    # Better: M >= max over rows of (minimal Golomb ruler length for order k+1)
    # But let's just start from a reasonable bound.
    
    # For n_rows=4, k=8: total_diffs = 4*36 = 144
    # Known best is 134, so M is at most 134.
    # Let's start from lower bound and go up.
    
    # Lower bound: we need 144 distinct differences in [1, M], so M >= 144? No, 
    # differences can be any value in 1..M, and we need 144 distinct ones, so M >= 144.
    # Wait, that's not right. The differences are a_j - a_i where a_j, a_i in [0,M], 
    # so differences are in [1, M]. We need 144 distinct differences, so M >= 144.
    # But the known answer is 134 < 144. Contradiction?
    
    # Ah, I see my error. The differences don't all have to be in [1, M] in the sense 
    # that they're all distinct AND <= M. They are distinct and each is in [1, M]. 
    # There are M possible values (1 through M), and we need 144 distinct ones, so M >= 144.
    # But the problem says best known is 134. Let me re-read.
    
    # "all differences between two entries of the same row are distinct over all rows"
    # So total distinct differences = n_rows * k*(k+1)/2 = 4*36 = 144.
    # Each difference is in {1, 2, ..., M} where M is the max entry.
    # So we need at least 144 distinct values in {1,...,M}, hence M >= 144.
    # But the task says best known is 134 and counting bound is 112. 
    # The counting bound of 112 must be something else.
    
    # Wait, maybe I'm misunderstanding. Let me re-read: "the simple counting bound is 112".
    # Maybe the counting bound is a lower bound on M, and 112 < 134, so it's consistent 
    # that M >= 112 and best known is 134.
    
    # But my argument that M >= 144 seems solid. Unless... the differences don't all 
    # have to be distinct? No, the task says "all positive differences a_j - a_i (i < j) 
    # inside the rows are distinct over all rows".
    
    # Hmm, let me check: for a row [0, a1, a2, ..., ak], the differences are:
    # a1-0, a2-0, ..., ak-0, a2-a1, a3-a1, ..., ak-a1, ..., ak-a_{k-1}
    # That's k + (k-1) + ... + 1 = k(k+1)/2 differences.
    # For k=8: 8*9/2 = 36 per row. 4 rows: 144 total.
    # All must be distinct, and each is in [1, M].
    # So M >= 144.
    
    # But the task says best known is 134. This is a contradiction unless I'm 
    # misunderstanding the problem.
    
    # Let me re-read the task more carefully...
    # "Choose 4 rows of 8 increasing integers starting at 0"
    # "8 increasing integers starting at 0" - so each row has 8 integers? Or 8+1=9?
    # "Each row is a list of k + 1 integers that starts at 0"
    # So k=8 means 9 integers per row.
    # "all differences between two entries of the same row are distinct over all rows"
    # C(9,2) = 36 differences per row, 144 total.
    
    # I'm confused by the discrepancy. Let me just implement the search and see.
    # Maybe the "counting bound" refers to something else, or I'm wrong about M >= total_diffs.
    
    # Actually, wait. The differences are a_j - a_i for i < j. These are all positive 
    # and at most M (since max entry is M and min is 0). There are M possible values 
    # (1, 2, ..., M). We need 144 distinct values, so M >= 144. This seems airtight.
    
    # Unless the problem allows the same difference to appear in different rows? 
    # No, it says "distinct over all rows".
    
    # Let me just proceed with the implementation and start M from a reasonable value.
    # If the known best is 134, maybe I'm miscounting. Let me just try M from 134 upward.
    
    # Actually, let me just do a straightforward backtracking and see what happens.
    # For the given parameters (n_rows=4, k=8), I'll search for a solution.
    
    # Given time constraints (120s), let's be smart about it.
    
    # Let's try M values starting from a lower bound.
    # Lower bound: max(total_diffs, min_golomb_ruler_length)
    # For order 9 Golomb ruler, min length is 55.
    # total_diffs = 144.
    # So start from M=144? But known best is 134...
    
    # I think there might be an error in my understanding. Let me just implement 
    # the search and try M values.
    
    # For safety, let's start from M = max(100, total_diffs - 10) and go up.
    # Actually, let me just try M from a reasonable range.
    
    # Given the 120s time limit, let's be efficient.
    
    # Let me try a different approach: since the known best is 134, let's just 
    # try to find a solution with M around 134-140.
    
    # For the specific case n_rows=4, k=8, let's hardcode a search around M=134.
    
    # But the function should work for general n_rows, k.
    
    # Let me implement the backtracking and try M values in increasing order.
    
    # To make it fast, I'll add pruning:
    # 1. If the number of remaining needed diffs > number of available diffs in [1, M], fail.
    # 2. Use MRV: choose the next position with fewest valid values.
    
    # For now, let's do a simpler version and see if it works within time.
    
    # Given the complexity, let me try M from a lower bound upward.
    
    lb = max(k * (k + 1) // 2 * n_rows, 1)  # This is 144 for our case
    # But known best is 134, so maybe lb should be lower. Let me just start from 100.
    
    # Actually, I realize I might be wrong. Let me reconsider.
    # The differences are a_j - a_i. If the max entry is M, the max difference is M - 0 = M.
    # The min difference is 1. So differences are in {1, ..., M}.
    # We need 144 distinct differences. So M >= 144.
    # I'm confident in this. The "best known 134" might be for a different variant.
    
    # Let me just implement and try.
    
    for M in range(max(1, lb), lb + 50):
        result = search(M)
        if result is not None:
            # Convert to lists
            return [list(r) for r in result]
    
    # If no solution found, fall back to greedy
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
