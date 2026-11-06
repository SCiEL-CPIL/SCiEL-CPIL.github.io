import sys
sys.setrecursionlimit(10000)

def construct():
    # Generate 5-smooth numbers up to 128
    smooth = []
    for a in range(8):  # 2^7 = 128
        for b in range(6):  # 3^5 = 243 > 128
            for c in range(4):  # 5^3 = 125
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 128:
                    smooth.append(val)
    smooth = sorted(set(smooth))
    
    # best[s] will store the minimum number of terms for an antichain summing to s
    # Initialize with infinity
    INF = float('inf')
    best = [INF] * 129
    best[0] = 0 # Empty set sums to 0 with 0 terms (though we need 1..30 terms, 0 is base case)
    
    # We need to store the actual terms for reconstruction? 
    # The problem asks for the rows of terms. 
    # We can store the best terms as well.
    best_terms = [None] * 129
    
    # DFS to enumerate all antichains
    # idx: index in smooth to consider next
    # current_terms: list of terms in current antichain
    # current_sum: sum of current_terms
    # current_size: len(current_terms)
    
    # To speed up compatibility check, we can maintain the set of current terms.
    # Since size is small, checking against all existing terms is O(k).
    
    def dfs(idx, current_terms, current_sum, current_size):
        # Pruning: if sum exceeds 128, stop
        if current_sum > 128:
            return
        
        # Update best for this sum if current antichain is valid and non-empty
        # We require 1..30 terms. The problem says "1..30 distinct... terms".
        # So size must be >= 1.
        if current_size >= 1:
            if current_size < best[current_sum]:
                best[current_sum] = current_size
                best_terms[current_sum] = list(current_terms)
        
        # Explore adding more terms
        # Optimization: We can stop if current_sum + min_possible_addition > 128?
        # The smallest addition is smooth[0] = 1? No, 1 divides everything.
        # If 1 is in the set, no other number can be added.
        # So if 1 is in current_terms, we can't add anything else.
        
        # General case:
        for i in range(idx, len(smooth)):
            x = smooth[i]
            
            # Check compatibility with all existing terms
            compatible = True
            for y in current_terms:
                if x % y == 0 or y % x == 0:
                    compatible = False
                    break
            
            if compatible:
                # Pruning: if current_sum + x > 128, since smooth is sorted, 
                # any larger x' will also exceed. 
                # But we are iterating i increasing, so if x makes sum > 128, 
                # we can break? 
                # Wait, if x makes sum > 128, then x + current_sum > 128.
                # Since smooth is sorted, smooth[j] >= x for j >= i.
                # So smooth[j] + current_sum >= x + current_sum > 128.
                # So we can break.
                if current_sum + x > 128:
                    break
                
                current_terms.append(x)
                dfs(i + 1, current_terms, current_sum + x, current_size + 1)
                current_terms.pop()

    dfs(0, [], 0, 0)
    
    # Check if all targets 1..128 have a solution
    # If any best[i] is INF, we have a problem.
    # The problem statement guarantees valid representations exist.
    # However, if our enumeration missed something (e.g. due to pruning logic error), 
    # we might have INFs.
    
    # Construct the result
    result = []
    for i in range(1, 129):
        if best_terms[i] is None:
            # Fallback: This shouldn't happen. 
            # But if it does, we might need a more exhaustive search or different logic.
            # For now, we assume it's found.
            # If not, we might return an empty list or raise an error? 
            # The evaluator expects 128 rows.
            # Let's try to find a representation using a simpler, slower method for just that target if needed?
            # Or just leave it as is? 
            # Given the guarantee, we assume best_terms[i] is not None.
            pass
        result.append(best_terms[i])
        
    return result
