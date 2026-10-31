            if max1 < min2 or max2 < min1:
                return 0.0 # Disjoint
            
            # Calculate overlap depth
            depth = min(max1, max2) - max(min1, min2)
            if depth < min_depth:
                min_depth = depth
