        is_valid, new_score = evaluate(new_points)
        if not is_valid:
            T *= cooling
            continue
        
        delta = new_score - current_score
        # ...
