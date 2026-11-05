<<<<<<< SEARCH
    # Main loop
    attempts = 0
    while time.time() - start_time < time_limit:
        # Try randomized greedy
        rows = randomized_greedy()
        if rows is not None:
            score = max(max(r) for r in rows)
            if score < best_score:
                best_score = score
                best_rows = [list(r) for r in rows]
                # Try local search
                if time.time() - start_time < time_limit - 5:
                    improved_rows, improved_score = local_search(best_rows)
                    # Validate improved_rows before accepting
                    if validate_solution(improved_rows):
                        if improved_score < best_score:
                            best_score = improved_score
                            best_rows = improved_rows
                if best_score <= 105:
                    break
        
        attempts += 1
        if attempts % 100 == 0:
            # Occasionally try a more aggressive local search by shuffling row order
            if best_rows is not None and time.time() - start_time < time_limit - 10:
                shuffled = list(best_rows)
                random.shuffle(shuffled)
                improved_rows, improved_score = local_search(shuffled)
                if validate_solution(improved_rows):
                    if improved_score < best_score:
                        best_score = improved_score
                        best_rows = improved_rows
=======
    # Main loop
    attempts = 0
    while time.time() - start_time < time_limit:
        # Try randomized greedy
        rows = randomized_greedy()
        if rows is not None:
            score = max(max(r) for r in rows)
            if score < best_score:
                best_score = score
                best_rows = [list(r) for r in rows]
                # Try local search
                if time.time() - start_time < time_limit - 10:
                    improved_rows, improved_score = local_search(best_rows)
                    # Validate improved_rows before accepting
                    if validate_solution(improved_rows):
                        if improved_score < best_score:
                            best_score = improved_score
                            best_rows = improved_rows
                if best_score <= 105:
                    break
        
        attempts += 1
        # More frequent local search attempts
        if attempts % 50 == 0:
            if best_rows is not None and time.time() - start_time < time_limit - 10:
                shuffled = list(best_rows)
                random.shuffle(shuffled)
                improved_rows, improved_score = local_search(shuffled)
                if validate_solution(improved_rows):
                    if improved_score < best_score:
                        best_score = improved_score
                        best_rows = improved_rows
>>>>>>> REPLACE

