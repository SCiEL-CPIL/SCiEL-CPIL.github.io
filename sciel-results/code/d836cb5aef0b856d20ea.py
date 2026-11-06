<<<<<<< SEARCH
    # --- Phase 2: Local Search from Best ---
    if time.time() < deadline - 10:
        current = list(best_arr)
        current_score = best_score
        # Run local search for a fixed amount of time
        local_deadline = min(deadline - 5, start_time + 30)
        best_score, best_arr = _local_search(current, current_score, local_deadline, best_score, best_arr)
    
    # --- Phase 3: Random Restarts ---
    if time.time() < deadline - 5:
        restarts_done = 0
        while time.time() < deadline - 3 and restarts_done < 500:
            restarts_done += 1
            
            # Generate diverse random candidates
            r = random.random()
            if r < 0.15:
                # Small numbers
                arr = random.sample(range(1, 100), 12)
            elif r < 0.30:
                # Small even numbers
                arr = random.sample(range(2, 300, 2), 12)
            elif r < 0.45:
                # Multiples of 3
                arr = random.sample(range(3, 300, 3), 12)
            elif r < 0.60:
                # Multiples of 6 (LCM of 2,3)
                arr = random.sample(range(6, 500, 6), 12)
            elif r < 0.75:
                # Modular construction: half ≡ 1 mod 6, half ≡ 5 mod 6
                candidates1 = [x for x in range(1, 500) if x % 6 == 1]
                candidates5 = [x for x in range(1, 500) if x % 6 == 5]
                if len(candidates1) >= 6 and len(candidates5) >= 6:
                    arr = random.sample(candidates1, 6) + random.sample(candidates5, 6)
                else:
                    arr = random.sample(range(1, 300), 12)
            elif r < 0.85:
                # Modular construction for prime 5: half ≡ 1 mod 5, half ≡ 4 mod 5
                candidates1 = [x for x in range(1, 500) if x % 5 == 1]
                candidates4 = [x for x in range(1, 500) if x % 5 == 4]
                if len(candidates1) >= 6 and len(candidates4) >= 6:
                    arr = random.sample(candidates1, 6) + random.sample(candidates4, 6)
                else:
                    arr = random.sample(range(1, 300), 12)
            else:
                # Random multiples of a random small base
                base = random.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
                max_m = 10000 // base
                if max_m < 12:
                    base = 2
                    max_m = 5000
                mults = random.sample(range(1, max_m + 1), 12)
                arr = [base * m for m in mults]
            
            if len(set(arr)) != 12 or any(x < 1 or x > 10000 for x in arr):
                continue
            
            score = _objective(arr)
            if score < best_score:
                best_score = score
                best_arr = list(arr)
            
            # Quick local search from this candidate
            if time.time() < deadline - 3:
                cur = list(arr)
                cur_score = score
                # Short local search
                search_deadline = min(deadline - 3, time.time() + 2.0)
                best_score, best_arr = _local_search(cur, cur_score, search_deadline, best_score, best_arr)

    return best_arr
=======
    # --- Phase 2: Local Search from Best ---
    if time.time() < deadline - 5:
        current = list(best_arr)
        current_score = best_score
        # Run local search for a substantial amount of time
        local_deadline = min(deadline - 3, start_time + 40)
        best_score, best_arr = _local_search(current, current_score, local_deadline, best_score, best_arr)
    
    # --- Phase 3: Random Restarts with Local Search ---
    if time.time() < deadline - 3:
        restarts_done = 0
        while time.time() < deadline - 2 and restarts_done < 300:
            restarts_done += 1
            
            # Generate diverse random candidates
            r = random.random()
            if r < 0.15:
                # Small numbers
                arr = random.sample(range(1, 100), 12)
            elif r < 0.30:
                # Small even numbers
                arr = random.sample(range(2, 300, 2), 12)
            elif r < 0.45:
                # Multiples of 3
                arr = random.sample(range(3, 300, 3), 12)
            elif r < 0.60:
                # Multiples of 6 (LCM of 2,3)
                arr = random.sample(range(6, 500, 6), 12)
            elif r < 0.75:
                # Modular construction: half ≡ 1 mod 6, half ≡ 5 mod 6
                candidates1 = [x for x in range(1, 500) if x % 6 == 1]
                candidates5 = [x for x in range(1, 500) if x % 6 == 5]
                if len(candidates1) >= 6 and len(candidates5) >= 6:
                    arr = random.sample(candidates1, 6) + random.sample(candidates5, 6)
                else:
                    arr = random.sample(range(1, 300), 12)
            elif r < 0.85:
                # Modular construction for prime 5: half ≡ 1 mod 5, half ≡ 4 mod 5
                candidates1 = [x for x in range(1, 500) if x % 5 == 1]
                candidates4 = [x for x in range(1, 500) if x % 5 == 4]
                if len(candidates1) >= 6 and len(candidates4) >= 6:
                    arr = random.sample(candidates1, 6) + random.sample(candidates4, 6)
                else:
                    arr = random.sample(range(1, 300), 12)
            else:
                # Random multiples of a random small base
                base = random.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
                max_m = 10000 // base
                if max_m < 12:
                    base = 2
                    max_m = 5000
                mults = random.sample(range(1, max_m + 1), 12)
                arr = [base * m for m in mults]
            
            if len(set(arr)) != 12 or any(x < 1 or x > 10000 for x in arr):
                continue
            
            score = _objective(arr)
            if score < best_score:
                best_score = score
                best_arr = list(arr)
            
            # Quick local search from this candidate
            if time.time() < deadline - 2:
                cur = list(arr)
                cur_score = score
                # Short local search
                search_deadline = min(deadline - 2, time.time() + 1.5)
                best_score, best_arr = _local_search(cur, cur_score, search_deadline, best_score, best_arr)

    return best_arr
>>>>>>> REPLACE

