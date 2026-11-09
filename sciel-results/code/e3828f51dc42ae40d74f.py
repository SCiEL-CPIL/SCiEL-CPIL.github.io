            # 50% chance to use smart move
            if random.random() < 0.5:
                # Pick 3 distinct points
                p_idx = random.sample(range(16), 3)
                p1, p2, p3 = current_pts[p_idx[0]], current_pts[p_idx[1]], current_pts[p_idx[2]]
                
                # Check if they are collinear
                # If not, we can't form a line from them directly.
                # Actually, any 3 points define a line (unless 2 are same, which is not possible).
                # Wait, 3 points define a line. We want to move a 4th point onto this line.
                
                # Get line key
                key = _line_key(p1, p2)
                
                # Find a 4th point to move
                # It must not be p1, p2, p3
                candidates = [i for i in range(16) if i not in p_idx]
                if not candidates:
                    continue
                idx = random.choice(candidates)
                
                # Find a new integer point on this line
                # Line: a*x + b*y + c = 0
                a, b, c = key
                # We need to find x, y in [-1000, 1000] such that a*x + b*y + c = 0
                # And (x,y) is not one of the existing points.
                
                # Simple heuristic: try a few random x values
                found = False
                for _ in range(20):
                    x_new = random.randint(-100, 100)
                    # b*y = -a*x - c
                    if b == 0:
                        if a == 0: continue
                        val = -a*x_new - c
                        if val % b != 0: continue
                        y_new = val // b
                    else:
                        val = -a*x_new - c
                        if val % b != 0: continue
                        y_new = val // b
                        
                    if abs(x_new) <= 1000 and abs(y_new) <= 1000:
                        # Check if this point is already occupied
                        if [x_new, y_new] not in current_pts:
                            current_pts[idx] = [x_new, y_new]
                            new_score = _evaluate(current_pts)
                            if new_score > current_score:
                                current_score = new_score
                                no_improve = 0
                                if current_score > local_best_score:
                                    local_best_score = current_score
                                    local_best_pts = [p[:] for p in current_pts]
                            else:
                                current_pts[idx] = old_pt # Need to store old_pt before this block
                            found = True
                            break
                if not found:
                    # Revert if we changed it? 
                    # In the code above, I only change current_pts[idx] if found.
                    # But I need to handle the revert if score doesn't improve.
                    # Let's restructure the smart move to be cleaner.
            else:
                # Random move as before
