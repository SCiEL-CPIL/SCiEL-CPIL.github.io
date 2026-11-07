        while alive() and no_improve_count < max_no_improve:
            improved_in_pass = False
            for axis in range(d):
                if not alive():
                    break
                    
                s = steps[axis]
                
                # Try positive
                cand = x.copy()
                cand[axis] = x[axis] + s
                cand = np.clip(cand, -5, 5)
                if not np.allclose(cand, x, rtol=0, atol=1e-12):
                    fc = wrapped(cand)
                    if fc < fx:
                        x = cand
                        fx = fc
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        improved_in_pass = True
                        steps[axis] = min(steps[axis] * 1.1, max_step)
                        continue

                # Try negative
                cand = x.copy()
                cand[axis] = x[axis] - s
                cand = np.clip(cand, -5, 5)
                if not np.allclose(cand, x, rtol=0, atol=1e-12):
                    fc = wrapped(cand)
                    if fc < fx:
                        x = cand
                        fx = fc
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        improved_in_pass = True
                        steps[axis] = min(steps[axis] * 1.1, max_step)
                        continue

                # No improvement
                steps[axis] *= 0.7
                if steps[axis] < min_step:
                    steps[axis] = min_step

            if improved_in_pass:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if np.all(steps < 1e-4):
                    break
