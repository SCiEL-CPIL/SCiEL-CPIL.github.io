                # Grow and relax
                if time.time() > deadline - 5:
                    break
                params, s = _grow_and_relax(params, deadline - 5)
                params = params.flatten() # Ensure 1D for next iteration
