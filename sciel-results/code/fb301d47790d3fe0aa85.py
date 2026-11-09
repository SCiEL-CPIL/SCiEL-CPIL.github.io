            if calls_used + lam > budget:
                # Do single steps
                while calls_used < budget and time.time() < deadline:
                    z = np.random.randn(dim)
                    x_new = x_c + np.sqrt(d) * z
                    fx = evaluate(x_new)
                    if fx is None: break
                    if fx < best_fx:
                        x_c = x_new
                        d *= 0.95
                    else:
                        d *= 0.9
                break
