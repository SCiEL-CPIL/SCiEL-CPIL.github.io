<<<<<<< SEARCH
            # Validate
            circles = res3.x.reshape(N, 3)
            if _strictly_valid(circles, w, h, tol=1e-7):
                val = float(np.sum(circles[:, 2]))
                if val > best_val:
                    best_val = val
                    best_result = circles.copy()
=======
            # Validate
            circles = res3.x.reshape(N, 3)
            if _strictly_valid(circles, w, h, tol=1e-7):
                val = float(np.sum(circles[:, 2]))
                if val > best_val:
                    best_val = val
                    best_result = circles.copy()
                    best_w = w
                    best_h = h
>>>>>>> REPLACE

