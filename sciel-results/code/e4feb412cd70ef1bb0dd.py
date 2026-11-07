<<<<<<< SEARCH
            # Run Powell's method from this start
            try:
                result = minimize(
                    counting_func,
                    start,
                    method='Powell',
                    options={'maxiter': 100, 'maxfev': budget - calls_made}
                )
=======
            # Run Nelder-Mead from this start
            try:
                result = minimize(
                    counting_func,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'maxfev': budget - calls_made, 'xatol': 1e-8, 'fatol': 1e-8}
                )
>>>>>>> REPLACE
