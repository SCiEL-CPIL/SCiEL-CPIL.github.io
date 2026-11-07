            x_best, f_best, budget = self._nelder_mead(
                func, start_x, start_f, budget, t0, deadline
            )
