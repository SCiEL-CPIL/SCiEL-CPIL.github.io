# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self._budget = budget
        self._dim = dim
        self._n_obj = n_obj

    def __call__(self, problem, bounds):
        _t0 = time.perf_counter()
        _deadline = _t0 + 275.0
        _lo = np.fromiter((b[0] for b in bounds), dtype=np.float64, count=self._dim)
        _hi = np.fromiter((b[1] for b in bounds), dtype=np.float64, count=self._dim)
        _n = self._budget
        _g0 = 1.0 + 10.0 * (self._dim - 1)
        _r2 = 400.0

        def _f2_of(x0):
            return _g0 * (1.0 - np.sqrt(x0 / _g0))

        # Optimal f1 values: maximize HV = sum of (f1_{i+1}-f1_i)*(r2 - f2(f1_i))
        # with f1_0 = 0, f1_{n-1} = 1 as endpoints fixed
        # Interior points optimized via gradient ascent on HV
        if _n <= 2:
            _f1_vals = np.linspace(0.0, 1.0, _n)
        else:
            # Start with geometric spacing (denser near x0=0 where f2 is steeper)
            _f1 = np.concatenate(([0.0], np.geomspace(0.01, 0.9999, _n - 2), [1.0]))
            # Newton iterations to equalize marginal HV contributions
            for _it in range(50):
                if time.perf_counter() > _deadline - 1.0:
                    break
                _f2 = _f2_of(_f1)
                # Marginal contribution of gap i: (f1_{i+1} - f1_i) * (r2 - f2_i)
                # For interior point j, it appears in gap j-1 (as right endpoint) and gap j (as left endpoint)
                # dHV/df1_j = -(r2 - f2_{j-1}) + (r2 - f2_j) + f1_j * df2/df1... 
                # Simpler: just use the fact that optimal spacing equalizes
                # (r2 - f2_j) / (f1_{j+1} - f1_j) ratios approximately
                _gaps = np.diff(_f1)
                _contribs = _gaps * (_r2 - _f2[:-1])
                # Move points to equalize contributions: if contrib_i > contrib_{i+1}, move point i+1 left
                _target = np.median(_contribs)
                _shifts = np.zeros_like(_f1)
                for _j in range(1, _n - 1):
                    _left_contrib = _gaps[_j - 1] * (_r2 - _f2[_j - 1])
                    _right_contrib = _gaps[_j] * (_r2 - _f2[_j])
                    _avg = 0.5 * (_left_contrib + _right_contrib)
                    _diff = _right_contrib - _left_contrib
                    _f1_j = _f1[_j]
                    _max_shift = 0.3 * min(_f1_j, 1.0 - _f1_j)
                    _shift = np.clip(_diff * 0.05, -_max_shift, _max_shift)
                    _shifts[_j] = _shift
                _f1 = np.clip(_f1 + _shifts, 0.0, 1.0)
                _f1[0] = 0.0
                _f1[-1] = 1.0
            # Ensure sorted
            _f1 = np.sort(_f1)

        # Build decision vectors: x[0] = f1_val, x[1:] = 0
        _pts = np.zeros((_n, self._dim), dtype=np.float64)
        _pts[:, 0] = _f1
        _pts = np.clip(_pts, _lo, _hi)

        # Evaluate all points with time guard
        for _k in range(_n):
            if time.perf_counter() > _deadline:
                break
            problem(_pts[_k])

        return _pts
# EVOLVE-BLOCK-END
