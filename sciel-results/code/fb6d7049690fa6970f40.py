import os, sys, inspect
try:
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    print("BASELINE SOURCE:")
    print(inspect.getsource(pebble_baseline))
    print("MODULE DIR:", os.path.dirname(inspect.getfile(pebble_baseline)))
except Exception as e:
    print("ERR", e)
