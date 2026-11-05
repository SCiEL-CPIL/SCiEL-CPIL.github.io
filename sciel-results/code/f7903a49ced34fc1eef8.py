import inspect, sys, time
try:
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    print("BASELINE SRC:")
    print(inspect.getsource(pebble_baseline))
except Exception as e:
    print("ERR baseline:", e)
# try to find a sample dag
import autoresearch_suite.science_tcs_expansion as m
print("MODULE DIR:", [x for x in dir(m) if not x.startswith('__')])
