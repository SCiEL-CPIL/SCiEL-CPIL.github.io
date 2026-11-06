# Let me try to understand the DAG format by looking at the baseline
# Let me look at what the baseline function does
import autoresearch_suite.science_tcs_expansion as mod
import inspect

# Get the source
src = inspect.getsource(mod.pebble_baseline)
print(src)
