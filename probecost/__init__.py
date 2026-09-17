"""probecost -- what the profiler cost, and what its output entitles you to say.

Three results from one afternoon of real profiling data:

* the same probe cost **1.00x** with a kernel-name filter and **179x** without
  one, and the expensive run used *fewer* replay passes per launch;
* the run whose filter matched **zero** kernels wrote a report, printed a
  throughput 0.1% from the unprofiled baseline, and landed *between* the two
  runs that worked -- the wall clock cannot detect a probe that measured
  nothing;
* per-kernel-*name* summaries average over a mixture: over half the names in
  both workloads span 2x or more within the name, the worst spans a thousand,
  and the two instruments' medians are not two estimates of one quantity.

The device is :mod:`probecost.instrument`: every number declares the probe, its
sampling rule, and whether anything came back.

Zero dependencies, CPU only. Producing ``data/`` needed a GPU and two
profilers; re-checking every conclusion drawn from it needs neither.
"""

from .instrument import InstrumentError, assert_comparable
from .model import CostTable, LaunchSet, SummaryTable, load_all

__version__ = "0.1.0"

__all__ = ["CostTable", "InstrumentError", "LaunchSet", "SummaryTable",
           "__version__", "assert_comparable", "load_all"]
