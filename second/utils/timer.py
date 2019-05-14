from __future__ import print_function
from __future__ import division
import time
from contextlib import contextmanager

@contextmanager
def simple_timer(name=''):
    t = time.time()
    yield
    print("{} exec time: {}".format(name, time.time() - t))
