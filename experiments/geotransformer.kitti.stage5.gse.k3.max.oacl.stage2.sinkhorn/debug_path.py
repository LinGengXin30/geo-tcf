
import sys
import os
sys.path.insert(0, os.path.abspath('../../..'))

import geotransformer
import geotransformer.modules.geotransformer.local_global_registration as lgr

print("Python executable:", sys.executable)
print("System path:", sys.path)
print("geotransformer path:", geotransformer.__file__)
print("LocalGlobalRegistration path:", lgr.__file__)

from geotransformer.modules.geotransformer.local_global_registration import LocalGlobalRegistration
import inspect
print("LocalGlobalRegistration.forward signature:", inspect.signature(LocalGlobalRegistration.forward))
