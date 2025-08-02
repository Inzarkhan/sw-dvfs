from m5.params import *
from m5.proxy import *
from m5.SimObject import SimObject
from m5.util import *

from m5.util.pybind import *


def cxxMethod(*args, **kwargs):
    """Decorator to export C++ functions to Python"""

    def decorate(func):
        name = func.__name__
        override = kwargs.get("override", False)
        cxx_name = kwargs.get("cxx_name", name)
        return_value_policy = kwargs.get("return_value_policy", None)
        static = kwargs.get("static", False)

        # Create a list of tuples of (argument, default). The `PyBindMethod`
        # class expects the `args` argument to be a list of either argument
        # names, in the case that argument does not have a default value, and
        # a tuple of (argument, default) in the case where an argument does.
        args = []
        sig = inspect.signature(func)
        for param_name in sig.parameters.keys():
            if param_name == "self":
                # We don't count 'self' as an argument in this case.
                continue
            param = sig.parameters[param_name]
            if param.default is param.empty:
                args.append(param_name)
            else:
                args.append((param_name, param.default))

        @wraps(func)
        def cxx_call(self, *args, **kwargs):
            ccobj = self.getCCClass() if static else self.getCCObject()
            return getattr(ccobj, name)(*args, **kwargs)

        @wraps(func)
        def py_call(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        f = py_call if override else cxx_call
        f.__pybind = PyBindMethod(
            name,
            cxx_name=cxx_name,
            args=args,
            return_value_policy=return_value_policy,
            static=static,
        )

        return f

    if len(args) == 0:
        return decorate
    elif len(args) == 1 and len(kwargs) == 0:
        return decorate(*args)
    else:
        raise TypeError("One argument and no kwargs, or only kwargs expected")


class DVFSHandler(SimObject):
    type = "DVFSHandler"
    cxx_header = "sim/dvfs_handler.hh"
    cxx_class = "gem5::DVFSHandler"

    sysClkDomain = Param.SrcClockDomain(
        Parent.clk_domain, "Clk domain in which the handler is instantiated"
    )
    enable = Param.Bool(False, "Enable/Disable the handler")
    performanceLevels = VectorParam.String([], "List of performance levels")
    coreIdsToMonitor = VectorParam.UInt32([], "List of core IDs to monitor")
    transitionLatency = Param.Latency(
        "100us", "Fixed latency for performance level migration"
    )
    domains = VectorParam.SrcClockDomain(
        [], "List of clock domains managed by the DVFS handler"
    )
    currentPerf_level = Param.UInt32(1, "Current performance level")

    cxx_exports = [
        PyBindMethod("setPerformanceLevels"),
        PyBindMethod("setCoreIdsToMonitor"),
        PyBindMethod("monitorWorkload"),
        PyBindMethod("adjustDVFS"),
    ]


__all__ = ["DVFSHandler", "cxxMethod", "PyBindMethod", "PyBindProperty"]
