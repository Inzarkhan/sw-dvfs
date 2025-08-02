# configs/example/arm/raspberrypi4_B_8GB_o3_modifiedwithL3.py

import os
import m5
from m5.util import addToPath
from m5.objects import *
from m5.options import *
import argparse
import sys

print(sys.path)
m5.util.addToPath("../..")
from common import SysPaths
from common import ObjectList
from common import MemConfig
from common.cores.arm import O3_ARM_v7a, HPI
import devices
from devices import *
from devices import L3


# Default values for kernel, disk image, and root device
default_kernel = "vmlinux.arm64"
default_disk = "linaro-minimal-aarch64.img"
default_root_device = "/dev/vda1"

cpu_types = {
    "atomic": (AtomicSimpleCPU, None, None, None),
    "minor": (MinorCPU, devices.L1I, devices.L1D, devices.L2),
    "hpi": (HPI.HPI, HPI.HPI_ICache, HPI.HPI_DCache, HPI.HPI_L2),
    "o3": (
        DerivO3CPU,  # Note: Was ArmO3CPU in your log, but DerivO3CPU is standard
        L1I,
        L1D,
        L2Cache,  # Note: Was L2 in your log, but L2Cache is defined in common/Caches.py
    ),
}

performance_levels = {
    "very_low": {"frequency": "0.5GHz", "voltage": "0.7V"},
    "low": {"frequency": "0.8GHz", "voltage": "0.8V"},
    "medium": {"frequency": "1.2GHz", "voltage": "1.0V"},
    "high": {"frequency": "1.8GHz", "voltage": "1.2V"},
}

print("Performance levels:", performance_levels)


def create_cow_image(name):
    image = CowDiskImage()
    image.child.image_file = SysPaths.disk(name)
    return image


# --- CpuPowerModel (Standard structure, reverted from passing subsystem in __init__) ---
class CpuPowerOn(MathExprPowerModel):
    def __init__(self, cpu_path, **kwargs):
        super().__init__(**kwargs)
        self.dyn = f"voltage * (2 * {cpu_path}.ipc + 3 * 0.000000001 * {cpu_path}.dcache.overallMisses / simSeconds)"
        self.st = "4 * temp"


class CpuPowerOff(MathExprPowerModel):
    dyn = "0"
    st = "0"


class CpuPowerModel(PowerModel):
    obj = Param.ClockedObject("The CPU this power model belongs to")

    def __init__(self, cpu_path, **kwargs):
        super(CpuPowerModel, self).__init__(**kwargs)
        self.pm = [
            CpuPowerOn(cpu_path),
            CpuPowerOff(),
            CpuPowerOff(),
            CpuPowerOff(),
        ]


# --- L2 Power Model (From original script) ---
class L2PowerOn(MathExprPowerModel):
    def __init__(self, l2_path, **kwargs):
        super().__init__(**kwargs)
        self.dyn = f"voltage * (2 * {l2_path}.overallHits + 3 * {l2_path}.overallMisses)"
        self.st = "3 * temp"
        # Debug prints removed for cleaner output, can be re-added if needed
        # print(f"L2PowerOn dyn expression: {self.dyn}")
        # print(f"L2PowerOn st expression: {self.st}")


class L2PowerOff(MathExprPowerModel):
    dyn = "0"
    st = "0"


class L2PowerModel(PowerModel):
    obj = Param.ClockedObject("The L2 cache this power model belongs to")

    def __init__(self, l2_path, **kwargs):
        super(L2PowerModel, self).__init__(**kwargs)
        self.pm = [
            L2PowerOn(l2_path),
            L2PowerOff(),
            L2PowerOff(),
            L2PowerOff(),
        ]


# --- L3 Power Model (Standard structure, reverted from passing subsystem in __init__) ---
class L3PowerOn(MathExprPowerModel):
    def __init__(self, l3_path, **kwargs):
        super().__init__(**kwargs)
        self.dyn = f"voltage * (2 * {l3_path}.overallHits + 3 * {l3_path}.overallMisses)"
        self.st = "5 * temp"


class L3PowerOff(MathExprPowerModel):
    dyn = "0"
    st = "0"


class L3PowerModel(PowerModel):
    obj = Param.ClockedObject("The L3 cache this power model belongs to")

    def __init__(self, l3_path, **kwargs):
        super(L3PowerModel, self).__init__(**kwargs)
        self.pm = [
            L3PowerOn(l3_path),
            L3PowerOff(),
            L3PowerOff(),
            L3PowerOff(),
        ]


def create(args):
    print("Creating system...")
    if args.script and not os.path.isfile(args.script):
        print(f"Error: Bootscript {args.script} does not exist")
        sys.exit(1)
    cpu_class = cpu_types[args.cpu][0]
    print(f"Using CPU: {cpu_class}")
    mem_mode = cpu_class.memory_mode()
    want_caches = True if mem_mode == "timing" else False
    print("Initializing SimpleSystem...")
    system = devices.SimpleSystem(
        want_caches,
        args.mem_size,
        mem_mode=mem_mode,
        workload=ArmFsLinux(object_file=SysPaths.binary(args.kernel)),
        readfile=args.script,
    )
    print("System initialized.")

    from m5.util import convert

    # Define memory start and size
    mem_start = 0x80000000
    mem_size = convert.toMemorySize(args.mem_size)
    addr_range = AddrRange(start=mem_start, size=mem_size)
    # Set this range for the system memory
    system.mem_ranges = [addr_range]
    print("Memory range configured as:", system.mem_ranges)

    MemConfig.config_mem(args, system)
    print("Configuring PCI devices...")
    system.pci_devices = [
        PciVirtIO(vio=VirtIOBlock(image=create_cow_image(args.disk_image)))
    ]
    for dev in system.pci_devices:
        system.attach_pci(dev)

    # Get the performance level configuration
    performance_level = performance_levels[args.performance_level]
    print(f"Performance level: {performance_level}")

    # Create and configure CPU cluster using the system's addCpuCluster method
    cpu_cluster = devices.ArmCpuCluster(
        system,
        args.num_cores,
        performance_level["frequency"],
        performance_level["voltage"],
        *cpu_types[args.cpu],
        tarmac_gen=args.tarmac_gen,
        tarmac_dest=args.tarmac_dest,
    )
    print("CPU cluster created.")

    # Access the cluster through the system's _clusters list
    if len(system._clusters) > 0:
        cpu_cluster = system._clusters[
            0
        ]  # Get the first (and should be only) cluster

        # Check and add caches
        if want_caches:
            # Add L1 caches if they don't already exist
            for i, cpu in enumerate(cpu_cluster.cpus):
                if hasattr(cpu, "icache") and hasattr(cpu, "dcache"):
                    print(f"L1 caches already exist for CPU {i}")
                else:
                    print(f"Adding L1 caches to CPU {i}")
                    cpu_cluster.addL1()

            # Add L2 cache only if it doesn't already exist
            if hasattr(cpu_cluster, "l2"):
                print("L2 cache already exists in CPU cluster")
            else:
                print("Adding L2 cache to CPU cluster")
                cpu_cluster.addL2(cpu_cluster.clk_domain)

            # Add L3 Cache conditionally
            if args.enable_l3:
                print("Adding L3 Cache...")
                system.toL3Bus = L2XBar(
                    width=64, clk_domain=cpu_cluster.clk_domain
                )
                system.l3 = L3(clk_domain=cpu_cluster.clk_domain)
                system.toL3Bus.mem_side_ports = system.l3.cpu_side
                system.l3.mem_side = system.membus.cpu_side_ports

                # Connect L2 to L3
                if hasattr(cpu_cluster, "l2"):
                    # Check if already connected to avoid duplicates
                    if not hasattr(cpu_cluster.l2, "_connected_to_l3"):
                        cpu_cluster.l2.mem_side = system.toL3Bus.cpu_side_ports
                        cpu_cluster.l2._connected_to_l3 = True
                        print("L2 connected to L3.")
                    else:
                        print(
                            "Debug: L2 is already connected to L3. Skipping duplicate connection."
                        )
                else:
                    # Handle case with no L2 (connects L1 directly)
                    print(
                        "Warning: No L2 cache found. Connecting L1 directly to L3."
                    )
                    for cpu in cpu_cluster.cpus:
                        # Check if already connected to avoid duplicates
                        if not hasattr(cpu.icache, "_connected_to_l3"):
                            cpu.icache.mem_side = system.toL3Bus.cpu_side_ports
                            cpu.dcache.mem_side = system.toL3Bus.cpu_side_ports
                            cpu.icache._connected_to_l3 = True
                            cpu.dcache._connected_to_l3 = True
                            print(f"L1 caches of CPU {cpu} connected to L3.")
                        else:
                            print(
                                f"Debug: L1 caches of CPU {cpu} are already connected to L3. Skipping duplicate connection."
                            )
            else:
                # Connect L2 to membus if L3 is disabled
                if hasattr(cpu_cluster, "l2"):
                    print("Debug: L2 cache found in CPU cluster.")
                    # Check if already connected to avoid duplicates
                    if not hasattr(cpu_cluster.l2, "_connected_to_membus"):
                        cpu_cluster.l2.mem_side = system.membus.cpu_side_ports
                        cpu_cluster.l2._connected_to_membus = True
                        print("L2 connected to membus.")
                    else:
                        print(
                            "Debug: L2 is already connected to membus. Skipping duplicate connection."
                        )
                else:
                    # Handle case with no L2 (connects L1 directly)
                    print(
                        "Warning: No L2 cache found. Connecting L1 directly to membus."
                    )
                    for cpu in cpu_cluster.cpus:
                        # Check if already connected to avoid duplicates
                        if not hasattr(cpu.icache, "_connected_to_membus"):
                            cpu.icache.mem_side = system.membus.cpu_side_ports
                            cpu.dcache.mem_side = system.membus.cpu_side_ports
                            cpu.icache._connected_to_membus = True
                            cpu.dcache._connected_to_membus = True
                            print(
                                f"L1 caches of CPU {cpu} connected to membus."
                            )
                        else:
                            print(
                                f"Debug: L1 caches of CPU {cpu} are already connected to membus. Skipping duplicate connection."
                            )

        # --- Integrate CPU power model (Reverted to standard way) ---
        for cpu in cpu_cluster.cpus:
            cpu.power_state.default_state = "ON"
            # Removed SubSystem parameter and references to simplify the model
            cpu.power_model = CpuPowerModel(cpu_path=cpu.path())
            print(f"Integrated CpuPowerModel for CPU: {cpu}")

        # --- Integrate L2 power model (Robust handling for potential SimObjectVector) ---
        if hasattr(cpu_cluster, "l2"):
            l2_cache_candidate = cpu_cluster.l2
            print(
                f"DEBUG: Initial cpu_cluster.l2 type: {type(l2_cache_candidate)}"
            )
            print(f"DEBUG: Initial cpu_cluster.l2 value: {l2_cache_candidate}")

            # Import SimObjectVector for type checking
            from m5.SimObject import SimObjectVector

            # Handle potential SimObjectVector robustly
            if isinstance(l2_cache_candidate, SimObjectVector):
                print(
                    "WARNING: cpu_cluster.l2 appears to be a SimObjectVector."
                )
                # Iterate through the vector and configure each L2 cache
                for i, single_l2_cache in enumerate(l2_cache_candidate):
                    print(
                        f"Configuring power model for L2 cache {i}: {single_l2_cache}"
                    )
                    single_l2_cache.power_state.default_state = "ON"
                    single_l2_cache.power_model = L2PowerModel()
                    print(f"L2 power model integrated for cache {i}.")
            else:
                # It's the expected single L2 cache object
                l2_cache = l2_cache_candidate
                print(f"DEBUG: Final l2_cache type: {type(l2_cache)}")
                print(f"DEBUG: Final l2_cache value: {l2_cache}")

                # Configure power state and model for the identified L2 cache
                l2_path = l2_cache.path()  # Define the l2_path explicitly
                l2_cache.power_state.default_state = "ON"
                l2_cache.power_model = L2PowerModel(l2_path=l2_path)
                print("L2 power model integrated successfully.")
        else:
            print("Warning: L2 cache not found in cpu_cluster.")

        # --- Integrate L3 power model (Robust handling for potential SimObjectVector) ---
        if hasattr(system, "l3"):
            print("DEBUG: system.l3 type:", type(system.l3))
            print("DEBUG: system.l3 value:", system.l3)
            l3_cache_candidate = system.l3

            # Handle potential SimObjectVector robustly
            if isinstance(l3_cache_candidate, SimObjectVector):
                print("WARNING: system.l3 appears to be a SimObjectVector.")
                # Iterate through the vector and configure each L3 cache
                for i, single_l3_cache in enumerate(l3_cache_candidate):
                    print(
                        f"Configuring power model for L3 cache {i}: {single_l3_cache}"
                    )
                    single_l3_cache.power_state.default_state = "ON"
                    single_l3_cache.power_model = L3PowerModel()
                    print(f"L3 power model integrated for cache {i}.")
            else:
                # It's the expected single L3 cache object
                l3_cache = l3_cache_candidate
                try:
                    # Configure power state and model for the identified L3 cache
                    l3_path = system.l3.path()  # Define the l3_path explicitly
                    system.l3.power_state.default_state = "ON"
                    system.l3.power_model = L3PowerModel(l3_path)
                    print("L3 power model integrated successfully.")
                except Exception as e:
                    print("ERROR: Could not set L3 power model:", e)
                    raise

        # Integrate L3 power model for runtime power generation using PMC
        if hasattr(system, "l3"):
            system.l3.power_state.default_state = "ON"
            system.l3.power_model = L3PowerModel(system.l3.path())
            print("L3 power model integrated for runtime power generation.")

        # --- DVFS integration ---
        if args.enable_dvfs:
            print("DVFS is enabled. Initializing DVFSHandler...")
            # 1. Create per-core clock domains and collect them
            cpu_domains = []
            for i, cpu in enumerate(cpu_cluster.cpus):
                cpu.clk_domain = SrcClockDomain(
                    clock=performance_level["frequency"],
                    voltage_domain=VoltageDomain(
                        voltage=performance_level["voltage"]
                    ),
                )
                cpu.clk_domain.domain_id = i + 1
                cpu_domains.append(cpu.clk_domain)

            # 2. Pass them to DVFSHandler with correct parameter names
            # Note: Adjust coreIdsToMonitor based on actual number of cores if needed
            core_ids_to_monitor = list(range(args.num_cores))
            system.dvfs_handler = DVFSHandler(
                enable=True,
                transitionLatency="200us",
                coreIdsToMonitor=core_ids_to_monitor,
                sysClkDomain=system.clk_domain,
                performanceLevels=[
                    "very_low:0.5GHz:0.7V",
                    "low:0.8GHz:0.8V",
                    "medium:1.2GHz:1.0V",
                    "high:1.8GHz:1.2V",
                ],
                domains=cpu_domains,
            )
            print("DVFSHandler initialized.")

            # Link DVFS handler to CPU cluster
            cpu_cluster.dvfs_handler = system.dvfs_handler
            print(f"Attached DVFSHandler to CPU cluster: {cpu_cluster}")

        # Store reference for later access
        system.cpu_cluster = cpu_cluster

    # NOW call connect() after all components are configured
    print("Connecting system components...")
    system.connect()

    # Print cache details
    if hasattr(system, "cpu_cluster"):
        print("CPU cluster found. Checking for caches...")
        # Check for L1 caches
        l1_found = False
        for i, cpu in enumerate(system.cpu_cluster.cpus):
            if hasattr(cpu, "icache") and hasattr(cpu, "dcache"):
                l1_found = True
                print(f"L1 Caches for CPU {i}:")
                print(f"  L1 I-Cache: {cpu.icache}")
                print(f"    Size: {cpu.icache.size}")
                print(f"    Associativity: {cpu.icache.assoc}")
                print(f"  L1 D-Cache: {cpu.dcache}")
                print(f"    Size: {cpu.dcache.size}")
                print(f"    Associativity: {cpu.dcache.assoc}")
        if not l1_found:
            print("No L1 caches found in the system.")

        # Check for L2 cache
        if hasattr(system.cpu_cluster, "l2"):
            # Handle potential vector for display
            l2_obj = system.cpu_cluster.l2
            from m5.SimObject import SimObjectVector

            if isinstance(l2_obj, SimObjectVector) and len(l2_obj) > 0:
                l2_display = l2_obj[0]  # Display info for the first one
                print(f"L2 Cache (from vector): {l2_display}")
            elif not isinstance(l2_obj, SimObjectVector):
                l2_display = l2_obj
                print(f"L2 Cache: {l2_display}")
            else:
                print("L2 Cache: <empty vector>")
                l2_display = None

            if l2_display:
                print(f"  L2 Cache Size: {getattr(l2_display, 'size', 'N/A')}")
                print(
                    f"  L2 Cache Associativity: {getattr(l2_display, 'assoc', 'N/A')}"
                )
        else:
            print("No L2 cache found in the cpu_cluster.")

        # Check for L3 cache conditionally
        if args.enable_l3 and hasattr(system, "l3"):
            print(f"L3 Cache: {system.l3}")
            print(f"  L3 Cache Size: {system.l3.size}")
            print(f"  L3 Cache Associativity: {system.l3.assoc}")
        elif not args.enable_l3:
            print("L3 cache is disabled.")
        else:
            print("No L3 cache found.")

    # DTB and kernel command line setup
    if args.dtb:
        print(f"Using DTB file: {args.dtb}")
        system.workload.dtb_filename = args.dtb
    else:
        print("Generating DTB...")
        system.workload.dtb_filename = os.path.join(
            m5.options.outdir, "system.dtb"
        )
    system.generateDtb(system.workload.dtb_filename)

    if args.initrd:
        print(f"Using initrd: {args.initrd}")
        system.workload.initrd_filename = args.initrd

    kernel_cmd = [
        "console=ttyAMA0",
        "lpj=19988480",
        "norandmaps",
        f"root={args.root_device}",
        "rw",
        f"mem={args.mem_size}",
    ]
    print("Kernel command line:", " ".join(kernel_cmd))
    system.workload.command_line = " ".join(kernel_cmd)

    # PMU setup
    if args.with_pmu:
        print(f"Adding PMUs with PPI number: {args.pmu_ppi_number}")
        interrupt_numbers = [args.pmu_ppi_number] * args.num_cores
        system.cpu_cluster.addPMUs(interrupt_numbers)

    print("System creation complete.")
    return system


def run(args):
    cptdir = m5.options.outdir
    if args.checkpoint:
        print(f"Checkpoint directory: {cptdir}")
        while True:
            event = m5.simulate()
            exit_msg = event.getCause()
            if exit_msg == "checkpoint":
                print("Dropping checkpoint at tick %d" % m5.curTick())
                cpt_dir = os.path.join(
                    m5.options.outdir, "cpt.%d" % m5.curTick()
                )
                m5.checkpoint(os.path.join(cpt_dir))
                print("Checkpoint done.")
                # Add periodic stat dumps
                m5.stats.periodicStatDump(
                    m5.ticks.fromSeconds(0.1)
                )  # Dump stats every 0.1 seconds of simulated time
                # Dump stats at the end of the simulation
                m5.stats.dump()
                m5.stats.reset()
            else:
                print(f"{exit_msg} ({event.getCode()}) @ {m5.curTick()}")
                break
    else:
        while True:
            event = m5.simulate()
            exit_msg = event.getCause()
            print(f"{exit_msg} ({event.getCode()}) @ {m5.curTick()}")
            if exit_msg != "simulate":
                break
            # After each stats dump, call monitorWorkload for each core if DVFS is enabled
            # Note: This part might need adjustment based on how DVFS monitoring is intended
            # if (
            #     hasattr(args, "enable_dvfs")
            #     and args.enable_dvfs
            #     and hasattr(system, "dvfs_handler") # 'system' not available here
            # ):
            #     # This loop needs access to the system object, which isn't directly available here
            #     # You might need to pass it or access it differently
            #     pass
        m5.stats.dump()  # Ensure stats are dumped at the end of simulation
        m5.stats.reset()


# Checking the validity of the PPI (Peripheral Private Interrupt) number
def arm_ppi_arg(int_num: int) -> int:
    int_num = int(int_num)
    if 16 <= int_num <= 31:
        return int_num
    raise ValueError(f"{int_num} is not a valid Arm PPI number")


def main():
    parser = argparse.ArgumentParser(epilog=__doc__)
    parser.add_argument(
        "--performance-level",
        type=str,
        choices=list(performance_levels.keys()),
        default="high",
        help="Performance level for DVFS",
    )
    parser.add_argument(
        "--dtb", type=str, default=None, help="DTB file to load"
    )
    parser.add_argument(
        "--kernel", type=str, default=default_kernel, help="Linux kernel"
    )
    parser.add_argument(
        "--initrd",
        type=str,
        default=None,
        help="initrd/initramfs file to load",
    )
    parser.add_argument(
        "--disk-image",
        type=str,
        default=default_disk,
        help="Disk to instantiate",
    )
    parser.add_argument(
        "--root-device",
        type=str,
        default=default_root_device,
        help=f"OS device name for root partition (default: {default_root_device})",
    )
    parser.add_argument(
        "--script", type=str, default="", help="Linux bootscript"
    )
    parser.add_argument(
        "--cpu",
        type=str,
        choices=list(cpu_types.keys()),
        default="o3",
        help="CPU model to use",
    )
    parser.add_argument("--cpu-freq", type=str, default="1.8GHz")
    parser.add_argument(
        "--num-cores", type=int, default=4, help="Number of CPU cores"
    )
    parser.add_argument(
        "--mem-type",
        default="DDR4_2400_4x16",
        choices=ObjectList.mem_list.get_names(),
        help="type of memory to use",
    )
    parser.add_argument(
        "--mem-channels", type=int, default=1, help="number of memory channels"
    )
    parser.add_argument(
        "--mem-ranks",
        type=int,
        default=None,
        help="number of memory ranks per channel",
    )
    parser.add_argument(
        "--mem-size",
        action="store",
        type=str,
        default="4GB",
        help="Specify the physical memory size",
    )
    parser.add_argument(
        "--tarmac-gen", action="store_true", help="Write a Tarmac trace."
    )
    parser.add_argument(
        "--tarmac-dest",
        choices=TarmacDump.vals,
        default="stdoutput",
        help="Destination for the Tarmac trace output. [Default: stdoutput]",
    )
    parser.add_argument(
        "--with-pmu",
        action="store_true",
        help="Add a PMU to each core in the cluster.",
    )
    parser.add_argument(
        "--pmu-ppi-number",
        type=arm_ppi_arg,
        default=23,
        help="The number of the PPI to use to connect each PMU to its core. Must be an integer and a valid PPI number (16 <= int_num <= 31).",
    )
    parser.add_argument(
        "--enable-dvfs", action="store_true", help="Enable DVFS"
    )
    parser.add_argument(
        "--caches", action="store_true", help="Instantiate caches"
    )
    parser.add_argument("--restore", type=str, default=None)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument(
        "--enable-l3",
        action="store_true",
        default=True,
        dest="enable_l3",
        help="Enable L3 cache (default)",
    )
    parser.add_argument(
        "--disable-l3",
        action="store_false",
        dest="enable_l3",
        help="Disable L3 cache",
    )
    args = parser.parse_args()

    root = Root(full_system=True)
    system = create(args)
    root.system = system

    # Ensure the eventq_index is set for the system and its components
    if not hasattr(system, "eventq_index"):
        system.eventq_index = 0
    if hasattr(system, "cpu_cluster"):
        for cpu in system.cpu_cluster.cpus:
            if not hasattr(cpu, "eventq_index"):
                cpu.eventq_index = system.eventq_index
    if hasattr(system, "l3"):
        if not hasattr(system.l3, "eventq_index"):
            system.l3.eventq_index = system.eventq_index

    # Fix parent-child relationships (These might be handled by devices.py already)
    # if hasattr(system, 'cpu_cluster'):
    #     system.cpu_cluster.set_parent(system, "cpu_cluster")
    # if hasattr(system, 'l3'):
    #     system.l3.set_parent(system, "l3")

    # Instantiate the system
    if args.restore is not None:
        m5.instantiate(args.restore)
    else:
        print("About to instantiate...")
        m5.instantiate()
        print("Instantiated.")

    # Enable DVFS if specified (Moved to after instantiate)
    # Note: The way DVFS is enabled/interacted with might need adjustment
    if args.enable_dvfs and hasattr(system, "dvfs_handler"):
        system.dvfs_handler.enable = True
        print("DVFS enabled post-instantiate.")
        print(
            "Initial currentPerf_level:", system.dvfs_handler.currentPerf_level
        )
        try:
            dvfs_handler = system.dvfs_handler
            print("DVFSHandler attributes:", dir(dvfs_handler))

        except AttributeError as e:
            print(f"Error accessing DVFSHandler attributes/methods: {e}")
            # Optionally print system attributes to see what's available
            # print("System attributes:", dir(system))
        except Exception as e:
            print(f"Unexpected error during DVFS post-instantiate setup: {e}")

    # --- End of block to add back ---

    # Run the simulation
    print("Starting simulation...")
    run(args)


if __name__ == "__m5_main__":
    main()
