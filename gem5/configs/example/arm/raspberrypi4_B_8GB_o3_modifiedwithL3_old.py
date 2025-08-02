import os
import m5
from m5.util import *
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
import devices1
from devices1 import L3


# Default values for kernel, disk image, and root device
default_kernel = "vmlinux.arm64"
default_disk = "linaro-minimal-aarch64.img"
default_root_device = "/dev/vda1"

cpu_types = {
    "atomic": (AtomicSimpleCPU, None, None, None),
    "minor": (MinorCPU, devices1.L1I, devices1.L1D, devices1.L2),
    "hpi": (HPI.HPI, HPI.HPI_ICache, HPI.HPI_DCache, HPI.HPI_L2),
    "o3": (
        O3_ARM_v7a.O3_ARM_v7a_3,
        O3_ARM_v7a.O3_ARM_v7a_ICache,
        O3_ARM_v7a.O3_ARM_v7a_DCache,
        O3_ARM_v7a.O3_ARM_v7aL2,
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


class CpuPowerOn(MathExprPowerModel):
    def __init__(self, cpu_path, **kwargs):
        super(CpuPowerOn, self).__init__(**kwargs)
        self.dyn = "voltage * (2 * {}.ipc + 3 * 0.000000001 * {}.dcache.overallMisses / simSeconds)".format(
            cpu_path, cpu_path
        )
        # self.dyn = "voltage * voltage * freqGHz * (0.7 * ipc + 0.3 * dcache.overallMisses / simSeconds)"
        self.st = "4 * temp"


class CpuPowerOff(MathExprPowerModel):
    dyn = "0"
    st = "0"


class CpuPowerModel(PowerModel):
    def __init__(self, cpu_path, **kwargs):
        super(CpuPowerModel, self).__init__(**kwargs)
        self.pm = [
            CpuPowerOn(cpu_path),  # ON
            CpuPowerOff(),  # CLK_GATED
            CpuPowerOff(),  # SRAM_RETENTION
            CpuPowerOff(),  # OFF
        ]


# L2 Power Model
class L2PowerOn(MathExprPowerModel):
    def __init__(self, l2_path, **kwargs):
        super(L2PowerOn, self).__init__(**kwargs)
        print(f"Initializing L2PowerOn with l2_path: {l2_path}")
        if (
            not l2_path
            or not hasattr(l2_path, "overallHits")
            or not hasattr(l2_path, "overallMisses")
        ):
            print(f"Warning: Invalid l2_path: {l2_path}")
            self.dyn = "0"
            self.st = "0"
        else:
            self.dyn = f"voltage * (2 * {l2_path}.overallHits + 3 * {l2_path}.overallMisses)"
            self.st = "3 * temp"

        # Debug prints to check expressions
        print(f"L2PowerOn dyn expression: {self.dyn}")
        print(f"L2PowerOn st expression: {self.st}")


class L2PowerOff(MathExprPowerModel):
    dyn = "0"
    st = "0"


class L2PowerModel(PowerModel):
    def __init__(self, l2_path, **kwargs):
        super(L2PowerModel, self).__init__(**kwargs)
        # Choose a power model for every power state
        self.pm = [
            L2PowerOn(l2_path),  # ON
            L2PowerOff(),  # CLK_GATED
            L2PowerOff(),  # SRAM_RETENTION
            L2PowerOff(),  # OFF
        ]


# L3 Power Model
class L3PowerOn(MathExprPowerModel):
    def __init__(self, l3_path, **kwargs):
        super(L3PowerOn, self).__init__(**kwargs)
        self.dyn = f"voltage * (2 * {l3_path}.overallHits + 3 * {l3_path}.overallMisses)"
        self.st = "5 * temp"


class L3PowerOff(MathExprPowerModel):
    dyn = "0"
    st = "0"


class L3PowerModel(PowerModel):
    def __init__(self, l3_path, **kwargs):
        super(L3PowerModel, self).__init__(**kwargs)
        self.pm = [
            L3PowerOn(l3_path),  # ON
            L3PowerOff(),  # CLK_GATED
            L3PowerOff(),  # SRAM_RETENTION
            L3PowerOff(),  # OFF
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
    system = devices1.SimpleSystem(
        want_caches,
        args.mem_size,
        mem_mode=mem_mode,
        workload=ArmFsLinux(object_file=SysPaths.binary(args.kernel)),
        readfile=args.script,
    )

    MemConfig.config_mem(args, system)
    print("Configuring PCI devices...")
    system.pci_devices = [
        PciVirtIO(vio=VirtIOBlock(image=create_cow_image(args.disk_image)))
    ]

    for dev in system.pci_devices:
        system.attach_pci(dev)

    system.connect()

    # Get the performance level configuration
    performance_level = performance_levels[args.performance_level]
    print(f"Performance level: {performance_level}")

    # Create and configure CPU cluster
    system.cpu_cluster = devices1.ArmCpuCluster(
        system,
        args.num_cores,
        performance_level["frequency"],
        performance_level["voltage"],
        *cpu_types[args.cpu],
        tarmac_gen=args.tarmac_gen,
        tarmac_dest=args.tarmac_dest,
    )
    print("CPU cluster created.")

    # Check and add caches
    if want_caches:
        # Add L1 caches if they don't already exist
        for i, cpu in enumerate(system.cpu_cluster.cpus):
            if hasattr(cpu, "icache") and hasattr(cpu, "dcache"):
                print(f"L1 caches already exist for CPU {i}")
            else:
                print(f"Adding L1 caches to CPU {i}")
                system.cpu_cluster.addL1()

        # Add L2 cache only if it doesn't already exist
        if hasattr(system.cpu_cluster, "l2"):
            print("L2 cache already exists in CPU cluster")
        else:
            print("Adding L2 cache to CPU cluster")
            system.cpu_cluster.addL2(system.cpu_cluster.clk_domain)

        # Add L3 Cache conditionally
        if args.enable_l3:
            print("Adding L3 Cache...")
            # system.toL3Bus = L2XBar(width=64)
            system.toL3Bus = L2XBar(
                width=64, clk_domain=system.cpu_cluster.clk_domain
            )
            system.l3 = L3(clk_domain=system.cpu_cluster.clk_domain)
            system.toL3Bus.mem_side_ports = system.l3.cpu_side
            system.l3.mem_side = system.membus.cpu_side_ports

            # Connect L2 to L3 (or L1 to L3 if no L2)
            if hasattr(system.cpu_cluster, "l2"):
                system.cpu_cluster.l2.mem_side = system.toL3Bus.cpu_side_ports
            else:
                print(
                    "Warning: L2 cache not found. Connecting L1 directly to L3."
                )
                for cpu in system.cpu_cluster.cpus:
                    cpu.icache.mem_side = system.toL3Bus.cpu_side_ports
                    cpu.dcache.mem_side = system.toL3Bus.cpu_side_ports

            # Optionally integrate L3 power model
            system.l3.power_state.default_state = "ON"
            system.l3.power_model = L3PowerModel(system.l3.path())
            print("L3 cache integrated with power model.")
        else:
            print("L3 Cache is disabled.")
            # Connect L2 directly to membus if L3 is disabled
            if hasattr(system.cpu_cluster, "l2"):
                system.cpu_cluster.l2.mem_side = system.membus.cpu_side_ports
            else:
                print(
                    "Warning: No L2 cache found. Connecting L1 directly to membus."
                )
                for cpu in system.cpu_cluster.cpus:
                    cpu.icache.mem_side = system.membus.cpu_side_ports
                    cpu.dcache.mem_side = system.membus.cpu_side_ports

    # Integrate CPU power model
    for cpu in system.cpu_cluster.cpus:
        cpu.power_state.default_state = "ON"
        cpu.power_model = CpuPowerModel(cpu.path())
        print(f"Integrated CpuPowerModel for CPU: {cpu.path()}")

    # Integrate L2 power model
    if hasattr(system.cpu_cluster, "l2"):
        l2_cache = system.cpu_cluster.l2
        l2_cache.power_state.default_state = "ON"
        l2_cache.power_model = L2PowerModel(l2_cache.path())
        print("L2 power model integrated.")
    else:
        print("Warning: L2 cache not found in cpu_cluster.")

    # DVFS integration
    if args.enable_dvfs:
        print("DVFS is enabled. Initializing DVFSHandler...")
        # 1. Create per-core clock domains and collect them
        cpu_domains = []
        for i, cpu in enumerate(system.cpu_cluster.cpus):
            cpu.clk_domain = SrcClockDomain(
                clock=performance_level["frequency"],
                voltage_domain=VoltageDomain(
                    voltage=performance_level["voltage"]
                ),
            )
            cpu.clk_domain.domain_id = i + 1
            cpu_domains.append(cpu.clk_domain)

        # 2. Pass them to DVFSHandler at construction
        system.dvfs_handler = DVFSHandler(
            enable=True,
            transition_latency="200us",
            core_ids_to_monitor=[0, 1],  # or range(args.num_cores)
            sys_clk_domain=system.clk_domain,
            performance_levels=[
                "low:0.8GHz:0.8V",
                "medium:1.2GHz:1.0V",
                "high:1.8GHz:1.2V",
            ],
            domains=cpu_domains,
            downscale_window=4,
            upscale_window=4,
            ipc_idle_threshold=0.1,
        )
        print("DVFSHandler initialized.")

        # Link DVFS handler to CPU cluster
        system.cpu_cluster.dvfs_handler = system.dvfs_handler
        print(f"Attached DVFSHandler to CPU cluster: {system.cpu_cluster}")

    # Print cache details
    if hasattr(system, "cpu_cluster"):
        print("CPU cluster found. Checking for caches...")

        # Check for L1 caches
        l1_found = False
        for i, cpu in enumerate(system.cpu_cluster.cpus):
            if hasattr(cpu, "icache") and hasattr(cpu, "dcache"):
                l1_found = True
                print(f"L1 Caches for CPU {i}:")
                print(f"L1 I-Cache: {cpu.icache}")
                print(f"Size: {cpu.icache.size}")
                print(f"Associativity: {cpu.icache.assoc}")
                print(f"L1 D-Cache: {cpu.dcache}")
                print(f"Size: {cpu.dcache.size}")
                print(f"Associativity: {cpu.dcache.assoc}")

        if not l1_found:
            print("No L1 caches found in the system.")

        # Check for L2 cache
        if hasattr(system.cpu_cluster, "l2"):
            print(f"L2 Cache: {system.cpu_cluster.l2}")
            print(f"L2 Cache Size: {system.cpu_cluster.l2.size}")
            print(f"L2 Cache Associativity: {system.cpu_cluster.l2.assoc}")
        else:
            print("No L2 cache found in the cpu_cluster.")

        # Check for L3 cache conditionally
        if args.enable_l3 and hasattr(system, "l3"):
            print(f"L3 Cache: {system.l3}")
            print(f"L3 Cache Size: {system.l3.size}")
            print(f"L3 Cache Associativity: {system.l3.assoc}")
        elif not args.enable_l3:
            print("L3 cache is disabled.")
        else:
            print("No L3 cache found.")

    system.realview.setupBootLoader(system, SysPaths.binary)

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
            if (
                hasattr(args, "enable_dvfs")
                and args.enable_dvfs
                and hasattr(system, "dvfs_handler")
            ):
                dvfs_handler = system.dvfs_handler
                for core_id in range(args.num_cores):
                    workload_name = f"core_{core_id}_workload"
                    dvfs_handler.monitorWorkload(core_id, workload_name)
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
        default="8GB",
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

    # Instantiate the system
    if args.restore is not None:
        m5.instantiate(args.restore)
    else:
        m5.instantiate()

    # Enable DVFS if specified
    if args.enable_dvfs and hasattr(system, "dvfs_handler"):
        system.dvfs_handler.enable = True
        print("DVFS enabled.")

        print("currentPerf_level:", system.dvfs_handler.currentPerf_level)
        try:
            dvfs_handler = system.dvfs_handler
            print("DVFSHandler attributes:", dir(dvfs_handler))

            if hasattr(dvfs_handler, "setPerformanceLevels"):
                print("Setting performance levels...")
                dvfs_handler.setPerformanceLevels(
                    [
                        "low:0.8GHz:0.8V",
                        "medium:1.2GHz:1.0V",
                        "high:1.8GHz:1.2V",
                    ]
                )
                print("Performance levels set successfully.")
            else:
                print(
                    "DVFSHandler doesn't have a setPerformanceLevels method."
                )

            if hasattr(dvfs_handler, "setCoreIdsToMonitor"):
                # Generate core IDs based on the number of cores
                core_ids_to_monitor = list(range(args.num_cores))
                print(f"Setting core IDs to monitor: {core_ids_to_monitor}")
                dvfs_handler.setCoreIdsToMonitor(core_ids_to_monitor)
                print("Core IDs set successfully.")
            else:
                print("DVFSHandler doesn't have a setCoreIdsToMonitor method.")

            if hasattr(dvfs_handler, "monitorWorkload"):
                print("Starting workload monitoring for each core...")
                for core_id in core_ids_to_monitor:
                    workload_name = f"core_{core_id}_workload"
                    dvfs_handler.monitorWorkload(core_id, workload_name)
                    print(
                        f"Monitoring started for {workload_name} on core {core_id}"
                    )
            else:
                print("DVFSHandler doesn't have a monitorWorkload method.")

        except AttributeError as e:
            print(f"Error accessing DVFSHandler: {e}")
            print("System attributes:", dir(system))
        except Exception as e:
            print(f"Unexpected error: {e}")

    # Run the simulation
    run(args)


if __name__ == "__m5_main__":
    main()
