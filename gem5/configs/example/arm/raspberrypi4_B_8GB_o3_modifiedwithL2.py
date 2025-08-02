# configs/example/arm/raspberrypi4_B_8GB_o3_modifiedwithL2.py

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

# Default values for kernel, disk image, and root device
default_kernel = "vmlinux.arm64"
default_disk = "linaro-minimal-aarch64.img"
default_root_device = "/dev/vda1"

cpu_types = {
    "atomic": (AtomicSimpleCPU, None, None, None),
    "minor": (MinorCPU, L1I, L1D, L2),
    "hpi": (HPI.HPI, HPI.HPI_ICache, HPI.HPI_DCache, HPI.HPI_L2),
    "o3": (
        DerivO3CPU,
        L1I,
        L1D,
        L2,
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
    # Use the full path directly if provided, otherwise use SysPaths.disk
    if os.path.isabs(name):
        image.child.image_file = name
    else:
        image.child.image_file = SysPaths.disk(name)
    print("[DEBUG] Using disk image:", image.child.image_file)
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
    # obj = Param.ClockedObject("The CPU this power model belongs to")
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
    # obj = Param.ClockedObject("The L2 cache this power model belongs to")
    def __init__(self, l2_path, **kwargs):
        super(L2PowerModel, self).__init__(**kwargs)
        self.pm = [
            L2PowerOn(l2_path),
            L2PowerOff(),
            L2PowerOff(),
            L2PowerOff(),
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
    print(f"DEBUG: CPU Class: {cpu_class}, Memory Mode: {mem_mode}, Want Caches: {want_caches}")
    print("Initializing SimpleSystem...")
    system = SimpleSystem(
        want_caches,
        args.mem_size,
        mem_mode=mem_mode,
        workload=ArmFsLinux(object_file=SysPaths.binary(args.kernel)),
        readfile=args.script,  # <-- This should be set and the file should exist
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
    # print("Configuring PCI devices...")
    # system.pci_devices = [
    #     PciVirtIO(vio=VirtIOBlock(image=create_cow_image(args.disk_image)))
    # ]
    # for dev in system.pci_devices:
    #     system.attach_pci(dev)

    # --- Attach Disk Image using PciVirtIO and system.attach_pci ---
    print("[DEBUG] Attaching disk image using system.attach_pci...")
    disk_image = create_cow_image(args.disk_image)
    virtio_blk = VirtIOBlock(image=disk_image)
    pci_virtio_blk = PciVirtIO(vio=virtio_blk)
    system.pci_disk_device = pci_virtio_blk  # Assign as a named child attribute
    system.attach_pci(pci_virtio_blk)        # Use the provided method
    print("[DEBUG] Disk image attached using system.attach_pci.")
    # --- End of Disk Attachment ---

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
    # Attach the cluster as a system attribute to avoid orphan nodes
    system.cpu_cluster = cpu_cluster
    # Use system.cpu_cluster for all further setup
    cpu_cluster = system.cpu_cluster
    # Ensure L2 cache is a direct attribute of the cluster
    if hasattr(cpu_cluster, "l2"):
        cpu_cluster.l2 = cpu_cluster.l2
    # Ensure L1 caches are direct attributes of the CPUs in the cluster
    for i, cpu in enumerate(cpu_cluster.cpus):
        if hasattr(cpu, "icache"):
            cpu.icache = cpu.icache
        if hasattr(cpu, "dcache"):
            cpu.dcache = cpu.dcache
    print("CPU cluster created.")

    # Access the cluster through the system's _clusters list
    if len(system._clusters) > 0:
        cpu_cluster = system._clusters[0] # Get the first (and should be only) cluster

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

        # --- Integrate CPU power model (Reverted to standard way) ---
        for cpu in cpu_cluster.cpus:
            cpu.power_state.default_state = "ON"
            cpu.power_model = CpuPowerModel(cpu_path=cpu.path())
            print(f"Integrated CpuPowerModel for CPU: {cpu}")

        # --- Integrate L2 power model (Robust handling for potential SimObjectVector) ---
        if hasattr(cpu_cluster, "l2"):
            l2_cache_candidate = cpu_cluster.l2
            from m5.SimObject import SimObjectVector
            if isinstance(l2_cache_candidate, SimObjectVector):
                for i, single_l2_cache in enumerate(l2_cache_candidate):
                    single_l2_cache.power_state.default_state = "ON"
                    try:
                        single_l2_cache.power_model = L2PowerModel(l2_path=single_l2_cache.path())
                        print(f"L2 power model integrated for cache {i}.")
                    except RuntimeError as e:
                        if "Cycle found" in str(e):
                            print(f"Skipped L2PowerModel assignment for L2 cache {i} to avoid cycle.")
                        else:
                            raise
                    except Exception as e:
                        print(f"Failed to assign L2PowerModel to L2 cache {i}: {e}")
            else:
                l2_cache = l2_cache_candidate
                l2_cache.power_state.default_state = "ON"
                try:
                    l2_cache.power_model = L2PowerModel(l2_path=l2_cache.path())
                    print("L2 power model integrated successfully.")
                except RuntimeError as e:
                    if "Cycle found" in str(e):
                        print("Skipped L2PowerModel assignment to avoid cycle.")
                    else:
                        raise
                except Exception as e:
                    print(f"Failed to assign L2PowerModel to L2 cache: {e}")
        else:
            print("Warning: L2 cache not found in cpu_cluster.")


        # --- DVFS integration (DELAYED ASSIGNMENT APPROACH) ---
        if args.enable_dvfs:
            print("DVFS is enabled. Preparing DVFSHandler for delayed assignment...")
            # --- REMOVED: Explicit setting of system.clk_domain.eventq_index ---
            # The system.clk_domain should inherit eventq_index correctly.
            # Explicitly setting it might interfere with internal connections.
            # --- END OF REMOVAL ---

            # --- Create and configure domains, handler, delay assignment ...
            cpu_domains = []
            core_ids_to_monitor = []
            for i, cpu in enumerate(cpu_cluster.cpus):
                clk_domain_instance = SrcClockDomain(
                    clock=performance_level["frequency"],
                    voltage_domain=VoltageDomain(voltage=performance_level["voltage"]),
                )
                # KEEP THIS - Explicitly setting eventq_index on the domains we create is good
                clk_domain_instance.eventq_index = 0
                clk_domain_instance.domain_id = i + 1
                cpu.clk_domain = clk_domain_instance
                cpu_domains.append(clk_domain_instance)
                core_ids_to_monitor.append(i)
                print(f"CPU {i} clock domain created/configured/assigned (eventq_index=0).")
            print("All CPU clock domains created/configured/assigned.")

            dvfs_handler_for_later = DVFSHandler(
                enable=True,
                transitionLatency="200us",
                coreIdsToMonitor=core_ids_to_monitor,
                sysClkDomain=system.clk_domain, # Pass the system's domain
                performanceLevels=[
                    "very_low:0.5GHz:0.7V",
                    "low:0.8GHz:0.8V",
                    "medium:1.2GHz:1.0V",
                    "high:1.8GHz:1.2V",
                ],
                domains=cpu_domains,
            )
            # KEEP THIS - Explicitly setting eventq_index on the handler instance is good
            dvfs_handler_for_later.eventq_index = 0
            print("DVFSHandler instance created, configured, and eventq_index=0 set.")
            print("DVFSHandler prepared for assignment AFTER m5.instantiate().")
            system.dvfs_handler = dvfs_handler_for_later
            print("DVFSHandler assigned to system.dvfs_handler.")

        # END of if args.enable_dvfs block

    # Print cache details
    # if hasattr(system, "cpu_cluster"):
    #     ...
    # Instead, use the cluster from system._clusters[0]
    
    if len(system._clusters) > 0:
        cpu_cluster = system._clusters[0]
        print("CPU cluster found. Checking for caches...")
        # Check for L1 caches
        l1_found = False
        for i, cpu in enumerate(cpu_cluster.cpus):
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
        if hasattr(cpu_cluster, "l2"):
            l2_obj = cpu_cluster.l2
            from m5.SimObject import SimObjectVector
            if isinstance(l2_obj, SimObjectVector) and len(l2_obj) > 0:
                l2_display = l2_obj[0] # Display info for the first one
                print(f"L2 Cache (from vector): {l2_display}")
            elif not isinstance(l2_obj, SimObjectVector):
                l2_display = l2_obj
                print(f"L2 Cache: {l2_display}")
            else:
                print("L2 Cache: <empty vector>")
                l2_display = None

            if l2_display:
                print(f"  L2 Cache Size: {getattr(l2_display, 'size', 'N/A')}")
                print(f"  L2 Cache Associativity: {getattr(l2_display, 'assoc', 'N/A')}")
        else:
            print("No L2 cache found in the cpu_cluster.")

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
        "earlycon=pl011,0x1c090000",  # Added for early UART output
        "earlyprintk",               # Added for early printk output
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

    # After all system/cpu/cache setup and before DTB/initrd/kernel_cmd
    # Ensure bootloader is set up
    system.realview.setupBootLoader(system, SysPaths.binary)

    # --- THIS IS THE CORRECT PLACE ---
    print("Connecting system components...")
    system.connect()
    # --- END OF CORRECT PLACE ---

    print("System creation complete.")
    prepared_dvfs_handler = getattr(system, '_prepared_dvfs_handler', None)
    return system, prepared_dvfs_handler, cpu_cluster

def run(args, system, prepared_dvfs_handler):
    cptdir = m5.options.outdir
    if args.checkpoint:
        print(f"Checkpoint directory: {cptdir}")
        while True:
            event = m5.simulate()
            exit_msg = event.getCause()
            if exit_msg == "checkpoint":
                print("Dropping checkpoint at tick %d" % m5.curTick())
                cpt_dir = os.path.join(m5.options.outdir, "cpt.%d" % m5.curTick())
                m5.checkpoint(cpt_dir)
                print("Checkpoint done.")

                # Dump stats
                m5.stats.dump()
                m5.stats.reset()

                # Call monitorWorkload
                if args.enable_dvfs and hasattr(system, "dvfs_handler"):
                    dvfs_handler = system.dvfs_handler
                    for core_id in getattr(dvfs_handler, 'coreIdsToMonitor', []):
                        print(f"[DVFS] Calling monitorWorkload for core {core_id}")
                        dvfs_handler.monitorWorkload(core_id, "dynamic")

            else:
                print(f"{exit_msg} ({event.getCode()}) @ {m5.curTick()}")
                break
    else:
        while True:
            event = m5.simulate()
            exit_msg = event.getCause()
            print(f"{exit_msg} ({event.getCode()}) @ {m5.curTick()}")

            # Periodic stats dump every 100ms
            if m5.curTick() % (100 * 1000000) == 0:
                m5.stats.dump()
                m5.stats.reset()

                if args.enable_dvfs and hasattr(system, "dvfs_handler"):
                    dvfs_handler = system.dvfs_handler
                    for core_id in getattr(dvfs_handler, 'coreIdsToMonitor', []):
                        dvfs_handler.monitorWorkload(core_id, "dynamic")

            if exit_msg != "simulate":
                break

        m5.stats.dump()
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
    args = parser.parse_args()
    root = Root(full_system=True)
    # --- NEW FIX ATTEMPT: Explicitly set Root's eventq_index ---
    root.eventq_index = 0
    print("Root object created and eventq_index=0 set.")
    # ---------------------------------------------
    system, prepared_dvfs_handler, cpu_cluster = create(args)
    root.system = system

    # Remove all post-construction eventq_index assignments

    # Print eventq_index debug info BEFORE instantiate (to avoid AttributeError)
    # print("DEBUG: System eventq_index:", getattr(system, 'eventq_index', None))
    # if hasattr(system, 'cpu_cluster'):
    #     for cpu in system.cpu_cluster.cpus:
    #         print(f"DEBUG: CPU {cpu} eventq_index:", getattr(cpu, 'eventq_index', None))
    # if hasattr(system, 'dvfs_handler'):
    #     print("DEBUG: DVFSHandler eventq_index:", getattr(system.dvfs_handler, 'eventq_index', None))

    # Ensure all clusters are connected to the memory hierarchy before instantiation
    for cluster in system._clusters:
        cluster.connectMemSide(system.membus)
    print(f"DEBUG: Parent of cpu_cluster: {cpu_cluster.get_parent()}")
    print(f"DEBUG: Parent of cpu_cluster.cpus[0]: {cpu_cluster.cpus[0].get_parent()}")
    print(f"DEBUG: Parent of system.membus: {system.membus.get_parent()}")

    # Instantiate the system
    if args.restore is not None:
        m5.instantiate(args.restore)
    else:
        m5.instantiate()
        print("System successfully instantiated.")

    # DO NOT print or access eventq_index after this point!

    # Enable DVFS if specified (Moved to after instantiate)
    # Note: The way DVFS is enabled/interacted with might need adjustment
    if args.enable_dvfs and hasattr(system, "dvfs_handler"):
        system.dvfs_handler.enable = True
        print("DVFS enabled post-instantiate.")
        print("Initial currentPerf_level:", system.dvfs_handler.currentPerf_level)


    # Run the simulation
    print("Starting simulation...")
    run(args, system, prepared_dvfs_handler)

if __name__ == "__m5_main__":
    main()
