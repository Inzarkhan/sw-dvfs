#include "sim/dvfs_handler.hh"
#include "base/trace.hh"
#include "debug/DVFS.hh"
#include "params/DVFSHandler.hh"
#include "sim/serialize.hh"
#include "sim/stat_control.hh"
#include "sim/voltage_domain.hh"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace gem5
{

DVFSHandler *DVFSHandler::UpdateEvent::dvfsHandler;

DVFSHandler::DVFSHandler(const Params &p)
    : SimObject(p),
      enableHandler(p.enable),
      ipcThresholdLow(0.1),
      ipcThresholdHigh(0.3),
      cacheMissRateThreshold(0.1),
      fetchRateThreshold(1000000),
      hysteresisThreshold(3),
      currentPerf_level(static_cast<PerfLevelEnum>(p.currentPerf_level)), // Cast from uint32_t to enum
      coreIdsToMonitor(p.coreIdsToMonitor),
      performanceLevels(p.performanceLevels),
      sysClkDomain(p.sysClkDomain),
      _transLatency(p.transitionLatency)
{
    for (auto dit = p.domains.begin(); dit != p.domains.end(); ++dit)
    {
        SrcClockDomain *d = *dit;
        DomainID domain_id = d->domainID();
        fatal_if(sysClkDomain == d, "DVFS: Domain config list has a system clk domain entry");
        fatal_if(domain_id == SrcClockDomain::emptyDomainID,
                 "DVFS: Controlled domain %s needs to have a properly assigned ID.\n", d->name());
        auto entry = std::make_pair(domain_id, d);
        bool new_elem = domains.insert(entry).second;
        fatal_if(!new_elem, "DVFS: Domain %s with ID %d does not have a unique ID.\n", d->name(), domain_id);
        UpdateEvent *event = &updatePerfLevelEvents[domain_id];
        event->domainIDToSet = d->domainID();
        domainIDList.push_back(d->domainID());
    }
    UpdateEvent::dvfsHandler = this;
}

DVFSHandler::UpdateEvent::UpdateEvent()
    : Event(DVFS_Update_Pri), domainIDToSet(0), perfLevelToSet(0) {}

DVFSHandler::DomainID
DVFSHandler::domainID(uint32_t index) const
{
    fatal_if(index >= numDomains(), "DVFS: Requested index out of bound, max value %d\n", (domainIDList.size() - 1));
    assert(domains.find(domainIDList[index]) != domains.end());
    return domainIDList[index];
}

bool
DVFSHandler::validDomainID(DomainID domain_id) const
{
    assert(isEnabled());
    if (domains.find(domain_id) != domains.end())
        return true;
    warn("DVFS: invalid domain ID %d, the DVFS handler does not handle this domain\n", domain_id);
    return false;
}



void
DVFSHandler::setPerformanceLevels(const std::vector<std::string>& levels)
{
    performanceLevels = levels;
    if (!levels.empty()) {
        std::string concatenatedLevels = std::accumulate(std::next(levels.begin()), levels.end(), levels[0],
            [](const std::string &a, const std::string &b) {
                return a + ", " + b;
            });
        DPRINTF(DVFS, "Performance levels set: %s\n", concatenatedLevels.c_str());
        std::cout << "First link established" << std::endl;
    } else {
        DPRINTF(DVFS, "Performance levels set: \n");
    }
}



void
DVFSHandler::setCoreIdsToMonitor(const std::vector<uint32_t>& core_ids)
{
    coreIdsToMonitor = core_ids;
    std::string coreIdsStr = std::accumulate(core_ids.begin(), core_ids.end(), std::string(),
        [](const std::string &a, uint32_t b) {
            return a.empty() ? std::to_string(b) : a + ", " + std::to_string(b);
        });
    DPRINTF(DVFS, "Core IDs to monitor set: %s\n", coreIdsStr.c_str());
}

// void
// DVFSHandler::monitorWorkload(uint32_t core_id, const std::string &workload)
// {
//     DPRINTF(DVFS, "Monitoring workload for core %d: %s\n", core_id, workload.c_str());
//     // Implement the logic to monitor workload for a specific core
// }


DVFSHandler::WorkloadType DVFSHandler::classifyWorkload(double ipc, double cacheMissRate, double fetchRate) {
    if (ipc < ipcThresholdLow && cacheMissRate > cacheMissRateThreshold) {
        return WorkloadType::MEMORY_BOUND;
    } else if (ipc > ipcThresholdHigh && fetchRate > fetchRateThreshold) {
        return WorkloadType::COMPUTE_BOUND;
    } else {
        return WorkloadType::MIXED;
    }
}

double DVFSHandler::estimateEnergyEfficiency(uint32_t coreId, PerfLevelEnum level) {
    double frequency = 1.0; // GHz
    double voltage = 1.0; // V
    switch (level) {
        case LOW:
            frequency = 0.8;
            voltage = 0.8;
            break;
        case MEDIUM:
            frequency = 1.2;
            voltage = 1.0;
            break;
        case HIGH:
            frequency = 1.8;
            voltage = 1.2;
            break;
    }

    double power = voltage * voltage * frequency; // Simplified power model
    double performance = frequency * coreStates[coreId].avgIPC;
    return performance / power;
}

void DVFSHandler::updateCoreState(uint32_t coreId, double ipc, double cacheMissRate, double fetchRate) {
    auto& state = coreStates[coreId];
    const double alpha = 0.8; // Exponential moving average factor
    state.avgIPC = alpha * state.avgIPC + (1 - alpha) * ipc;
    state.avgCacheMissRate = alpha * state.avgCacheMissRate + (1 - alpha) * cacheMissRate;
    state.avgFetchRate = alpha * state.avgFetchRate + (1 - alpha) * fetchRate;
}

void DVFSHandler::setConfigurableParameters(const DVFSConfig& config) {
    ipcThresholdLow = config.ipcThresholdLow;
    ipcThresholdHigh = config.ipcThresholdHigh;
    cacheMissRateThreshold = config.cacheMissRateThreshold;
    fetchRateThreshold = config.fetchRateThreshold;
    hysteresisThreshold = config.hysteresisThreshold;
}



void DVFSHandler::monitorWorkload(uint32_t core_ids, const std::string &workload)
{
    DPRINTF(DVFS, "Monitoring workload for core %d: %s\n", core_ids, workload.c_str());
    std::cout << "Link established for the cores." << std::endl;


    // Debugging: Print a message before opening the file
    std::cout << "Opening m5out/stats.txt for reading stats" << std::endl;
    std::cout.flush();

    // Read stats from m5out/stats.txt
    std::ifstream statsFile("m5out/stats.txt");
    if (!statsFile.is_open()) {
        warn("Could not open m5out/stats.txt for reading stats");
        return;
    }

    double ipc = 0.0;
    double cacheMissRate = 0.0;
    double fetchRate = 0.0;

    std::string line;
    while (std::getline(statsFile, line)) {
        std::istringstream iss(line);
        std::string key;
        double value;

        // Debugging: Print the line being read
        std::cout << "Reading line: " << line << std::endl;
        std::cout.flush();

        if (!(iss >> key >> value)) {
            std::cout << "Invalid line: " << line << std::endl;
            std::cout.flush();
            continue;  // Skip invalid lines
        }

        // Debugging: Print the value being read
        std::cout << "Zan la para e test kawam " << value << std::endl;
        std::cout.flush();

        if (key == "system.cpu" + std::to_string(core_ids) + ".ipc") {
            ipc = value;
            std::cout << "IPC for core " << core_ids << ": " << ipc << std::endl;
            std::cout.flush();
        } else if (key == "system.cpu" + std::to_string(core_ids) + ".dcache.overall_miss_rate::total") {
            cacheMissRate = value;
            std::cout << "Cache Miss Rate for core " << core_ids << ": " << cacheMissRate << std::endl;
            std::cout.flush();
        } else if (key == "system.cpu" + std::to_string(core_ids) + ".fetch.fetchRate") {
            fetchRate = value;
            std::cout << "Fetch Rate for core " << core_ids << ": " << fetchRate << std::endl;
            std::cout.flush();
        }
    }

// *********** THEAS Algorithm Implementation*********** //

    // Define thresholds
    const double IPC_THRESHOLD_LOW = 0.1;
    const double IPC_THRESHOLD_HIGH = 0.3;
    const double CACHE_MISS_RATE_THRESHOLD = 0.1;  // 10% miss rate
    const double FETCH_RATE_THRESHOLD = 1000000;  // 1M instructions/second

    updateCoreState(core_ids, ipc, cacheMissRate, fetchRate);
    WorkloadType workloadType = classifyWorkload(ipc, cacheMissRate, fetchRate);

    PerfLevelEnum newLevel = currentPerf_level;
    double currentEfficiency = estimateEnergyEfficiency(core_ids, currentPerf_level);

    if (workloadType == WorkloadType::COMPUTE_BOUND && ipc < ipcThresholdLow) {
        newLevel = static_cast<PerfLevelEnum>(std::min(static_cast<int>(newLevel) + 1, static_cast<int>(HIGH)));
        DPRINTF(DVFS, "Increasing performance level for core %d\n", core_ids);
    } else if (workloadType == WorkloadType::MEMORY_BOUND && cacheMissRate > cacheMissRateThreshold) {
        newLevel = static_cast<PerfLevelEnum>(std::max(static_cast<int>(newLevel) - 1, static_cast<int>(LOW)));
        DPRINTF(DVFS, "Decreasing performance level for core %d\n", core_ids);
    }

    double newEfficiency = estimateEnergyEfficiency(core_ids, newLevel);
    if (newEfficiency > currentEfficiency && ++coreStates[core_ids].stableCount >= hysteresisThreshold) {
        adjustDVFS(core_ids, newLevel);
        coreStates[core_ids].stableCount = 0;
    } else if (newLevel == currentPerf_level) {
        coreStates[core_ids].stableCount = 0;
    }

    DPRINTF(DVFS, "Core %d - IPC: %.2f, Cache Miss Rate: %.2f, Fetch Rate: %.2f, Workload: %s, Efficiency: %.2f\n",
            core_ids, ipc, cacheMissRate, fetchRate, static_cast<int>(workloadType), currentEfficiency);
}

void DVFSHandler::adjustDVFS(uint32_t core_ids, PerfLevelEnum newLevel)
{
    if (newLevel != currentPerf_level) {
        // Get the clock domain for the specified core
        SrcClockDomain *domain = findDomain(core_ids);

        // Set the new voltage and frequency
        switch (newLevel) {
            case LOW:
                // domain->clockPeriod(Tick(1000000000));  // 1 GHz
                domain->clockPeriod(Tick(1250000000));  // 0.8 GHz

                domain->voltageDomain()->voltage(0.8);
                DPRINTF(DVFS, "Set core %d to LOW performance level (1 GHz, 0.8V)\n", core_ids);
                break;
            case MEDIUM:
                // domain->clockPeriod(Tick(500000000));   // 2 GHz
                domain->clockPeriod(Tick(833333333));   // 1.2 GHz
                domain->voltageDomain()->voltage(1.0);
                DPRINTF(DVFS, "Set core %d to MEDIUM performance level (2 GHz, 1.0V)\n", core_ids);
                break;
            case HIGH:
                // domain->clockPeriod(Tick(333333333));   // 3 GHz
                domain->clockPeriod(Tick(555555556));   // 1.8 GHz
                domain->voltageDomain()->voltage(1.2);
                DPRINTF(DVFS, "Set core %d to HIGH performance level (3 GHz, 1.2V)\n", core_ids);
                break;
        }

        // Update the currentPerf_level
        currentPerf_level = newLevel;

        DPRINTF(DVFS, "Adjusted DVFS for core %d to performance level %d\n", core_ids, newLevel);

        DPRINTF(DVFS, "New energy efficiency for core %d: %.2f\n", core_ids, estimateEnergyEfficiency(core_ids, newLevel));
    }
}


std::vector<uint32_t>
DVFSHandler::determineNewCoresBasedOnMetrics()
{
    // Implement the logic to determine new cores based on metrics
    return std::vector<uint32_t>(); // Placeholder return value
}

bool
DVFSHandler::perfLevel(DomainID domain_id, PerfLevel perf_level)
{
    assert(isEnabled());
    DPRINTF(DVFS, "DVFS: setPerfLevel domain %d -> %d\n", domain_id, perf_level);
    auto d = findDomain(domain_id);
    if (!d->validPerfLevel(perf_level))
    {
        warn("DVFS: invalid performance level %d for domain ID %d, request ignored\n", perf_level, domain_id);
        return false;
    }

    UpdateEvent *update_event = &updatePerfLevelEvents[domain_id];
    if (update_event->scheduled())
    {
        DPRINTF(DVFS, "DVFS: Overwriting the previous DVFS event.\n");
        deschedule(update_event);
    }

    update_event->perfLevelToSet = perf_level;
    if (d->perfLevel() == perf_level)
    {
        DPRINTF(DVFS, "DVFS: Ignoring ineffective performance level change %d -> %d\n", d->perfLevel(), perf_level);
        return false;
    }

    Tick when = curTick() + _transLatency;
    DPRINTF(DVFS, "DVFS: Update for perf event scheduled for %ld\n", when);
    schedule(update_event, when);
    return true;
}

DVFSHandler::PerfLevel
DVFSHandler::perfLevel(DomainID domain_id) const
{
    assert(isEnabled());
    return findDomain(domain_id)->perfLevel();
}

void
DVFSHandler::UpdateEvent::updatePerfLevel()
{
    statistics::dump();
    statistics::reset();
    auto d = dvfsHandler->findDomain(domainIDToSet);
    assert(d->perfLevel() != perfLevelToSet);
    d->perfLevel(perfLevelToSet);
}

double
DVFSHandler::voltageAtPerfLevel(DomainID domain_id, PerfLevel perf_level) const
{
    VoltageDomain *d = findDomain(domain_id)->voltageDomain();
    assert(d);
    PerfLevel n = d->numVoltages();
    if (perf_level < n)
        return d->voltage(perf_level);
    if (n == 1)
    {
        DPRINTF(DVFS, "DVFS: Request for perf-level %i for single-point voltage domain %s. Returning voltage at level 0: %.2f V\n", perf_level, d->name(), d->voltage(0));
        return d->voltage(0);
    }

    warn("DVFSHandler %s reads illegal voltage level %u from VoltageDomain %s. Returning 0 V\n", name(), perf_level, d->name());
    return 0.;
}

SrcClockDomain *
DVFSHandler::findDomain(DomainID domain_id) const
{
    auto it = domains.find(domain_id);
    panic_if(it == domains.end(), "DVFS: Could not find a domain for ID %d.\n", domain_id);
    return it->second;
}

DVFSHandler::PerfLevel
DVFSHandler::numPerfLevels(PerfLevel domain_id) const
{
    return findDomain(static_cast<DomainID>(domain_id))->numPerfLevels();
}

Tick
DVFSHandler::clkPeriodAtPerfLevel(DomainID domain_id, PerfLevel perf_level) const
{
    SrcClockDomain *d = findDomain(domain_id);
    assert(d);
    PerfLevel n = d->numPerfLevels();
    if (perf_level < n)
        return d->clkPeriodAtPerfLevel(perf_level);
    warn("DVFSHandler %s reads illegal frequency level %u from SrcClockDomain %s. Returning 0\n", name(), perf_level, d->name());
    return Tick(0);
}

void
DVFSHandler::serialize(CheckpointOut &cp) const
{
    SERIALIZE_SCALAR(enableHandler);
    std::vector<DomainID> domain_ids;
    std::vector<PerfLevel> perf_levels;
    std::vector<Tick> whens;
    for (const auto &ev_pair : updatePerfLevelEvents)
    {
        DomainID id = ev_pair.first;
        const UpdateEvent *event = &ev_pair.second;
        assert(id == event->domainIDToSet);
        domain_ids.push_back(id);
        perf_levels.push_back(event->perfLevelToSet);
        whens.push_back(event->scheduled() ? event->when() : 0);
    }

    SERIALIZE_CONTAINER(domain_ids);
    SERIALIZE_CONTAINER(perf_levels);
    SERIALIZE_CONTAINER(whens);
}

void
DVFSHandler::unserialize(CheckpointIn &cp)
{
    bool temp = enableHandler;
    UNSERIALIZE_SCALAR(enableHandler);
    if (temp != enableHandler)
    {
        warn("DVFS: Forcing enable handler status to unserialized value of %d", enableHandler);
    }

    std::vector<DomainID> domain_ids;
    std::vector<PerfLevel> perf_levels;
    std::vector<Tick> whens;
    UNSERIALIZE_CONTAINER(domain_ids);
    UNSERIALIZE_CONTAINER(perf_levels);
    UNSERIALIZE_CONTAINER(whens);
    for (size_t i = 0; i < domain_ids.size(); ++i)
    {
        UpdateEvent *event = &updatePerfLevelEvents[domain_ids[i]];
        event->domainIDToSet = domain_ids[i];
        event->perfLevelToSet = perf_levels[i];
        if (whens[i])
            schedule(event, whens[i]);
    }

    UpdateEvent::dvfsHandler = this;
}

} // namespace gem5
