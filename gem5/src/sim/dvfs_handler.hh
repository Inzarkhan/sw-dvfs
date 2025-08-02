#ifndef __SIM_DVFS_HANDLER_HH__
#define __SIM_DVFS_HANDLER_HH__

#include <cassert>
#include <map>
#include <vector>
#include <pybind11/pybind11.h>
#include "base/logging.hh"
#include "base/types.hh"
#include "debug/DVFS.hh"
#include "params/DVFSHandler.hh"
#include "sim/clock_domain.hh"
#include "sim/eventq.hh"
#include "sim/sim_object.hh"

namespace gem5
{

class DVFSHandler : public SimObject
{
  public:
    typedef DVFSHandlerParams Params;
    DVFSHandler(const Params &p);

    typedef SrcClockDomain::DomainID DomainID;
    typedef SrcClockDomain::PerfLevel PerfLevel;

    uint32_t numDomains() const { return domainIDList.size(); }
    DomainID domainID(uint32_t index) const;
    bool validDomainID(DomainID domain_id) const;
    Tick transLatency() const { return _transLatency; }
    bool perfLevel(DomainID domain_id, PerfLevel perf_level);
    PerfLevel perfLevel(DomainID domain_id) const;
    Tick clkPeriodAtPerfLevel(DomainID domain_id, PerfLevel perf_level) const;
    double voltageAtPerfLevel(DomainID domain_id, PerfLevel perf_level) const;
    PerfLevel numPerfLevels(PerfLevel domain_id) const;
    bool isEnabled() const { return enableHandler; }

    void serialize(CheckpointOut &cp) const override;
    void unserialize(CheckpointIn &cp) override;

    // New methods
    enum PerfLevelEnum
    {
        LOW,
        MEDIUM,
        HIGH
    };

    void updateMonitoredCores();
    bool checkMetricsExceedThreshold();
    void setPerformanceLevels(const std::vector<std::string>& levels);
    void setCoreIdsToMonitor(const std::vector<uint32_t>& core_ids);
    void monitorWorkload(uint32_t core_ids, const std::string& workload);
    void adjustDVFS(uint32_t core_ids, PerfLevelEnum newLevel);


    // New update in the code  Nov13, 2024
    double ipcThresholdLow;
    double ipcThresholdHigh;
    double cacheMissRateThreshold;
    double fetchRateThreshold;
    int hysteresisThreshold;

    // --- ADAPTIVE DVFS MEMBERS ---
    enum class WorkloadType {
        COMPUTE_BOUND,
        MEMORY_BOUND,
        MIXED
    };

    struct CoreState {
        double avgIPC;
        double avgCacheMissRate;
        double avgFetchRate;
        int stableCount;
        Tick lastUpdateTick;
        WorkloadType currentWorkload;
        // Sliding window for IPC history
        std::deque<double> ipcHistory;
    };

    std::map<uint32_t, CoreState> coreStates;

    // Dynamic thresholds (computed at runtime)
    double dynamicIpcThresholdLow;
    double dynamicIpcThresholdHigh;

    // Sliding window size
    static const size_t IPC_HISTORY_WINDOW_SIZE = 50;
    static constexpr double IPC_PERCENTILE_LOW = 0.15;
    static constexpr double IPC_PERCENTILE_HIGH = 0.85;

    // --- ADAPTIVE DVFS METHODS ---
    WorkloadType classifyWorkload(double avgIpc, double avgCacheMissRate, double avgFetchRate);
    double estimateEnergyEfficiency(uint32_t coreId, PerfLevelEnum level);
    void updateCoreState(uint32_t coreId, double ipc, double cacheMissRate, double fetchRate);
    void updateDynamicThresholds(uint32_t coreId);


    bool enable;
    // std::vector<uint32_t> core_ids_to_monitor; // REMOVED: use coreIdsToMonitor only
    // std::vector<std::string> performance_levels;

  protected:
    bool enableHandler;
    std::vector<uint32_t> coreIdsToMonitor;
    std::vector<std::string> performanceLevels;
    SrcClockDomain* sysClkDomain;
    Tick _transLatency;

  private:
    typedef std::map<int, SrcClockDomain*> Domains;
    Domains domains;
    std::vector<DomainID> domainIDList;
    PerfLevelEnum currentPerf_level;


    SrcClockDomain *findDomain(DomainID domain_id) const;
    std::vector<uint32_t> determineNewCoresBasedOnMetrics();

    struct UpdateEvent : public Event
    {
        UpdateEvent();
        static DVFSHandler *dvfsHandler;
        DomainID domainIDToSet;
        PerfLevel perfLevelToSet;
        void updatePerfLevel();
        void process() { updatePerfLevel(); }
    };

    typedef std::map<DomainID, UpdateEvent> UpdatePerfLevelEvents;
    UpdatePerfLevelEvents updatePerfLevelEvents;
};

} // namespace gem5

#endif // __SIM_DVFS_HANDLER_HH__