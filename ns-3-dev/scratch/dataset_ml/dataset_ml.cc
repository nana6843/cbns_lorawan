/*
 * Copyright (c) 2017 University of Padova
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Davide Magrin <magrinda@dei.unipd.it>
 */

/*
 * This script simulates a complex scenario with multiple gateways and end
 * devices. The metric of interest for this script is the throughput of the
 * network.
 */

#include "ns3/building-allocator.h"
#include "ns3/building-penetration-loss.h"
#include "ns3/buildings-helper.h"
#include "ns3/class-a-end-device-lorawan-mac.h"
#include "ns3/command-line.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/correlated-shadowing-propagation-loss-model.h"
#include "ns3/double.h"
#include "ns3/end-device-lora-phy.h"
#include "ns3/forwarder-helper.h"
#include "ns3/gateway-lora-phy.h"
#include "ns3/gateway-lorawan-mac.h"
#include "ns3/log.h"
#include "ns3/lora-helper.h"
#include "ns3/mobility-helper.h"
#include "ns3/network-server-helper.h"
#include "ns3/node-container.h"
#include "ns3/periodic-sender-helper.h"
#include "ns3/pointer.h"
#include "ns3/position-allocator.h"
#include "ns3/random-variable-stream.h"
#include "ns3/simulator.h"

#include "ns3/lora-phy.h"
#include "ns3/lora-tag.h"


// ENERGY MODEL (WAJIB)
#include "ns3/basic-energy-source-helper.h"
#include "ns3/lora-radio-energy-model-helper.h"
#include "ns3/energy-source.h"


#include "ns3/lorawan-mac-header.h"
#include "ns3/lora-frame-header.h"
#include <map>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <algorithm>
#include <ctime>

using namespace ns3;
using namespace lorawan;

NS_LOG_COMPONENT_DEFINE("ComplexLorawanNetworkExample");


// Structure untuk simpan konfigurasi node
struct NodeExperimentConfig {
    uint8_t dr;
    double freq;
    double txPower;
};


// Function untuk baca experiment schedule dari CSV
std::map<uint32_t, NodeExperimentConfig> LoadExperimentConfig(uint32_t experimentId, std::string csvPath)
{
    std::map<uint32_t, NodeExperimentConfig> configs;
    std::ifstream file(csvPath);
    
    if (!file.is_open())
    {
        std::cerr << "ERROR: Cannot open " << csvPath << std::endl;
        std::cerr << "Please run generate_allocation.py first!" << std::endl;
        exit(1);
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;
        
        // Parse CSV: experiment,nodeId,combId,dr,freq,txPower
        uint32_t exp, nodeId, combId;
        NodeExperimentConfig config;
        
        std::getline(ss, cell, ','); exp = std::stoi(cell);
        
        // Only load data for THIS experiment
        if (exp != experimentId) continue;
        
        std::getline(ss, cell, ','); nodeId = std::stoi(cell);
        std::getline(ss, cell, ','); combId = std::stoi(cell); // Skip combId
        std::getline(ss, cell, ','); config.dr = std::stoi(cell);
        std::getline(ss, cell, ','); config.freq = std::stod(cell);
        std::getline(ss, cell, ','); config.txPower = std::stod(cell);
        
        configs[nodeId] = config;
    }
    
    file.close();
    std::cout << "✓ Loaded " << configs.size() << " node configurations for Experiment " << experimentId << std::endl;
    return configs;
}




// Load fixed positions from CSV
std::vector<Vector> LoadNodePositions(std::string csvPath, uint32_t numNodes)
{
    std::vector<Vector> positions;
    std::ifstream file(csvPath);
    
    if (!file.is_open())
    {
        std::cerr << "ERROR: Cannot open " << csvPath << std::endl;
        std::cerr << "Please run generate_positions.py first!" << std::endl;
        exit(1);
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;
        Vector pos;
        
        // Parse: nodeIndex,x,y,z,distance
        std::getline(ss, cell, ','); // nodeIndex (skip)
        std::getline(ss, cell, ','); pos.x = std::stod(cell);
        std::getline(ss, cell, ','); pos.y = std::stod(cell);
        std::getline(ss, cell, ','); pos.z = std::stod(cell);
        
        positions.push_back(pos);
    }
    
    file.close();
    
    if (positions.size() != numNodes)
    {
        std::cerr << "ERROR: Expected " << numNodes << " positions, got " << positions.size() << std::endl;
        exit(1);
    }
    
    std::cout << "✓ Loaded " << positions.size() << " fixed positions from " << csvPath << std::endl;
    return positions;
}





// Store node information for RSSI/SNR calculation
struct NodeInfo {
    uint32_t nodeId;
    Vector position;
    double distance;
    double txPower;
    uint8_t dataRate;
    double frequency;
};

std::map<LoraDeviceAddress, NodeInfo> addressToNodeInfo;



static std::map<uint32_t, double> totalTimeOnAir; // nodeId → total seconds
static std::map<uint32_t, uint32_t> txCount;      // nodeId → tx count

// --- Local minimal struct for TX parameters (if header missing) ---
struct MyLoraTxParameters
{
    uint8_t sf = 7;          // Spreading factor
    double bandwidthHz = 125000;  // 125 kHz typical
    uint8_t codingRate = 1;       // CR 4/5
    uint16_t preambleLength = 8;  // standard LoRa preamble
};

// Callback: called every time an ED transmits a packet
void StartSendingTrace(Ptr<const Packet> packet, uint32_t nodeId)
{
    if (packet == nullptr)
        return;

    Ptr<Packet> pktCopy = packet->Copy();

    // Try to get spreading factor from LoraTag
    LoraTag tag;
    uint8_t sf = 7;
    if (pktCopy->PeekPacketTag(tag))
    {
        sf = tag.GetSpreadingFactor();
    }

    // Build Tx parameters
    MyLoraTxParameters txParams;
    txParams.sf = sf;

    // Compute Time-On-Air
    //Time toa = LoraPhy::GetOnAirTime(pktCopy, txParams);

    // --- Compute Time-On-Air manually ---
    uint32_t payloadSize = pktCopy->GetSize(); // bytes
    double bw = txParams.bandwidthHz;
    //int sfInt = txParams.sf;
    int cr = txParams.codingRate;
    int preamble = txParams.preambleLength;
    bool lowDataRateOpt = (bw == 125000 && (sf == 11 || sf == 12));

    double tSym = pow(2.0, sf) / bw; // symbol duration (s)
    double tPreamble = (preamble + 4.25) * tSym;
    double payloadSymbNb = 8 + std::max(
        std::ceil((8.0 * payloadSize - 4.0 * sf + 28 + 16 - 20) /
                (4.0 * (sf - 2 * lowDataRateOpt)) * cr + 4),
        0.0);
    double tPayload = payloadSymbNb * tSym;
    double toaSeconds = tPreamble + tPayload;



  

    // Accumulate per-node totals
    totalTimeOnAir[nodeId] += toaSeconds;
    txCount[nodeId] += 1;

    // std::cout << "ED=" << nodeId
    //           << " | ToA=" << std::fixed << std::setprecision(2)
    //           << toaSeconds * 1000.0 << " ms" << std::endl;
}



// ========== ADD THESE 4 FUNCTIONS HERE ==========

// Function 1: Calculate RSSI from distance


// Function 3: Extract sender address from packet
LoraDeviceAddress GetSenderAddress(Ptr<const Packet> packet)
{
    Ptr<Packet> packetCopy = packet->Copy();
    
    LorawanMacHeader mHdr;
    packetCopy->RemoveHeader(mHdr);
    
    LoraFrameHeader fHdr;
    fHdr.SetAsUplink();
    packetCopy->RemoveHeader(fHdr);
    
    return fHdr.GetAddress();
}



// ========== END OF FUNCTIONS TO ADD ==========

//tambahan

// === Enhanced trace output ===
// void
// ReceivedPacketTrace(Ptr<const Packet> packet, uint32_t gwNodeId)
// {
//     if (packet == nullptr)
//     {
//         return;
//     }

//     // Use the LoraTag if present for RX power (safe)
//     LoraTag tag;
//     double rxPower = NAN;
//     bool hasTag = packet->PeekPacketTag(tag);
//     if (hasTag)
//     {
//         rxPower = tag.GetReceivePower();
//     }

//     // We intentionally DO NOT parse headers here to avoid Deserialize crashes.
//     // If sender info is unavailable, show ED=?.
//     std::ostringstream info;
//     info << std::fixed << std::setprecision(2);
//     info << Simulator::Now().GetSeconds()
//          << "s | GW=" << gwNodeId
//          << " | ED=?"
//          << " | RX=" << (hasTag ? std::to_string(rxPower) + " dBm" : "N/A")
//          << " | Size=" << packet->GetSize();

// }





void TimeOnAirTrace(double toaSeconds, uint32_t nodeId)
{
    // accumulate total seconds
    totalTimeOnAir[nodeId] += toaSeconds;

    // print per-transmission line (in ms)
    // std::cout << "ED=" << nodeId
    //           << " | ToA=" << std::fixed << std::setprecision(2)
    //           << (toaSeconds * 1000.0) << " ms" << std::endl;
}


// Experiment parameters
uint32_t experimentId = 1;  // Default experiment 1
std::string csvPath = "/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/experiment_schedule_lgbm_1000_5000.csv";  // Path to CSV


// Network settings
int nDevices = 1000;                 //!< Number of end device nodes to create
int nGateways = 1;                  //!< Number of gateway nodes to create
double radiusMeters = 5000;         //!< Radius (m) of the deployment
double simulationTimeSeconds = 60000; //!< Scenario duration (s) in simulated time

// Channel model
bool realisticChannelModel = false; //!< Whether to use a more realistic channel model with
                                    //!< Buildings and correlated shadowing

int appPeriodSeconds = 600; //!< Duration (s) of the inter-transmission time of end devices

// Output control
bool printBuildingInfo = true; //!< Whether to print building information




int
main(int argc, char* argv[])
{
    CommandLine cmd(__FILE__);
    cmd.AddValue("nDevices", "Number of end devices to include in the simulation", nDevices);
    cmd.AddValue("radius", "The radius (m) of the area to simulate", radiusMeters);
    cmd.AddValue("realisticChannel",
                 "Whether to use a more realistic channel model",
                 realisticChannelModel);
    cmd.AddValue("simulationTime", "The time (s) for which to simulate", simulationTimeSeconds);
    cmd.AddValue("appPeriod",
                 "The period in seconds to be used by periodically transmitting applications",
                 appPeriodSeconds);
    cmd.AddValue("print", "Whether or not to print building information", printBuildingInfo);

    //tambahan arguement
    cmd.AddValue("experiment", "Experiment ID (1-90)", experimentId);
    cmd.AddValue("csv", "Path to experiment_schedule.csv", csvPath);


    cmd.Parse(argc, argv);


    std::cout << "\n=== Loading Experiment " << experimentId << " Configuration ===" << std::endl;
    std::map<uint32_t, NodeExperimentConfig> experimentConfigs = LoadExperimentConfig(experimentId, csvPath);
    std::cout << "========================================\n" << std::endl;

    // Set up logging
    // LogComponentEnable("ComplexLorawanNetworkExample", LOG_LEVEL_ALL);
    // LogComponentEnable("LoraChannel", LOG_LEVEL_INFO);
    // LogComponentEnable("LoraPhy", LOG_LEVEL_ALL);
    // LogComponentEnable("EndDeviceLoraPhy", LOG_LEVEL_ALL);
    // LogComponentEnable("GatewayLoraPhy", LOG_LEVEL_ALL);
    // LogComponentEnable("LoraInterferenceHelper", LOG_LEVEL_ALL);
    // LogComponentEnable("LorawanMac", LOG_LEVEL_ALL);
    // LogComponentEnable("EndDeviceLorawanMac", LOG_LEVEL_ALL);
    // LogComponentEnable("ClassAEndDeviceLorawanMac", LOG_LEVEL_ALL);
    // LogComponentEnable("GatewayLorawanMac", LOG_LEVEL_ALL);
    // LogComponentEnable("LogicalLoraChannelHelper", LOG_LEVEL_ALL);
    // LogComponentEnable("LogicalLoraChannel", LOG_LEVEL_ALL);
    // LogComponentEnable("LoraHelper", LOG_LEVEL_ALL);
    // LogComponentEnable("LoraPhyHelper", LOG_LEVEL_ALL);
    // LogComponentEnable("LorawanMacHelper", LOG_LEVEL_ALL);
    // LogComponentEnable("PeriodicSenderHelper", LOG_LEVEL_ALL);
    // LogComponentEnable("PeriodicSender", LOG_LEVEL_ALL);
    // LogComponentEnable("LorawanMacHeader", LOG_LEVEL_ALL);
    // LogComponentEnable("LoraFrameHeader", LOG_LEVEL_ALL);
    // LogComponentEnable("NetworkScheduler", LOG_LEVEL_ALL);
    // LogComponentEnable("NetworkServer", LOG_LEVEL_ALL);
    // LogComponentEnable("NetworkStatus", LOG_LEVEL_ALL);
    // LogComponentEnable("NetworkController", LOG_LEVEL_ALL);

    /***********
     *  Setup  *
     ***********/

    // Create the time value from the period
    Time appPeriod = Seconds(appPeriodSeconds);

    // Mobility di ganti pake posisi tetap dari file .csv
    // MobilityHelper mobility;
    // mobility.SetPositionAllocator("ns3::UniformDiscPositionAllocator",
    //                               "rho",
    //                               DoubleValue(radiusMeters),
    //                               "X",
    //                               DoubleValue(0.0),
    //                               "Y",
    //                               DoubleValue(0.0));
    // mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");


    // Load FIXED positions from CSV

    std::string positionsCSV = "/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/node_positions_lgbm_1000_5000.csv";
    std::vector<Vector> fixedPositions = LoadNodePositions(positionsCSV, nDevices);
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> posAllocator = CreateObject<ListPositionAllocator>();

    for (uint32_t i = 0; i < nDevices; i++)
    {
        posAllocator->Add(fixedPositions[i]);
    }

    mobility.SetPositionAllocator(posAllocator);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    /************************
     *  Create the channel  *
     ************************/

    // Create the lora channel object
    Ptr<LogDistancePropagationLossModel> loss = CreateObject<LogDistancePropagationLossModel>();
    loss->SetPathLossExponent(3.76);
    loss->SetReference(1, 7.7);

    if (realisticChannelModel)
    {
        // Create the correlated shadowing component
        Ptr<CorrelatedShadowingPropagationLossModel> shadowing =
            CreateObject<CorrelatedShadowingPropagationLossModel>();

        // Aggregate shadowing to the logdistance loss
        loss->SetNext(shadowing);

        // Add the effect to the channel propagation loss
        Ptr<BuildingPenetrationLoss> buildingLoss = CreateObject<BuildingPenetrationLoss>();

        shadowing->SetNext(buildingLoss);
    }

    Ptr<PropagationDelayModel> delay = CreateObject<ConstantSpeedPropagationDelayModel>();

    Ptr<LoraChannel> channel = CreateObject<LoraChannel>(loss, delay);

    /************************
     *  Create the helpers  *
     ************************/

    // Create the LoraPhyHelper
    LoraPhyHelper phyHelper = LoraPhyHelper();
    phyHelper.SetChannel(channel);

    // Create the LorawanMacHelper
    LorawanMacHelper macHelper = LorawanMacHelper();

    // Create the LoraHelper
    LoraHelper helper = LoraHelper();
    helper.EnablePacketTracking(); // Output filename
    // helper.EnableSimulationTimePrinting ();

    // Create the NetworkServerHelper
    NetworkServerHelper nsHelper = NetworkServerHelper();

    // Create the ForwarderHelper
    ForwarderHelper forHelper = ForwarderHelper();

    /************************
     *  Create End Devices  *
     ************************/

    // Create a set of nodes
    NodeContainer endDevices;
    endDevices.Create(nDevices);

    // Assign a mobility model to each node
    mobility.Install(endDevices);

    // Make it so that nodes are at a certain height > 0
    for (auto j = endDevices.Begin(); j != endDevices.End(); ++j)
    {
        Ptr<MobilityModel> mobility = (*j)->GetObject<MobilityModel>();
        Vector position = mobility->GetPosition();
        //position.z = 0.0;
        mobility->SetPosition(position);
    }

    // Create the LoraNetDevices of the end devices
    uint8_t nwkId = 54;
    uint32_t nwkAddr = 1864;
    Ptr<LoraDeviceAddressGenerator> addrGen =
        CreateObject<LoraDeviceAddressGenerator>(nwkId, nwkAddr);

    // Create the LoraNetDevices of the end devices
    macHelper.SetAddressGenerator(addrGen);
    phyHelper.SetDeviceType(LoraPhyHelper::ED);
    macHelper.SetDeviceType(LorawanMacHelper::ED_A);
    helper.Install(phyHelper, macHelper, endDevices);

    // Now end devices are connected to the channel


    /************************
    * Install Energy Model *
    ************************/

    BasicEnergySourceHelper basicSourceHelper;
    LoraRadioEnergyModelHelper radioEnergyHelper;

    // Battery configuration
    basicSourceHelper.Set("BasicEnergySourceInitialEnergyJ", DoubleValue(20000.0));  // 20,000 Joule
    basicSourceHelper.Set("BasicEnergySupplyVoltageV", DoubleValue(3.3));

    // LoRa radio currents (SX1272 typical values)
    radioEnergyHelper.Set("StandbyCurrentA", DoubleValue(0.0014));
    radioEnergyHelper.Set("TxCurrentA", DoubleValue(0.028));
    radioEnergyHelper.Set("SleepCurrentA", DoubleValue(0.0000015));
    radioEnergyHelper.Set("RxCurrentA", DoubleValue(0.0112));

    radioEnergyHelper.SetTxCurrentModel(
        "ns3::ConstantLoraTxCurrentModel",
        "TxCurrent", DoubleValue(0.028));  // 28 mA TX

    // Step 1 - Install batteries to each ED
    EnergySourceContainer energySources = basicSourceHelper.Install(endDevices);

    // Step 2 - Convert NodeContainer → NetDeviceContainer
    NetDeviceContainer endDeviceNetDevices;
    for (uint32_t i = 0; i < endDevices.GetN(); i++)
    {
        endDeviceNetDevices.Add(endDevices.Get(i)->GetDevice(0));
    }

    // Step 3 - Install LoRa energy model
    DeviceEnergyModelContainer energyModels =
    radioEnergyHelper.Install(endDeviceNetDevices, energySources);



    /*********************
     *  Create Gateways  *
     *********************/

    // Create the gateway nodes (allocate them uniformly on the disc)
    NodeContainer gateways;
    gateways.Create(nGateways);

    Ptr<ListPositionAllocator> allocator = CreateObject<ListPositionAllocator>();
    // Make it so that nodes are at a certain height > 0
    allocator->Add(Vector(0.0, 0.0, 15.0));
    mobility.SetPositionAllocator(allocator);
    mobility.Install(gateways);

    // Create a netdevice for each gateway
    phyHelper.SetDeviceType(LoraPhyHelper::GW);
    macHelper.SetDeviceType(LorawanMacHelper::GW);
    helper.Install(phyHelper, macHelper, gateways);

    

    // // ---------- RSSI / SNR / Packet trace connections ----------
    // for (auto gwIt = gateways.Begin(); gwIt != gateways.End(); ++gwIt)
    // {
    //     Ptr<Node> gwNode = *gwIt;
    //     Ptr<NetDevice> dev = gwNode->GetDevice(0);
    //     Ptr<LoraNetDevice> loraDev = DynamicCast<LoraNetDevice>(dev);
    //     if (!loraDev)
    //         continue;

    //     Ptr<LoraPhy> phy = loraDev->GetPhy();
    //     if (!phy)
    //         continue;

    //     // Connect the traces to our C-style callbacks
    //     phy->TraceConnectWithoutContext("Rssi", MakeCallback(&RssiTrace));
    //     phy->TraceConnectWithoutContext("Snr", MakeCallback(&SnrTrace));
    //     phy->TraceConnectWithoutContext("ReceivedPacket", MakeCallback(&ReceivedPacketTrace));
    // }


    /**********************
     *  Handle buildings  *
     **********************/

    double xLength = 130;
    double deltaX = 32;
    double yLength = 64;
    double deltaY = 17;
    int gridWidth = 2 * radiusMeters / (xLength + deltaX);
    int gridHeight = 2 * radiusMeters / (yLength + deltaY);
    if (!realisticChannelModel)
    {
        gridWidth = 0;
        gridHeight = 0;
    }
    Ptr<GridBuildingAllocator> gridBuildingAllocator;
    gridBuildingAllocator = CreateObject<GridBuildingAllocator>();
    gridBuildingAllocator->SetAttribute("GridWidth", UintegerValue(gridWidth));
    gridBuildingAllocator->SetAttribute("LengthX", DoubleValue(xLength));
    gridBuildingAllocator->SetAttribute("LengthY", DoubleValue(yLength));
    gridBuildingAllocator->SetAttribute("DeltaX", DoubleValue(deltaX));
    gridBuildingAllocator->SetAttribute("DeltaY", DoubleValue(deltaY));
    gridBuildingAllocator->SetAttribute("Height", DoubleValue(6));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsX", UintegerValue(2));
    gridBuildingAllocator->SetBuildingAttribute("NRoomsY", UintegerValue(4));
    gridBuildingAllocator->SetBuildingAttribute("NFloors", UintegerValue(2));
    gridBuildingAllocator->SetAttribute(
        "MinX",
        DoubleValue(-gridWidth * (xLength + deltaX) / 2 + deltaX / 2));
    gridBuildingAllocator->SetAttribute(
        "MinY",
        DoubleValue(-gridHeight * (yLength + deltaY) / 2 + deltaY / 2));
    BuildingContainer bContainer = gridBuildingAllocator->Create(gridWidth * gridHeight);

    BuildingsHelper::Install(endDevices);
    BuildingsHelper::Install(gateways);

    // Print the buildings
    if (printBuildingInfo)
    {
        std::ofstream myfile;
        myfile.open("buildings.txt");
        std::vector<Ptr<Building>>::const_iterator it;
        int j = 1;
        for (it = bContainer.Begin(); it != bContainer.End(); ++it, ++j)
        {
            Box boundaries = (*it)->GetBoundaries();
            myfile << "set object " << j << " rect from " << boundaries.xMin << ","
                   << boundaries.yMin << " to " << boundaries.xMax << "," << boundaries.yMax
                   << std::endl;
        }
        myfile.close();
    }

    /**********************************************
     *  Set up the end device's spreading factor  *
     **********************************************/

    LorawanMacHelper::SetSpreadingFactorsUp(endDevices, gateways, channel);

    
    // =====================
    // Connect StartSending trace (safe, always available)
    // =====================
    uint32_t nodeIndex = 0;  // Track node index (bukan nodeId!)
    for (auto j = endDevices.Begin(); j != endDevices.End(); ++j)
    {
        Ptr<Node> node = *j;
        Ptr<LoraNetDevice> loraNetDevice = DynamicCast<LoraNetDevice>(node->GetDevice(0));
        if (!loraNetDevice) continue;
        Ptr<LoraPhy> phy = loraNetDevice->GetPhy();
        if (!phy) continue;

        // Connect StartSending trace (packet, nodeId)
        phy->TraceConnectWithoutContext("StartSending", MakeCallback(&StartSendingTrace));
    }


        // Connect trace sources
    for (auto j = endDevices.Begin(); j != endDevices.End(); ++j)
    {
        Ptr<Node> node = *j;
        Ptr<LoraNetDevice> loraNetDevice = DynamicCast<LoraNetDevice>(node->GetDevice(0));
        Ptr<LoraPhy> phy = loraNetDevice->GetPhy();

        // Get mobility (for position and distance)
        Ptr<MobilityModel> mob = node->GetObject<MobilityModel>();
        Vector pos = mob->GetPosition();
        
        

        // Example of setting Tx Power for each end device
        Ptr<LoraNetDevice> dev = node->GetDevice(0)->GetObject<LoraNetDevice>();
        Ptr<EndDeviceLorawanMac> mac = DynamicCast<EndDeviceLorawanMac>(dev->GetMac());

        uint32_t nodeId = node->GetId();
        

    // Assign transmission power if MAC is valid
        if (experimentConfigs.find(nodeIndex) == experimentConfigs.end())
        {
            std::cerr << "ERROR: No config found for node index " << nodeIndex << std::endl;
            nodeIndex++;
            continue;
        }

            NodeExperimentConfig config = experimentConfigs[nodeIndex];
    
        // ✅ APPLY CONFIGURATION FROM CSV
        mac->SetDataRate(config.dr);
        mac->SetCurrentFrequency(config.freq);
        mac->SetTransmissionPowerDbm(config.txPower);
        
        // Get applied values (for verification)
        uint8_t appliedDR = mac->GetDataRate();
        double appliedFreq = mac->GetCurrentFrequency();
        double appliedTxPower = mac->GetTransmissionPowerDbm();
        
        // Get device address
        LoraDeviceAddress address = mac->GetDeviceAddress();
        
        // Calculate distance from gateway (assumed at origin 0,0,15)
        Vector gwPos(0.0, 0.0, 15.0);
        double distance = std::sqrt(
            std::pow(pos.x - gwPos.x, 2) +
            std::pow(pos.y - gwPos.y, 2) +
            std::pow(pos.z - gwPos.z, 2)
        );
        
        // Store node information
        NodeInfo info;
        info.nodeId = nodeId;
        info.position = pos;
        info.distance = distance;
        info.txPower = appliedTxPower;
        info.dataRate = appliedDR;
        info.frequency = appliedFreq;
        
        addressToNodeInfo[address] = info;
        
        // Print configuration
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Node " << nodeIndex
                << " (ID:" << nodeId << ")"
                //<< " | Addr: " << address
                << " | Pos: (" << pos.x << ", " << pos.y << ", " << pos.z << ")"
                << " | Dist: " << distance << " m"
                << " | DR" << unsigned(appliedDR)
                << " | Freq: " << appliedFreq / 1e6 << " MHz"
                << " | TxPower: " << appliedTxPower << " dBm"
                << std::endl;
        
        // Connect trace sources
        phy->TraceConnectWithoutContext("StartSending", MakeCallback(&StartSendingTrace));
        
        nodeIndex++;  // Increment node index


    }






    NS_LOG_DEBUG("Completed configuration");

    /*********************************************
     *  Install applications on the end devices  *
     *********************************************/

    Time appStopTime = Seconds(simulationTimeSeconds);
    PeriodicSenderHelper appHelper = PeriodicSenderHelper();
    appHelper.SetPeriod(Seconds(appPeriodSeconds));
    appHelper.SetPacketSize(23);
    Ptr<RandomVariableStream> rv =
        CreateObjectWithAttributes<UniformRandomVariable>("Min",
                                                          DoubleValue(0),
                                                          "Max",
                                                          DoubleValue(10));
    ApplicationContainer appContainer = appHelper.Install(endDevices);

    appContainer.Start(Time(0));
    appContainer.Stop(appStopTime);

    /**************************
     *  Create network server  *
     ***************************/

    // Create the network server node
    Ptr<Node> networkServer = CreateObject<Node>();

    // PointToPoint links between gateways and server
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));
    // Store network server app registration details for later
    P2PGwRegistration_t gwRegistration;
    for (auto gw = gateways.Begin(); gw != gateways.End(); ++gw)
    {
        auto container = p2p.Install(networkServer, *gw);
        auto serverP2PNetDev = DynamicCast<PointToPointNetDevice>(container.Get(0));
        gwRegistration.emplace_back(serverP2PNetDev, *gw);
    }

    // Create a network server for the network
    nsHelper.SetGatewaysP2P(gwRegistration);
    nsHelper.SetEndDevices(endDevices);
    nsHelper.Install(networkServer);

    // Create a forwarder for each gateway
    forHelper.Install(gateways);


 

    ////////////////
    // Simulation //
    ////////////////
    std::cout << "\n=== Starting Simulation ===" << std::endl;
    std::cout << "Simulation time: " << simulationTimeSeconds << " seconds\n" << std::endl;

    Simulator::Stop(appStopTime + Hours(1));

    NS_LOG_INFO("Running simulation...");
    Simulator::Run();

    

    

    ///////////////////////////
    // Print results to file //
    ///////////////////////////
    NS_LOG_INFO("Computing performance metrics...");

    std::cout << "======== OVERALL STATISTIK (TOTAL TX TOTAL RX@Gateway) ======" << std::endl;
    LoraPacketTracker& tracker = helper.GetPacketTracker();
    std::cout << tracker.CountMacPacketsGlobally(Time(0), appStopTime + Hours(1)) << std::endl;

    


    Time start = Seconds(0);
    Time stop = appStopTime + Hours(1);

    std::cout << "\n=== Per-End Device Tx/Rx Summary ===" << std::endl;
    for (uint32_t i = 0; i < endDevices.GetN(); ++i)
    {
        Ptr<Node> node = endDevices.Get(i);
        uint32_t nodeId = node->GetId();

        uint32_t tx = tracker.CountMacPacketsPerDevice(nodeId, start, stop);
        uint32_t rx = tracker.CountReceivedPacketsPerDevice(nodeId, start, stop);

        double ratio = (tx > 0) ? (100.0 * rx / tx) : 0.0;

        // std::cout << "Node " << nodeId
        //         << " | TxPackets: " << tx
        //         << " | RRxPacketsAtGatewayx: " << rx
        //         << " | DeliveryRatio: " << std::fixed << std::setprecision(2)
        //         << ratio << "%" << std::endl;
    }
        std::cout << "====================================\n";

        std::cout << "\n=== Total Time-On-Air per ED ===" << std::endl;
        for (auto const& p : totalTimeOnAir)
        {
            uint32_t nodeId = p.first;
            double totalSeconds = p.second;
            uint32_t txs = txCount.count(nodeId) ? txCount[nodeId] : 0;

            // std::cout << "ED=" << nodeId
            //         << " | TotalToA=" << std::fixed << std::setprecision(2)
            //         << (totalSeconds * 1000.0) << " ms"
            //         << " | TxCount=" << txs
            //         << " | AvgToA=" << (txs ? (totalSeconds / txs * 1000.0) : 0.0)
            //         << " ms" << std::endl;
        }
        std::cout << "=================================" << std::endl;







        // ============================
        // Save results to CSV file
        // ============================
        
        std::stringstream filename_1;
        filename_1 << "/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/results_experiment_lgbm_1000_5000_" << experimentId << ".csv";

        std::ofstream csvFile(filename_1.str());
        csvFile << "node_id,x,y,z,distance,dr,freq,txPower,TxPackets,RxPackets,TotalToA_ms,AvgToA_ms,RemainingEnergy_J\n";
        csvFile << std::fixed << std::setprecision(6);


        for (uint32_t i = 0; i < endDevices.GetN(); ++i)
        {
            Ptr<Node> node = endDevices.Get(i);
            if (node->GetNDevices() == 0)
            {
                std::cerr << "⚠️  Node " << node->GetId() << " has no devices, skipping...\n";
                continue;
            }

            Ptr<LoraNetDevice> dev = node->GetDevice(0)->GetObject<LoraNetDevice>();
            if (!dev)
            {
                std::cerr << "⚠️  Node " << node->GetId() << " device cast failed, skipping...\n";
                continue;
            }

            Ptr<MobilityModel> mob = node->GetObject<MobilityModel>();
            Vector pos = mob->GetPosition();

            Ptr<EndDeviceLorawanMac> mac = DynamicCast<EndDeviceLorawanMac>(dev->GetMac());
            double txPower = 0.0;
            uint8_t dr = 0;
            double freq = 0.0;

            if (mac)
            {
                txPower = mac->GetTransmissionPowerDbm();
                dr = mac->GetDataRate();
                freq = mac->GetCurrentFrequency() / 1e6; // MHz
            }

            double distance = 0.0;
            for (auto const& pair : addressToNodeInfo)
            {
                if (pair.second.nodeId == node->GetId())
                {
                    distance = pair.second.distance;
                    break;
                }
            }

            uint32_t tx = tracker.CountMacPacketsPerDevice(node->GetId(), start, stop);
            uint32_t rx = tracker.CountReceivedPacketsPerDevice(node->GetId(), start, stop);
            double totalToA_ms = totalTimeOnAir.count(node->GetId())
                                    ? (totalTimeOnAir.at(node->GetId()) * 1000.0)
                                    : 0.0;
            uint32_t txCountVal = txCount.count(node->GetId()) ? txCount[node->GetId()] : 0;
            double avgToA_ms = (txCountVal > 0) ? (totalToA_ms / txCountVal) : 0.0;

            double remainingEnergy = 0.0;
            Ptr<EnergySource> es = energySources.Get(i);
            if (es)
                remainingEnergy = es->GetRemainingEnergy();

            csvFile << node->GetId() << ","
                    << pos.x << "," << pos.y << "," << pos.z << ","
                    << distance << ","
                    << static_cast<int>(dr) << ","
                    << freq << ","
                    << txPower << ","
                    << tx << ","
                    << rx << ","
                    << totalToA_ms << ","
                    << avgToA_ms << ","
                    << remainingEnergy
                    << "\n";
        }

        csvFile.close();
        std::cout << "✅ Results saved" << std::endl;


    
        Simulator::Destroy();




    return 0;
}