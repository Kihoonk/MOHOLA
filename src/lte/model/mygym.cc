/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Technische Universität Berlin
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

#include "mygym.h"

#include "ns3/object.h"

#include "ns3/core-module.h"

#include <ns3/lte-module.h>

#include "ns3/node-list.h"

#include "ns3/log.h"

#include <sstream>

#include <iostream>

#include <ns3/lte-common.h>
#include "ns3/lte-enb-rrc.h"

#include <ns3/cell-individual-offset.h>

#include <stdlib.h>

#include <typeinfo>

#include <numeric>

#include <cmath>

#include <fstream>

namespace ns3 {

    NS_LOG_COMPONENT_DEFINE("MyGymEnv");

    NS_OBJECT_ENSURE_REGISTERED(MyGymEnv);

    MyGymEnv::MyGymEnv() {
        NS_LOG_FUNCTION(this);
    }

    MyGymEnv::MyGymEnv(double stepTime, uint32_t N1, uint32_t N2, uint16_t N3, uint32_t port) {
        NS_LOG_FUNCTION(this);
        collect = 0;
        collecting_window = 0.05; //50ms
        block_Thr = 0.5; // Blockage threshold 0.5 Mb/s
        m_chooseReward = 0; //0: average overall throughput, 1: PRBs utilization deviation, 2: number of blocked users
        m_interval = stepTime;
        m_cellCount = N1;
        m_userCount = N2;
        m_nRBTotal = N3 * collecting_window * 1000;
        m_port = port;
        m_rbUtil.assign(m_cellCount, 0);
        m_dlThroughput = 0;
        m_cellFrequency.assign(m_cellCount, 0);
        rewards.assign(m_cellCount, 0);
        rewards_sum.assign(m_cellCount, 0);
        m_dlThroughputVec.assign(m_cellCount, 0);
        m_UesNum.assign(m_cellCount, 0);
        std::vector < uint32_t > dummyVec(29, 0);
        m_MCSPen.assign(m_cellCount, dummyVec);
        Simulator::Schedule(Seconds(1), & MyGymEnv::ScheduleNextStateRead, this);
        Simulator::Schedule(Seconds(1 - collecting_window), & MyGymEnv::Start_Collecting, this);
        UserThrouput.clear();
        Currentaveragerbutil.assign(m_cellCount, 0.0);
        pre_averagerbutil.assign(m_cellCount, 0.0);
    }

    void
    MyGymEnv::ScheduleNextStateRead() {
        NS_LOG_FUNCTION(this);
        Simulator::Schedule(Seconds(m_interval), & MyGymEnv::ScheduleNextStateRead, this);
        Notify();
    }
    void
    MyGymEnv::Start_Collecting() {
        NS_LOG_FUNCTION(this);
        collect = 1;
        NS_LOG_LOGIC("%%%%%%%% Start collecting %%%%%%%%  time= " << Simulator::Now().GetSeconds() << " sec");

    }
    MyGymEnv::~MyGymEnv() {
        NS_LOG_FUNCTION(this);
    }

    TypeId
    MyGymEnv::GetTypeId(void) {
        static TypeId tid = TypeId("MyGymEnv")
            .SetParent < OpenGymEnv > ()
            .SetGroupName("OpenGym")
            .AddConstructor < MyGymEnv > ();
        return tid;
    }

    void
    MyGymEnv::DoDispose() {
        NS_LOG_FUNCTION(this);
    }
    
    void 
    MyGymEnv::GetRlcStats(Ptr<RadioBearerStatsCalculator> m_rlcStats) {
        RlcStats = m_rlcStats;
    }
    
    void 
    MyGymEnv::AddNewNode(uint16_t cellId, Ptr<LteEnbNetDevice> dev){
        m_enbs.insert(std::pair<uint32_t, Ptr<LteEnbNetDevice>> (cellId, dev));
    }
    void 
    MyGymEnv::AddNewUe(uint64_t imsi, Ptr<LteUeNetDevice> dev){
        m_ues.insert(std::pair<uint64_t, Ptr<LteUeNetDevice>> (imsi, dev));
    }
    
    Ptr < OpenGymSpace >
        MyGymEnv::GetActionSpace() {
         
            // Custom
            ///////////////////////////
            uint32_t nodeNum = m_enbs.size();
            std::vector<uint32_t> shape {nodeNum,};
            std::vector<uint32_t> shape2 {m_userCount,};
            std::string dtype = TypeNameGet<float> ();

            Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (-6, 6, shape, dtype); //todo range change for -30 to 30
            

            NS_LOG_INFO("GetObservationSapce: "<<space);
            ///////////////////////////

            return space;
        }

    Ptr < OpenGymSpace >
        MyGymEnv::GetObservationSpace() {
            NS_LOG_FUNCTION(this);

            // Custom
            /////////////////////////////
            float low = -50.0;
            float high = 50.0;
            std::vector < uint32_t > shape = {m_cellCount,};
            // std::vector < uint32_t > shape = { m_cellCount,};
            std::string dtype = TypeNameGet < float > ();
            Ptr < OpenGymBoxSpace > rbUtil = CreateObject < OpenGymBoxSpace > (low, high, shape, dtype);
            Ptr < OpenGymBoxSpace > FarUes = CreateObject < OpenGymBoxSpace > (low, high, shape, dtype);
            Ptr < OpenGymBoxSpace > dlPrbusage = CreateObject < OpenGymBoxSpace > (0, 1, shape, dtype);
            Ptr < OpenGymBoxSpace > AverageVelocity = CreateObject < OpenGymBoxSpace > (0, 15, shape, dtype);

            Ptr < OpenGymBoxSpace > AvgCqi = CreateObject < OpenGymBoxSpace > (0, 15, shape, dtype);
            Ptr < OpenGymBoxSpace > TotalCqi = CreateObject < OpenGymBoxSpace > (0, 50, shape, dtype);

            Ptr < OpenGymBoxSpace > MLBreward = CreateObject < OpenGymBoxSpace > (low, high, shape, dtype); 
            Ptr < OpenGymBoxSpace > MROreward = CreateObject < OpenGymBoxSpace > (low, high, shape, dtype);


            // shape = {2,};
            Ptr < OpenGymBoxSpace > Results = CreateObject < OpenGymBoxSpace > (-100000, 100000, shape, dtype);


            Ptr<OpenGymDictSpace> space = CreateObject<OpenGymDictSpace> ();
            space -> Add("rbUtil", rbUtil);
            space -> Add("FarUes", FarUes);

            space -> Add("AvgCqi", AvgCqi);
            space -> Add("TotalCqi",TotalCqi);

            space -> Add("MROreward",MROreward);
            space -> Add("AverageVelcity",AverageVelocity);
            space -> Add("MLBreward", MLBreward);

            return space;


        }

    bool
    MyGymEnv::GetGameOver() {
        NS_LOG_FUNCTION(this);
        bool isGameOver = false;
        NS_LOG_LOGIC("MyGetGameOver: " << isGameOver);
        return isGameOver;
    }

    Ptr < OpenGymDataContainer >
MyGymEnv::GetObservation() {
    NS_LOG_FUNCTION(this);
    // Custom
    /////////////////////////////
    calculate_rewards();

    std::cout << "***********************************************************************************************" << std::endl;
    std::cout << "*************************************  [STEP " << step << "]  *************************************" << std::endl;
    std::cout << "***********************************************************************************************" << std::endl;
    step++;

    Ptr<OpenGymDictContainer> obsContainer = CreateObject<OpenGymDictContainer>();
    std::vector<uint32_t> shape = {m_cellCount,};

    std::vector<float> Far_UEs;
    std::vector<float> Served_UEs;
    std::vector<float> normal_RB;

    Ptr<OpenGymBoxContainer<float>> box = CreateObject<OpenGymBoxContainer<float>>(shape);
    Ptr<OpenGymBoxContainer<double>> box2 = CreateObject<OpenGymBoxContainer<double>>(shape);

    std::cout << "******" << std::endl;

    // ###################### PRB utilization Ui(t)#########################

    std::map<uint8_t, float> enbStepRbUtil;
    float normalizedRBUtil;
    for (uint8_t idx = 0; idx < m_cellCount; ++idx) {
        normalizedRBUtil = float(m_rbUtil.at(idx)) / m_nRBTotal; // 2

        std::cout << "m_nRBTotal : " << m_nRBTotal << std::endl;
        std::cout << "float(m_rbUtil.at(idx)) : " << float(m_rbUtil.at(idx)) << std::endl;

        if (step == 1) {
            TotalPRB.push_back(float(m_rbUtil.at(idx)));
        } else {
            TotalPRB[idx] += float(m_rbUtil.at(idx));
        }

        normal_RB.push_back(normalizedRBUtil);
        box->AddValue(normalizedRBUtil);
        // To see normalized RB util
        enbStepRbUtil[idx + 1] = (float(m_rbUtil.at(idx)) / m_nRBTotal);
        std::cout << "eNB " << (idx + 1) << ": " << "Normalized RB Utilization : " << (float(m_rbUtil.at(idx)) / m_nRBTotal) << std::endl;
    }
    obsContainer->Add("rbUtil", box);

    uint32_t enbIdx = 1;
    for (std::map<uint32_t, Ptr<LteEnbNetDevice>>::iterator iter = m_enbs.begin(); iter != m_enbs.end(); ++iter) {
        std::cout << "eNB " << enbIdx << ": " << iter->second->GetRrc()->m_numofues << " UEs " << std::endl;
        enbIdx++;
    }

    // ###################### Edge ratio Ei(t) (far_ue) #########################

    std::map<uint16_t, double> distanceMap;
    double TotalFarUes = 0;

for (std::map<uint32_t, Ptr<LteEnbNetDevice>>::iterator iter = m_enbs.begin(); iter != m_enbs.end(); ++iter) {
   // double cellid = iter->first;
    std::vector<uint64_t> Imsi_List;
    double eNB_x = iter->second->GetNode()->GetObject<MobilityModel>()->GetPosition().x;
    double eNB_y = iter->second->GetNode()->GetObject<MobilityModel>()->GetPosition().y;
    double eNB_z = iter->second->GetNode()->GetObject<MobilityModel>()->GetPosition().z;

    std::map<uint16_t, Ptr<UeManager>> UeMap = iter->second->GetRrc()->m_ueMap;

    for (std::map<uint16_t, Ptr<UeManager>>::iterator iter2 = UeMap.begin(); iter2 != UeMap.end(); ++iter2) {
        uint64_t Imsi = iter2->second->GetImsi();
        Imsi_List.push_back(Imsi);
    }

    double distance_sum = 0;
    double far_distance_ues = 0;
    for (uint64_t i = 0; i < Imsi_List.size(); i++) {
        Ptr<LteUeNetDevice> UeNetDevice = m_ues[Imsi_List[i]];
        double Ue_x = UeNetDevice->GetNode()->GetObject<MobilityModel>()->GetPosition().x;
        double Ue_y = UeNetDevice->GetNode()->GetObject<MobilityModel>()->GetPosition().y;
        double Ue_z = UeNetDevice->GetNode()->GetObject<MobilityModel>()->GetPosition().z;

        double distance = sqrt(pow(eNB_x - Ue_x, 2) + pow(eNB_y - Ue_y, 2) + pow(eNB_z - Ue_z, 2));
        distance_sum += distance;

        if (distance >= 70) {
            far_distance_ues += 1;
            TotalFarUes += 1;
        }
        // else if ((distance >= 30) & (cellid >= 4)) {
        //     far_distance_ues += 1;
        //     TotalFarUes += 1;
        // }

        distanceMap[Imsi_List[i]] = distance;
    }

    float ratio_far_ues;
    float ratio_ues;
    float served_ues;

    if (Imsi_List.size() == 0) {
        ratio_far_ues = 0;
        ratio_ues = 0;
        served_ues = 0;
    } else {
        ratio_far_ues = static_cast<float>(far_distance_ues) / static_cast<float>(Imsi_List.size());
        ratio_ues = static_cast<float>(Imsi_List.size()) / static_cast<float>(m_userCount);
        served_ues = static_cast<float>(Imsi_List.size());
    }
    ratio_ues = round(ratio_ues * 100) / 100;
    //ratio_far_ues = round(ratio_far_ues * 100) / 100;
    std::cout << "ratio of far_ues : " << ratio_far_ues << std::endl;
    Served_UEs.push_back(served_ues);
    Far_UEs.push_back(ratio_far_ues);
}
    // operation controller state2 Edge ratio Ei(t)
    box = CreateObject<OpenGymBoxContainer<float>>(shape);
    box->SetData(Far_UEs);
    obsContainer->Add("FarUes", box);

    // ###################### Paper MDP state 1  Load state ρi(t) ) #########################

    std::map<uint32_t, double> enbMLBindicator;
    std::map<uint32_t, double> pre_enbMLBindicator;
    for (std::map<uint32_t, Ptr<LteEnbNetDevice>>::iterator iter = m_enbs.begin(); iter != m_enbs.end(); ++iter) {
        uint32_t CellId = iter->first;
        if (step == 0)
            pre_enbMLBindicator[CellId] = 0;
        else
            pre_enbMLBindicator[CellId] = enbMLBindicator[CellId];

        double MLBindicator = 0;

        if (enbStepRbUtil[CellId] < 0.3) {
            if (Far_UEs[CellId] < 0.3)
                MLBindicator = 0;
            else if (0.3 <= Far_UEs[CellId] && Far_UEs[CellId] < 0.6)
                MLBindicator = 1;
            else if (0.6 <= Far_UEs[CellId])
                MLBindicator = 2;
        } else if (0.3 <= enbStepRbUtil[CellId] && enbStepRbUtil[CellId] < 0.6) {
            if (Far_UEs[CellId] < 0.3)
                MLBindicator = 3;
            else if (0.3 <= Far_UEs[CellId] && Far_UEs[CellId] < 0.6)
                MLBindicator = 4;
            else if (0.6 <= Far_UEs[CellId])
                MLBindicator = 5;
        } else if (0.6 <= enbStepRbUtil[CellId]) {
            if (Far_UEs[CellId] < 0.3)
                MLBindicator = 6;
            else if (0.3 <= Far_UEs[CellId] && Far_UEs[CellId] < 0.6)
                MLBindicator = 7;
            else if (0.6 <= Far_UEs[CellId])
                MLBindicator = 8;
        }

        enbMLBindicator[CellId] = MLBindicator;
        std::cout << "MLBindicator " << CellId << ": " << MLBindicator << std::endl;
        box2->AddValue(MLBindicator);
    }

    obsContainer->Add("enbMLBindicator", box2);

    // ###################### Paper MDP state 2 Load comparasion indicator Li(t)) #########################
    box2 = CreateObject<OpenGymBoxContainer<double>>(shape);

    std::map<uint32_t, double> enbneigborMLBindicator;
    for (std::map<uint32_t, Ptr<LteEnbNetDevice>>::iterator iter = m_enbs.begin(); iter != m_enbs.end(); ++iter) {
        uint32_t CellId = iter->first;

        double neigborMLBindicator = 0;
        if (CellId == 0) {
            if (enbMLBindicator[5] < enbMLBindicator[CellId] && enbMLBindicator[6] < enbMLBindicator[CellId])
                neigborMLBindicator = 0;
            else
                neigborMLBindicator = 1;
        } else if (CellId == 1) {
            if (enbMLBindicator[9] < enbMLBindicator[CellId] && enbMLBindicator[10] < enbMLBindicator[CellId])
                neigborMLBindicator = 0;
            else
                neigborMLBindicator = 1;
        } else if (CellId == 2) {
            if (enbMLBindicator[5] < enbMLBindicator[CellId] && enbMLBindicator[6] < enbMLBindicator[CellId] && enbMLBindicator[7] < enbMLBindicator[CellId] &&
                enbMLBindicator[8] < enbMLBindicator[CellId] && enbMLBindicator[9] < enbMLBindicator[CellId] && enbMLBindicator[10] < enbMLBindicator[CellId] &&
                enbMLBindicator[11] < enbMLBindicator[CellId] && enbMLBindicator[12] < enbMLBindicator[CellId])
                neigborMLBindicator = 0;
            else
                neigborMLBindicator = 1;
        } else if (CellId == 3) {
            if (enbMLBindicator[11] < enbMLBindicator[CellId] && enbMLBindicator[12] < enbMLBindicator[CellId])
                neigborMLBindicator = 0;
            else
                neigborMLBindicator = 1;
        } else if (CellId == 4) {
            if (enbMLBindicator[7] < enbMLBindicator[CellId] && enbMLBindicator[8] < enbMLBindicator[CellId])
                neigborMLBindicator = 0;
            else
                neigborMLBindicator = 1;
        } else if (CellId == 5) {
            if (enbMLBindicator[6] < enbMLBindicator[CellId] && enbMLBindicator[2] < enbMLBindicator[CellId] &&
                enbMLBindicator[9] < enbMLBindicator[CellId] && enbMLBindicator[0] < enbMLBindicator[CellId])
                neigborMLBindicator = 0;
            else
                neigborMLBindicator = 1;
        } else if (CellId == 6) {
            if (enbMLBindicator[5] < enbMLBindicator[CellId] && enbMLBindicator[2] < enbMLBindicator[CellId] &&
                enbMLBindicator[0] < enbMLBindicator[CellId] && enbMLBindicator[11] < enbMLBindicator[CellId])
                neigborMLBindicator = 0;
            else
                neigborMLBindicator = 1;
        } else if (CellId == 7) {
            if (enbMLBindicator[8] < enbMLBindicator[CellId] && enbMLBindicator[2] < enbMLBindicator[CellId] &&
                enbMLBindicator[10] < enbMLBindicator[CellId] && enbMLBindicator[4] < enbMLBindicator[CellId])
                neigborMLBindicator = 0;
            else
                neigborMLBindicator = 1;
        } else if (CellId == 8) {
            if (enbMLBindicator[7] < enbMLBindicator[CellId] && enbMLBindicator[2] < enbMLBindicator[CellId] &&
                enbMLBindicator[12] < enbMLBindicator[CellId] && enbMLBindicator[4] < enbMLBindicator[CellId])
                neigborMLBindicator = 0;
            else
                neigborMLBindicator = 1;
        } else if (CellId == 9) {
            if (enbMLBindicator[10] < enbMLBindicator[CellId] && enbMLBindicator[1] < enbMLBindicator[CellId] &&
                enbMLBindicator[5] < enbMLBindicator[CellId] && enbMLBindicator[2] < enbMLBindicator[CellId])
                neigborMLBindicator = 0;
            else
                neigborMLBindicator = 1;
        } else if (CellId == 10) {
            if (enbMLBindicator[9] < enbMLBindicator[CellId] && enbMLBindicator[1] < enbMLBindicator[CellId] &&
                enbMLBindicator[2] < enbMLBindicator[CellId] && enbMLBindicator[7] < enbMLBindicator[CellId])
                neigborMLBindicator = 0;
            else
                neigborMLBindicator = 1;
        } else if (CellId == 11) {
            if (enbMLBindicator[12] < enbMLBindicator[CellId] && enbMLBindicator[3] < enbMLBindicator[CellId] &&
                enbMLBindicator[2] < enbMLBindicator[CellId] && enbMLBindicator[6] < enbMLBindicator[CellId])
                neigborMLBindicator = 0;
            else
                neigborMLBindicator = 1;
        } else if (CellId == 12) {
            if (enbMLBindicator[11] < enbMLBindicator[CellId] && enbMLBindicator[3] < enbMLBindicator[CellId] &&
                enbMLBindicator[2] < enbMLBindicator[CellId] && enbMLBindicator[8] < enbMLBindicator[CellId])
                neigborMLBindicator = 0;
            else
                neigborMLBindicator = 1;
        }
        enbneigborMLBindicator[CellId] = neigborMLBindicator;
        std::cout << "neigborMLBindicator " << CellId << " : " << neigborMLBindicator << std::endl; // paper state 2
        box2->AddValue(neigborMLBindicator);
    }
    obsContainer->Add("enbneigborMLBindicator", box2);

    // ###################### Paper MDP state 3 RSRP indicator Ri(t) #########################

    // edge UEs (far_UEs), Served UEs

    std::vector<uint16_t> RSRPIndicator;

    double TotalFarUesRatio = double(TotalFarUes / m_userCount);
    std::cout << "Total edge Ues Ratio : " << TotalFarUesRatio << std::endl;

    // For Step RLF, PP in eNB
    std::map<uint32_t, int> eNBStepRlf;
    std::map<uint32_t, int> eNBStepPp;

    std::map<uint32_t, float> eNBStepMroReward;
    float TotalMroReward = 0;
    box2 = CreateObject<OpenGymBoxContainer<double>>(shape);

    std::map<uint32_t, int> stepearly;
    std::map<uint32_t, int> steplate;
    std::map<uint32_t, int> stepwrong;
    int stepearly1;
    int steplate1;
    int stepwrong1;
    for (std::map<uint32_t, Ptr<LteEnbNetDevice>>::iterator iter = m_enbs.begin(); iter != m_enbs.end(); ++iter) {
        std::vector<uint64_t> Imsi_List;
        uint32_t CellId = iter->first;
        std::cout << "CellId : " << CellId << std::endl;
        std::map<uint16_t, Ptr<UeManager>> UeMap = iter->second->GetRrc()->m_ueMap;

        for (std::map<uint16_t, Ptr<UeManager>>::iterator iter2 = UeMap.begin(); iter2 != UeMap.end(); ++iter2) {
            uint64_t Imsi = iter2->second->GetImsi();
            std::cout << "Imsi : " << Imsi << std::endl;
            Imsi_List.push_back(Imsi);
        }

        int stepRlf = 0;
        int stepPp = 0;
        float stepReward = 0;
        uint16_t cellrsrpindicatr = 0;
        stepearly1 = 0;
        steplate1 = 0;
        stepwrong1 = 0;

        for (uint64_t i = 0; i < Imsi_List.size(); i++) {
            Ptr<LteUeNetDevice> UeNetDevice = m_ues[Imsi_List[i]];
            std::cout << "imsi size" << Imsi_List.size() << std::endl;
            // Step RLF
            int Counter1 = UeNetDevice->GetPhy()->GetTooLateHO_CNT();
            int Counter2 = UeNetDevice->GetPhy()->GetTooEarlyHO_CNT();
            int Counter3 = UeNetDevice->GetPhy()->GetWrongCellHO_CNT();
            int Counter5 = UeNetDevice->GetPhy()->GetRSRPIndicator();

            stepRlf = stepRlf + (Counter1 + Counter2 + Counter3);

            if (Counter5 == 0)
                cellrsrpindicatr += 1;
            // Step PP
            int Counter4 = UeNetDevice->GetPhy()->GetPingPong_CNT();

            stepPp = stepPp + Counter4;

            stepearly1 += Counter2;
            steplate1 += Counter1;
            stepwrong1 += Counter3;
            stepReward = stepReward - (0.3 * Counter4 + 0.1 * Counter3 + 0.2 * Counter2 + 0.5 * Counter1);
        }

        eNBStepRlf[CellId] = stepRlf;
        eNBStepPp[CellId] = stepPp;
        eNBStepMroReward[CellId] = stepReward;
        stepearly[CellId] = stepearly1;
        steplate[CellId] = steplate1;
        stepwrong[CellId] = stepwrong1;

        TotalMroReward = TotalMroReward - stepReward;
        if (cellrsrpindicatr >= 1)
            cellrsrpindicatr = 1;
        RSRPIndicator.push_back(cellrsrpindicatr);
        box2->AddValue(cellrsrpindicatr);
    }

    obsContainer->Add("RSRPIndicator", box2);

    // ###################### Paper MDP state 4 Average CQI indicator Ci(t) #########################

    box = CreateObject<OpenGymBoxContainer<float>>(shape);

    std::vector<float> TotalCqi(m_cellCount);
    std::vector<float> AverageCqi(m_cellCount);
    std::vector<float> CQIcompareidicator(m_cellCount);

    float CqiSum = 0;

    std::map<uint32_t, float> enbStepCqi;
    for (std::map<uint32_t, Ptr<LteEnbNetDevice>>::iterator iter = m_enbs.begin(); iter != m_enbs.end(); ++iter) {
        uint32_t CellId = iter->first;
        std::vector<uint64_t> Imsi_List;

        std::map<uint16_t, Ptr<UeManager>> UeMap = iter->second->GetRrc()->m_ueMap;

        for (std::map<uint16_t, Ptr<UeManager>>::iterator iter2 = UeMap.begin(); iter2 != UeMap.end(); ++iter2) {
            uint64_t Imsi = iter2->second->GetImsi();
            Imsi_List.push_back(Imsi);
        }

        float AvgCqi = 0;
        for (uint64_t i = 0; i < Imsi_List.size(); i++) {
            float UeCqi = float(m_ues[Imsi_List[i]]->GetPhy()->AvgCqi);
            AvgCqi += UeCqi;
        }

        if (Imsi_List.size() == 0) {
            AvgCqi = 0;
            CqiSum = 0;
        } else {
            CqiSum += AvgCqi;
            AvgCqi = AvgCqi / Imsi_List.size();
        }

        AvgCqi = round(AvgCqi * 100) / 100;

        enbStepCqi[CellId] = AvgCqi;

        box->AddValue(AvgCqi);
    }

    obsContainer->Add("AvgCqi", box);

    box = CreateObject<OpenGymBoxContainer<float>>(shape);

    CqiSum = CqiSum / m_ues.size();
    for (uint8_t idx = 0; idx < m_cellCount; ++idx) {
        TotalCqi[idx] = CqiSum;
    }
    box->SetData(TotalCqi);
    obsContainer->Add("TotalCqi", box);

    box2 = CreateObject<OpenGymBoxContainer<double>>(shape);

    for (std::map<uint32_t, Ptr<LteEnbNetDevice>>::iterator iter = m_enbs.begin(); iter != m_enbs.end(); ++iter) {
        uint32_t CellId = iter->first;

        double CQIidicator = 0;
        if (CellId == 0) {
            CQIidicator = (AverageCqi[5] + AverageCqi[6]) / 2;
        } else if (CellId == 1) {
            CQIidicator = (AverageCqi[9] + AverageCqi[10]) / 2;
        } else if (CellId == 2) {
            CQIidicator = (AverageCqi[5] + AverageCqi[6] + AverageCqi[7] + AverageCqi[8] + AverageCqi[9] + AverageCqi[10] + AverageCqi[11] + AverageCqi[12]) / 8;
        } else if (CellId == 3) {
            CQIidicator = (AverageCqi[11] + AverageCqi[12]) / 2;
        } else if (CellId == 4) {
            CQIidicator = (AverageCqi[7] + AverageCqi[8]) / 2;
        } else if (CellId == 5) {
            CQIidicator = (AverageCqi[6] + AverageCqi[2] + AverageCqi[9] + AverageCqi[0]) / 4;
        } else if (CellId == 6) {
            CQIidicator = (AverageCqi[5] + AverageCqi[2] + AverageCqi[0] + AverageCqi[11]) / 4;
        } else if (CellId == 7) {
            CQIidicator = (AverageCqi[8] + AverageCqi[2] + AverageCqi[10] + AverageCqi[4]) / 4;
        } else if (CellId == 8) {
            CQIidicator = (AverageCqi[7] + AverageCqi[2] + AverageCqi[12] + AverageCqi[4]) / 4;
        } else if (CellId == 9) {
            CQIidicator = (AverageCqi[10] + AverageCqi[1] + AverageCqi[5] + AverageCqi[2]) / 4;
        } else if (CellId == 10) {
            CQIidicator = (AverageCqi[9] + AverageCqi[1] + AverageCqi[2] + AverageCqi[7]) / 4;
        } else if (CellId == 11) {
            CQIidicator = (AverageCqi[12] + AverageCqi[3] + AverageCqi[2] + AverageCqi[6]) / 4;
        } else if (CellId == 12) {
            CQIidicator = (AverageCqi[11] + AverageCqi[3] + AverageCqi[2] + AverageCqi[8]) / 4;
        }

        if (CQIidicator <= AverageCqi[CellId])
            CQIcompareidicator[CellId] = 0;
        else
            CQIcompareidicator[CellId] = 1;

        box2->AddValue(CQIcompareidicator[CellId]);
    }
    obsContainer->Add("CQIcompareidicator", box2);

    // ###################### Paper reward1 (rHO) #########################

    int Case1_Counter = 0;
    int Case2_Counter = 0;
    int Case3_Counter = 0;
    int Case4_Counter = 0;

    double ThroughputSum = 0;
    int CurrentRlfNum = 0;
    int CurrentPpNum = 0;

    for (std::map<uint64_t, Ptr<LteUeNetDevice>>::iterator iter = m_ues.begin(); iter != m_ues.end(); ++iter) {
        // Step RLF
        int Counter1 = iter->second->GetPhy()->GetTooLateHO_CNT();
        int Counter2 = iter->second->GetPhy()->GetTooEarlyHO_CNT();
        int Counter3 = iter->second->GetPhy()->GetWrongCellHO_CNT();

        // Step PP
        int Counter4 = iter->second->GetPhy()->GetPingPong_CNT();//kihoon

        uint32_t dlThroughput = dlThroughput_IMSI[iter->first];

        RLF_Counter += Counter1 + Counter2 + Counter3;
        Pingpong_Counter += Counter4;

        iter->second->GetPhy()->ClearTooLateHO_CNT();
        iter->second->GetPhy()->ClearTooEarlyHO_CNT();
        iter->second->GetPhy()->ClearWrongCellHO_CNT();
        iter->second->GetPhy()->ClearPingPong_CNT();

        Case1_Counter += Counter1;
        Case2_Counter += Counter2;
        Case3_Counter += Counter3;
        Case4_Counter += Counter4;

        double UeThroughput = (double(dlThroughput) * 8.0 / 1000000.0);

        ThroughputSum = ThroughputSum + UeThroughput;
        CurrentRlfNum = CurrentRlfNum + Counter1 + Counter2 + Counter3;
        CurrentPpNum = CurrentPpNum + Counter4;
    }

    double RlfRate;
    double PpRate;
    if (CurrentRlfNum == 0) {
        RlfRate = 0;
    } else {
        RlfRate = double(CurrentRlfNum) / 10;
    }
    if (CurrentPpNum == 0) {
        PpRate = 0;
    } else {
        PpRate = double(CurrentPpNum) / 10;
    }

    double AverageThroughput = ThroughputSum / m_userCount;

    double Reward = -(0.4 * RlfRate + 0.2 * PpRate);

    Reward *= 10000;
    Reward = round(Reward);
    Reward /= 10000;

    std::cout << "RLF Rate: " << RlfRate << "  PingPong Rate: " << PpRate << "  Average Throughput: " << AverageThroughput << std::endl;
    std::cout << "Reward: " << Reward << std::endl;

    box = CreateObject<OpenGymBoxContainer<float>>(shape);
    box->AddValue(Reward);
    obsContainer->Add("Reward", box);
    box2 = CreateObject<OpenGymBoxContainer<double>>(shape);
    std::vector<double> MroRewardVector;
    for (std::map<uint32_t, float>::iterator iter = eNBStepMroReward.begin(); iter != eNBStepMroReward.end(); ++iter) {
        double MroReward = iter->second;
        MroRewardVector.push_back(MroReward);
    }
    box2->SetData(MroRewardVector);
    obsContainer->Add("MROreward", box2);

    std::cout << "MRO Case1 Counter: " << Case1_Counter << std::endl;
    std::cout << "MRO Case2 Counter: " << Case2_Counter << std::endl;
    std::cout << "MRO Case3 Counter: " << Case3_Counter << std::endl;
    std::cout << "pingpong Counter: " << Case4_Counter << std::endl;//kihoon
    std::cout << "Total MRO Case: " << RLF_Counter << std::endl;
    std::cout << "Total pingpong Case: " << Pingpong_Counter << std::endl;
    std::cout << "**************************************" << std::endl;

    // ###################### Paper reward2 (rLoad) #########################

    Ptr<OpenGymBoxContainer<double>> box3 = CreateObject<OpenGymBoxContainer<double>>(shape);

    for (std::map<uint32_t, Ptr<LteEnbNetDevice>>::iterator iter = m_enbs.begin(); iter != m_enbs.end(); ++iter) {
        uint32_t CellId = iter->first;
        CellId = CellId - 1;
        double MLBreward = 0;
        double varianceSum = 0;
        double TotalvarianceSum = 0;
        double stdDev = 0;
        std::vector<float> selectedValues;
        std::vector<float> TotalValues;

        pre_averagerbutil[CellId] = Currentaveragerbutil[CellId];

        if (CellId == 0) {
            selectedValues = {enbStepRbUtil[CellId], enbStepRbUtil[5], enbStepRbUtil[6]};
        } else if (CellId == 1) {
            selectedValues = {enbStepRbUtil[CellId], enbStepRbUtil[9], enbStepRbUtil[10]};
        } else if (CellId == 2) {
            selectedValues = {enbStepRbUtil[CellId], enbStepRbUtil[5], enbStepRbUtil[6], enbStepRbUtil[7], enbStepRbUtil[8], enbStepRbUtil[9], enbStepRbUtil[10], enbStepRbUtil[11], enbStepRbUtil[12]};
        } else if (CellId == 3) {
            selectedValues = {enbStepRbUtil[CellId], enbStepRbUtil[11], enbStepRbUtil[12]};
        } else if (CellId == 4) {
            selectedValues = {enbStepRbUtil[CellId], enbStepRbUtil[7], enbStepRbUtil[8]};
        } else if (CellId == 5) {
            selectedValues = {enbStepRbUtil[CellId], enbStepRbUtil[6], enbStepRbUtil[2], enbStepRbUtil[9], enbStepRbUtil[0]};
        } else if (CellId == 6) {
            selectedValues = {enbStepRbUtil[CellId], enbStepRbUtil[5], enbStepRbUtil[2], enbStepRbUtil[0], enbStepRbUtil[11]};
        } else if (CellId == 7) {
            selectedValues = {enbStepRbUtil[CellId], enbStepRbUtil[8], enbStepRbUtil[2], enbStepRbUtil[10], enbStepRbUtil[4]};
        } else if (CellId == 8) {
            selectedValues = {enbStepRbUtil[CellId], enbStepRbUtil[7], enbStepRbUtil[2], enbStepRbUtil[12], enbStepRbUtil[4]};
        } else if (CellId == 9) {
            selectedValues = {enbStepRbUtil[CellId], enbStepRbUtil[10], enbStepRbUtil[1], enbStepRbUtil[5], enbStepRbUtil[2]};
        } else if (CellId == 10) {
            selectedValues = {enbStepRbUtil[CellId], enbStepRbUtil[9], enbStepRbUtil[1], enbStepRbUtil[2], enbStepRbUtil[7]};
        } else if (CellId == 11) {
            selectedValues = {enbStepRbUtil[CellId], enbStepRbUtil[12], enbStepRbUtil[3], enbStepRbUtil[2], enbStepRbUtil[6]};
        } else if (CellId == 12) {
            selectedValues = {enbStepRbUtil[CellId], enbStepRbUtil[11], enbStepRbUtil[3], enbStepRbUtil[2], enbStepRbUtil[8]};
        }
        double sum = std::accumulate(selectedValues.begin(), selectedValues.end(), 0.0);
        double mean = sum / selectedValues.size();

        for (double value : selectedValues) {
            varianceSum += std::pow(value - mean, 2);
        }
        stdDev = std::sqrt(varianceSum / selectedValues.size());

        if (CellId == 12) {
            TotalValues = {enbStepRbUtil[CellId], enbStepRbUtil[0], enbStepRbUtil[1], enbStepRbUtil[2], enbStepRbUtil[3], enbStepRbUtil[4], enbStepRbUtil[5], enbStepRbUtil[6], enbStepRbUtil[7], enbStepRbUtil[8], enbStepRbUtil[9], enbStepRbUtil[10], enbStepRbUtil[11]};
            double totalsum = std::accumulate(TotalValues.begin(), TotalValues.end(), 0.0);
            double totlamean = totalsum / TotalValues.size();

            for (double Totalvalue : TotalValues) {
                TotalvarianceSum += std::pow(Totalvalue - totlamean, 2);
            }
        }

        Currentaveragerbutil[CellId] = stdDev;

        MLBreward = pre_averagerbutil[CellId] - Currentaveragerbutil[CellId];

        box3->AddValue(MLBreward);
        std::cout << "MLBreward : " << MLBreward << std::endl;
    }
    obsContainer->Add("MLBreward", box3);

    // ###################### Paper operation controller state 1 (Average velocity Vi(t)) #########################

    box = CreateObject<OpenGymBoxContainer<float>>(shape);

    for (std::map<uint32_t, Ptr<LteEnbNetDevice>>::iterator iter = m_enbs.begin(); iter != m_enbs.end(); ++iter) {
        std::vector<uint64_t> Imsi_List;

        std::map<uint16_t, Ptr<UeManager>> UeMap = iter->second->GetRrc()->m_ueMap;

        for (std::map<uint16_t, Ptr<UeManager>>::iterator iter2 = UeMap.begin(); iter2 != UeMap.end(); ++iter2) {
            uint64_t Imsi = iter2->second->GetImsi();
            Imsi_List.push_back(Imsi);
        }

        double SumVelocity = 0.0;
        double NumofUe = Imsi_List.size();
        for (uint64_t i = 0; i < Imsi_List.size(); i++) {
            Ptr<LteUeNetDevice> UeNetDevice = m_ues[Imsi_List[i]];

            double velocity_x = UeNetDevice->GetNode()->GetObject<MobilityModel>()->GetVelocity().x;
            double velocity_y = UeNetDevice->GetNode()->GetObject<MobilityModel>()->GetVelocity().y;
            double velocity_z = UeNetDevice->GetNode()->GetObject<MobilityModel>()->GetVelocity().z;

            double velocity = sqrt(pow(velocity_x, 2) + pow(velocity_y, 2) + pow(velocity_z, 2));

            SumVelocity = SumVelocity + velocity;
        }

        double AverageVelocity;
        if (Imsi_List.size() == 0) {
            AverageVelocity = 0;
        } else {
            AverageVelocity = SumVelocity / NumofUe;
        }

        std::cout << "Cell " << iter->first << "  Average Velocity: " << AverageVelocity << std::endl;

        box->AddValue(AverageVelocity);
    }
    obsContainer->Add("AverageVelocity", box);

    return obsContainer;
}

    void
    MyGymEnv::resetObs() {
        NS_LOG_FUNCTION(this);
        m_rbUtil.assign(m_cellCount, 0);
        rewards.assign(m_cellCount, 0);
        rewards_sum.assign(m_cellCount, 0);
        m_cellFrequency.assign(m_cellCount, 0);
        m_dlThroughputVec.assign(m_cellCount, 0);
        m_dlThroughput = 0;
        UserThrouput.clear();
        m_UesNum.assign(m_cellCount, 0);
        std::vector < uint32_t > dummyVec(29, 0);
        m_MCSPen.assign(m_cellCount, dummyVec);
        NS_LOG_LOGIC("%%%%%%%% Stop collecting %%%%%%%%  time= " << Simulator::Now().GetSeconds() << " sec");
        collect = 0;
        Simulator::Schedule(Seconds(m_interval - collecting_window), & MyGymEnv::Start_Collecting, this);
    }

    void
    MyGymEnv::calculate_rewards() {
        std::map < uint16_t, std::map < uint16_t, float >> ::iterator itr;
       

        std::map < uint16_t, float > ::iterator ptr;
        int Blocked_Users_num = 0;
        double min_thro = 100;
	    m_UesNum.assign(m_cellCount, 0);
        float sum_reward=0 ;
        for (itr = UserThrouput.begin(); itr != UserThrouput.end(); itr++) {
        // std::cout<<"XXXXX"<<std::endl;
            float all = 0;
            std::map < uint16_t, float > tempmap = itr -> second;
            m_UesNum.at(itr -> first - 1) = tempmap.size();
            
            // std::cout<<"UE Num : "<<m_UesNum.at(itr->first-1)<<std::endl;

            NS_LOG_LOGIC("Cell: " << itr -> first << " total throuput: " << m_dlThroughputVec.at(itr -> first - 1));
            std::cout<<"Cell: " << itr -> first << " total throuput: " << m_dlThroughputVec.at(itr -> first - 1)<<std::endl;
            NS_LOG_LOGIC("#users : " << tempmap.size());
            for (ptr = itr -> second.begin(); ptr != itr -> second.end(); ptr++) {
                if (ptr -> second < block_Thr)
                    Blocked_Users_num++;
                if (ptr -> second < min_thro)
                    min_thro = ptr -> second;
                NS_LOG_LOGIC("rnti: " << ptr -> first << " throughput:  " << ptr -> second);
                all = all + ptr -> second;
            }

            NS_LOG_LOGIC("using sum Cell: " << itr -> first << " total throuput: " << all);
            rewards.at(itr -> first - 1) = all;
            sum_reward += all;
            // rewards.at(itr -> first - 1) = normalized_throughput;
            // std::cout<<"Cell ID : "<<itr->first<<"  "<<"all : "<<all<<std::endl;
            // std::cout<<"Cell ID : "<<itr->first<<"  "<<"Total Throughput : "<<sum_reward<<std::endl;

        }
        for (itr = UserThrouput.begin(); itr != UserThrouput.end(); itr++) {
            rewards_sum.at(itr->first - 1) = sum_reward;
        }
    }

    float
    MyGymEnv::GetReward() {
        NS_LOG_FUNCTION(this);

        float reward = 0;
        // reward = rewards.at(0);
        // if (m_chooseReward == 0) {
        //     reward = rewards.at(0);
        // } else if (m_chooseReward == 1) {
        //     reward = rewards.at(1);
        // } else if (m_chooseReward == 2) {
        //     reward = rewards.at(2);
        // } else {
        //     NS_LOG_ERROR("m_chooseReward variable should be between 0-2");
        // }
        resetObs();
        // NS_LOG_LOGIC("MyGetReward: " << reward);
        return reward;
    }

    std::string
    MyGymEnv::GetExtraInfo() {
        NS_LOG_FUNCTION(this);
        return "";
    }

    bool
    MyGymEnv::ExecuteActions(Ptr < OpenGymDataContainer > action) {
        
        
        Ptr<OpenGymBoxContainer<float> > box = DynamicCast<OpenGymBoxContainer<float>>(action);
        std::list<LteRrcSap::CellsToAddMod> celllist_temp;

            
            uint32_t nodeNum = m_enbs.size();
            for(uint32_t i = 0; i < nodeNum; i++){
                LteRrcSap::CellsToAddMod cell_temp;
                cell_temp.cellIndex = i+1;
                cell_temp.physCellId = 0;
  
                int8_t cio = box->GetValue(i);
                cell_temp.cellIndividualOffset = cio;
                std::cout<<"Cell "<<i+1<<"  new CIO: "<<int(cio)<<std::endl;
                celllist_temp.push_back(cell_temp);
            }

            for (std::map<uint32_t, Ptr<LteEnbNetDevice>>::iterator iter = m_enbs.begin(); iter != m_enbs.end(); ++iter){
                iter->second->m_rrc->setCellstoAddModList(celllist_temp);
                std::map<uint16_t, Ptr<UeManager>> m_UeMap = iter->second->GetRrc()->m_ueMap;
                for(auto iter2 = m_UeMap.begin(); iter2 != m_UeMap.end(); iter2++)
                {
                    iter2->second->ScheduleRrcConnectionRecursive();
                }
            }
    
            std::vector<double> UeActions (120);

            for (std::map<uint32_t, Ptr<LteEnbNetDevice>>::iterator iter = m_enbs.begin(); iter != m_enbs.end(); ++iter){
                uint32_t cellid = (iter->first)-1;


                double HOM = box->GetValue(13+2*cellid);
                uint16_t TTT = box->GetValue(14+2*cellid);

                    
                std::vector<uint64_t> Imsi_List;

                std::map<uint16_t, Ptr<UeManager>> UeMap = iter->second->GetRrc ()->m_ueMap;
                    
                for (std::map<uint16_t, Ptr<UeManager>>::iterator iter2 = UeMap.begin(); iter2 != UeMap.end(); ++iter2){
                    uint64_t Imsi = iter2->second->GetImsi();
                    
                    Ptr<LteUeNetDevice> UeNetDevice = m_ues[Imsi];

                    double velocity_x = UeNetDevice->GetNode()->GetObject<MobilityModel>()->GetVelocity().x;
                    double velocity_y = UeNetDevice->GetNode()->GetObject<MobilityModel>()->GetVelocity().y;
                    double velocity_z = UeNetDevice->GetNode()->GetObject<MobilityModel>()->GetVelocity().z;

                    double velocity = sqrt( pow(velocity_x, 2) + pow(velocity_y, 2) + pow(velocity_z, 2) );
                    double mappedVelocity;
                    // The part that gives each UE a detailed additional TTT setup 
                    //(borrows the idea of QMRO :ref  S. S. Mwanje et al., ``Distributed Cooperative Q-Learning for Mobility Sensitive Handover Optimization in LTE SON,'' {in \it Proc. IEEE ISCC 2014}, Jun 2014.)
                        if(velocity <=20.0){
                        mappedVelocity = 1.0;
                        }
                        else if(velocity > 20.0 && velocity <= 30.0){
                            mappedVelocity = 0.75;
                        }
                        else if(velocity > 30.0 && velocity <= 50.0){
                            mappedVelocity = 0.50;
                        }
                        else{
                            mappedVelocity = 0.25;
                        }
                    


                    uint64_t Imsi_actions = Imsi -1;
                    std::cout<<"Cellid : "<<cellid <<"HoM : " <<HOM<< "TTT : "<<TTT<<std::endl;
                    UeActions[2*Imsi_actions] = HOM;
                    UeActions[2*Imsi_actions+1] = TTT*mappedVelocity;
                
                }
            }

            for (std::map<uint64_t, Ptr<LteUeNetDevice>>::iterator iter = m_ues.begin(); iter != m_ues.end(); ++iter)
            {
                uint64_t imsi = (iter->first)-1;
                double HOM = UeActions[2*imsi];
                uint16_t TTT = UeActions[2*imsi+1];
                

                std::cout<< "UE:: "<<imsi<<"  New TTT: "<<TTT<<"  New HOM:  "<<HOM<<std::endl;

                iter->second->GetRrc()->ChangeTtt(TTT);
                iter->second->GetRrc()->ChangeHom(HOM);
            }
            /////////////////

        



  
        return true;
    }

    uint8_t
    MyGymEnv::Convert2ITB(uint8_t mcsIdx) {
        uint8_t iTBS;
        if (mcsIdx < 10) {
            iTBS = mcsIdx;
        } else if (mcsIdx < 17) {
            iTBS = mcsIdx - 1;
        } else {
            iTBS = mcsIdx - 2;
        }

        return iTBS;
    }

    uint8_t
    MyGymEnv::GetnRB(uint8_t iTB, uint16_t tbSize) {
        uint32_t tbSizeb = uint32_t(tbSize) * uint32_t(8);
       
        // search in list
    uint32_t tbList[]  = {
			  16,32,56,88,120,152,176,208,224,256,288,328,344,376,392,424,456,488,504,536,568,600,616,648,680,712,744,776,776,808,840,872,904,936,968,1000,1032,1032,1064,1096,1128,1160,1192,1224,1256,1256,1288,1320,1352,1384,1416,1416,1480,1480,1544,1544,1608,1608,1608,1672,1672,1736,1736,1800,1800,1800,1864,1864,1928,1928,1992,1992,2024,2088,2088,2088,2152,2152,2216,2216,2280,2280,2280,2344,2344,2408,2408,2472,2472,2536,2536,2536,2600,2600,2664,2664,2728,2728,2728,2792,2792,2856,2856,2856,2984,2984,2984,2984,2984,3112,
		24,56,88,144,176,208,224,256,328,344,376,424,456,488,520,568,600,632,680,712,744,776,808,872,904,936,968,1000,1032,1064,1128,1160,1192,1224,1256,1288,1352,1384,1416,1416,1480,1544,1544,1608,1608,1672,1736,1736,1800,1800,1864,1864,1928,1992,1992,2024,2088,2088,2152,2152,2216,2280,2280,2344,2344,2408,2472,2472,2536,2536,2600,2600,2664,2728,2728,2792,2792,2856,2856,2856,2984,2984,2984,3112,3112,3112,3240,3240,3240,3240,3368,3368,3368,3496,3496,3496,3496,3624,3624,3624,3752,3752,3752,3752,3880,3880,3880,4008,4008,4008,
		32,72,144,176,208,256,296,328,376,424,472,520,568,616,648,696,744,776,840,872,936,968,1000,1064,1096,1160,1192,1256,1288,1320,1384,1416,1480,1544,1544,1608,1672,1672,1736,1800,1800,1864,1928,1992,2024,2088,2088,2152,2216,2216,2280,2344,2344,2408,2472,2536,2536,2600,2664,2664,2728,2792,2856,2856,2856,2984,2984,3112,3112,3112,3240,3240,3240,3368,3368,3368,3496,3496,3496,3624,3624,3624,3752,3752,3880,3880,3880,4008,4008,4008,4136,4136,4136,4264,4264,4264,4392,4392,4392,4584,4584,4584,4584,4584,4776,4776,4776,4776,4968,4968,
		40,104,176,208,256,328,392,440,504,568,616,680,744,808,872,904,968,1032,1096,1160,1224,1256,1320,1384,1416,1480,1544,1608,1672,1736,1800,1864,1928,1992,2024,2088,2152,2216,2280,2344,2408,2472,2536,2536,2600,2664,2728,2792,2856,2856,2984,2984,3112,3112,3240,3240,3368,3368,3496,3496,3624,3624,3624,3752,3752,3880,3880,4008,4008,4136,4136,4264,4264,4392,4392,4392,4584,4584,4584,4776,4776,4776,4776,4968,4968,4968,5160,5160,5160,5352,5352,5352,5352,5544,5544,5544,5736,5736,5736,5736,5992,5992,5992,5992,6200,6200,6200,6200,6456,6456,
		56,120,208,256,328,408,488,552,632,696,776,840,904,1000,1064,1128,1192,1288,1352,1416,1480,1544,1608,1736,1800,1864,1928,1992,2088,2152,2216,2280,2344,2408,2472,2600,2664,2728,2792,2856,2984,2984,3112,3112,3240,3240,3368,3496,3496,3624,3624,3752,3752,3880,4008,4008,4136,4136,4264,4264,4392,4392,4584,4584,4584,4776,4776,4968,4968,4968,5160,5160,5160,5352,5352,5544,5544,5544,5736,5736,5736,5992,5992,5992,5992,6200,6200,6200,6456,6456,6456,6456,6712,6712,6712,6968,6968,6968,6968,7224,7224,7224,7480,7480,7480,7480,7736,7736,7736,7992,
		72,144,224,328,424,504,600,680,776,872,968,1032,1128,1224,1320,1384,1480,1544,1672,1736,1864,1928,2024,2088,2216,2280,2344,2472,2536,2664,2728,2792,2856,2984,3112,3112,3240,3368,3496,3496,3624,3752,3752,3880,4008,4008,4136,4264,4392,4392,4584,4584,4776,4776,4776,4968,4968,5160,5160,5352,5352,5544,5544,5736,5736,5736,5992,5992,5992,6200,6200,6200,6456,6456,6712,6712,6712,6968,6968,6968,7224,7224,7224,7480,7480,7480,7736,7736,7736,7992,7992,7992,8248,8248,8248,8504,8504,8760,8760,8760,8760,9144,9144,9144,9144,9528,9528,9528,9528,9528,
		328,176,256,392,504,600,712,808,936,1032,1128,1224,1352,1480,1544,1672,1736,1864,1992,2088,2216,2280,2408,2472,2600,2728,2792,2984,2984,3112,3240,3368,3496,3496,3624,3752,3880,4008,4136,4136,4264,4392,4584,4584,4776,4776,4968,4968,5160,5160,5352,5352,5544,5736,5736,5992,5992,5992,6200,6200,6456,6456,6456,6712,6712,6968,6968,6968,7224,7224,7480,7480,7736,7736,7736,7992,7992,8248,8248,8248,8504,8504,8760,8760,8760,9144,9144,9144,9144,9528,9528,9528,9528,9912,9912,9912,10296,10296,10296,10296,10680,10680,10680,10680,11064,11064,11064,11448,11448,11448,
		104,224,328,472,584,712,840,968,1096,1224,1320,1480,1608,1672,1800,1928,2088,2216,2344,2472,2536,2664,2792,2984,3112,3240,3368,3368,3496,3624,3752,3880,4008,4136,4264,4392,4584,4584,4776,4968,4968,5160,5352,5352,5544,5736,5736,5992,5992,6200,6200,6456,6456,6712,6712,6712,6968,6968,7224,7224,7480,7480,7736,7736,7992,7992,8248,8248,8504,8504,8760,8760,8760,9144,9144,9144,9528,9528,9528,9912,9912,9912,10296,10296,10296,10680,10680,10680,11064,11064,11064,11448,11448,11448,11448,11832,11832,11832,12216,12216,12216,12576,12576,12576,12960,12960,12960,12960,13536,13536,
		120,256,392,536,680,808,968,1096,1256,1384,1544,1672,1800,1928,2088,2216,2344,2536,2664,2792,2984,3112,3240,3368,3496,3624,3752,3880,4008,4264,4392,4584,4584,4776,4968,4968,5160,5352,5544,5544,5736,5992,5992,6200,6200,6456,6456,6712,6968,6968,7224,7224,7480,7480,7736,7736,7992,7992,8248,8504,8504,8760,8760,9144,9144,9144,9528,9528,9528,9912,9912,9912,10296,10296,10680,10680,10680,11064,11064,11064,11448,11448,11448,11832,11832,12216,12216,12216,12576,12576,12576,12960,12960,12960,13536,13536,13536,13536,14112,14112,14112,14112,14688,14688,14688,14688,15264,15264,15264,15264,
		136,296,456,616,776,936,1096,1256,1416,1544,1736,1864,2024,2216,2344,2536,2664,2856,2984,3112,3368,3496,3624,3752,4008,4136,4264,4392,4584,4776,4968,5160,5160,5352,5544,5736,5736,5992,6200,6200,6456,6712,6712,6968,6968,7224,7480,7480,7736,7992,7992,8248,8248,8504,8760,8760,9144,9144,9144,9528,9528,9912,9912,10296,10296,10296,10680,10680,11064,11064,11064,11448,11448,11832,11832,11832,12216,12216,12576,12576,12960,12960,12960,13536,13536,13536,13536,14112,14112,14112,14112,14688,14688,14688,15264,15264,15264,15264,15840,15840,15840,16416,16416,16416,16416,16992,16992,16992,16992,17568,
		144,328,504,680,872,1032,1224,1384,1544,1736,1928,2088,2280,2472,2664,2792,2984,3112,3368,3496,3752,3880,4008,4264,4392,4584,4776,4968,5160,5352,5544,5736,5736,5992,6200,6200,6456,6712,6712,6968,7224,7480,7480,7736,7992,7992,8248,8504,8504,8760,9144,9144,9144,9528,9528,9912,9912,10296,10296,10680,10680,11064,11064,11448,11448,11448,11832,11832,12216,12216,12576,12576,12960,12960,12960,13536,13536,13536,14112,14112,14112,14688,14688,14688,14688,15264,15264,15264,15840,15840,15840,16416,16416,16416,16992,16992,16992,16992,17568,17568,17568,18336,18336,18336,18336,18336,19080,19080,19080,19080,
		176,376,584,776,1000,1192,1384,1608,1800,2024,2216,2408,2600,2792,2984,3240,3496,3624,3880,4008,4264,4392,4584,4776,4968,5352,5544,5736,5992,5992,6200,6456,6712,6968,6968,7224,7480,7736,7736,7992,8248,8504,8760,8760,9144,9144,9528,9528,9912,9912,10296,10680,10680,11064,11064,11448,11448,11832,11832,12216,12216,12576,12576,12960,12960,13536,13536,13536,14112,14112,14112,14688,14688,14688,15264,15264,15840,15840,15840,16416,16416,16416,16992,16992,16992,17568,17568,17568,18336,18336,18336,18336,19080,19080,19080,19080,19848,19848,19848,19848,20616,20616,20616,21384,21384,21384,21384,22152,22152,22152,
		208,440,680,904,1128,1352,1608,1800,2024,2280,2472,2728,2984,3240,3368,3624,3880,4136,4392,4584,4776,4968,5352,5544,5736,5992,6200,6456,6712,6712,6968,7224,7480,7736,7992,8248,8504,8760,8760,9144,9528,9528,9912,9912,10296,10680,10680,11064,11064,11448,11832,11832,12216,12216,12576,12576,12960,12960,13536,13536,14112,14112,14112,14688,14688,15264,15264,15264,15840,15840,16416,16416,16416,16992,16992,17568,17568,17568,18336,18336,18336,19080,19080,19080,19080,19848,19848,19848,20616,20616,20616,21384,21384,21384,21384,22152,22152,22152,22920,22920,22920,23688,23688,23688,23688,24496,24496,24496,24496,25456,
		224,488,744,1000,1256,1544,1800,2024,2280,2536,2856,3112,3368,3624,3880,4136,4392,4584,4968,5160,5352,5736,5992,6200,6456,6712,6968,7224,7480,7736,7992,8248,8504,8760,9144,9144,9528,9912,9912,10296,10680,10680,11064,11448,11448,11832,12216,12216,12576,12960,12960,13536,13536,14112,14112,14688,14688,14688,15264,15264,15840,15840,16416,16416,16992,16992,16992,17568,17568,18336,18336,18336,19080,19080,19080,19848,19848,19848,20616,20616,20616,21384,21384,21384,22152,22152,22152,22920,22920,22920,23688,23688,23688,24496,24496,24496,25456,25456,25456,25456,26416,26416,26416,26416,27376,27376,27376,27376,28336,28336,
		256,552,840,1128,1416,1736,1992,2280,2600,2856,3112,3496,3752,4008,4264,4584,4968,5160,5544,5736,5992,6200,6456,6968,7224,7480,7736,7992,8248,8504,8760,9144,9528,9912,9912,10296,10680,11064,11064,11448,11832,12216,12216,12576,12960,12960,13536,13536,14112,14112,14688,14688,15264,15264,15840,15840,16416,16416,16992,16992,17568,17568,18336,18336,18336,19080,19080,19848,19848,19848,20616,20616,20616,21384,21384,22152,22152,22152,22920,22920,22920,23688,23688,24496,24496,24496,25456,25456,25456,25456,26416,26416,26416,27376,27376,27376,28336,28336,28336,28336,29296,29296,29296,29296,30576,30576,30576,30576,31704,31704,
		280,600,904,1224,1544,1800,2152,2472,2728,3112,3368,3624,4008,4264,4584,4968,5160,5544,5736,6200,6456,6712,6968,7224,7736,7992,8248,8504,8760,9144,9528,9912,10296,10296,10680,11064,11448,11832,11832,12216,12576,12960,12960,13536,13536,14112,14688,14688,15264,15264,15840,15840,16416,16416,16992,16992,17568,17568,18336,18336,18336,19080,19080,19848,19848,20616,20616,20616,21384,21384,22152,22152,22152,22920,22920,23688,23688,23688,24496,24496,24496,25456,25456,25456,26416,26416,26416,27376,27376,27376,28336,28336,28336,29296,29296,29296,29296,30576,30576,30576,30576,31704,31704,31704,31704,32856,32856,32856,34008,34008,
		328,632,968,1288,1608,1928,2280,2600,2984,3240,3624,3880,4264,4584,4968,5160,5544,5992,6200,6456,6712,7224,7480,7736,7992,8504,8760,9144,9528,9912,9912,10296,10680,11064,11448,11832,12216,12216,12576,12960,13536,13536,14112,14112,14688,14688,15264,15840,15840,16416,16416,16992,16992,17568,17568,18336,18336,19080,19080,19848,19848,19848,20616,20616,21384,21384,22152,22152,22152,22920,22920,23688,23688,24496,24496,24496,25456,25456,25456,26416,26416,26416,27376,27376,27376,28336,28336,28336,29296,29296,29296,30576,30576,30576,30576,31704,31704,31704,31704,32856,32856,32856,34008,34008,34008,34008,35160,35160,35160,35160,
		336,696,1064,1416,1800,2152,2536,2856,3240,3624,4008,4392,4776,5160,5352,5736,6200,6456,6712,7224,7480,7992,8248,8760,9144,9528,9912,10296,10296,10680,11064,11448,11832,12216,12576,12960,13536,13536,14112,14688,14688,15264,15264,15840,16416,16416,16992,17568,17568,18336,18336,19080,19080,19848,19848,20616,20616,20616,21384,21384,22152,22152,22920,22920,23688,23688,24496,24496,24496,25456,25456,26416,26416,26416,27376,27376,27376,28336,28336,29296,29296,29296,30576,30576,30576,30576,31704,31704,31704,32856,32856,32856,34008,34008,34008,35160,35160,35160,35160,36696,36696,36696,36696,37888,37888,37888,39232,39232,39232,39232,
		376,776,1160,1544,1992,2344,2792,3112,3624,4008,4392,4776,5160,5544,5992,6200,6712,7224,7480,7992,8248,8760,9144,9528,9912,10296,10680,11064,11448,11832,12216,12576,12960,13536,14112,14112,14688,15264,15264,15840,16416,16416,16992,17568,17568,18336,18336,19080,19080,19848,19848,20616,21384,21384,22152,22152,22920,22920,23688,23688,24496,24496,24496,25456,25456,26416,26416,27376,27376,27376,28336,28336,29296,29296,29296,30576,30576,30576,31704,31704,31704,32856,32856,32856,34008,34008,34008,35160,35160,35160,36696,36696,36696,37888,37888,37888,37888,39232,39232,39232,40576,40576,40576,40576,42368,42368,42368,42368,43816,43816,
		408,840,1288,1736,2152,2600,2984,3496,3880,4264,4776,5160,5544,5992,6456,6968,7224,7736,8248,8504,9144,9528,9912,10296,10680,11064,11448,12216,12576,12960,13536,13536,14112,14688,15264,15264,15840,16416,16992,16992,17568,18336,18336,19080,19080,19848,20616,20616,21384,21384,22152,22152,22920,22920,23688,24496,24496,25456,25456,25456,26416,26416,27376,27376,28336,28336,29296,29296,29296,30576,30576,30576,31704,31704,32856,32856,32856,34008,34008,34008,35160,35160,35160,36696,36696,36696,37888,37888,37888,39232,39232,39232,40576,40576,40576,40576,42368,42368,42368,43816,43816,43816,43816,45352,45352,45352,46888,46888,46888,46888,
		440,904,1384,1864,2344,2792,3240,3752,4136,4584,5160,5544,5992,6456,6968,7480,7992,8248,8760,9144,9912,10296,10680,11064,11448,12216,12576,12960,13536,14112,14688,14688,15264,15840,16416,16992,16992,17568,18336,18336,19080,19848,19848,20616,20616,21384,22152,22152,22920,22920,23688,24496,24496,25456,25456,26416,26416,27376,27376,28336,28336,29296,29296,29296,30576,30576,31704,31704,31704,32856,32856,34008,34008,34008,35160,35160,35160,36696,36696,36696,37888,37888,39232,39232,39232,40576,40576,40576,42368,42368,42368,42368,43816,43816,43816,45352,45352,45352,46888,46888,46888,46888,48936,48936,48936,48936,48936,51024,51024,51024,
		488,1000,1480,1992,2472,2984,3496,4008,4584,4968,5544,5992,6456,6968,7480,7992,8504,9144,9528,9912,10680,11064,11448,12216,12576,12960,13536,14112,14688,15264,15840,15840,16416,16992,17568,18336,18336,19080,19848,19848,20616,21384,21384,22152,22920,22920,23688,24496,24496,25456,25456,26416,26416,27376,27376,28336,28336,29296,29296,30576,30576,31704,31704,31704,32856,32856,34008,34008,35160,35160,35160,36696,36696,36696,37888,37888,39232,39232,39232,40576,40576,40576,42368,42368,42368,43816,43816,43816,45352,45352,45352,46888,46888,46888,46888,48936,48936,48936,48936,51024,51024,51024,51024,52752,52752,52752,52752,55056,55056,55056,
		520,1064,1608,2152,2664,3240,3752,4264,4776,5352,5992,6456,6968,7480,7992,8504,9144,9528,10296,10680,11448,11832,12576,12960,13536,14112,14688,15264,15840,16416,16992,16992,17568,18336,19080,19080,19848,20616,21384,21384,22152,22920,22920,23688,24496,24496,25456,25456,26416,27376,27376,28336,28336,29296,29296,30576,30576,31704,31704,32856,32856,34008,34008,34008,35160,35160,36696,36696,36696,37888,37888,39232,39232,40576,40576,40576,42368,42368,42368,43816,43816,43816,45352,45352,45352,46888,46888,46888,48936,48936,48936,48936,51024,51024,51024,51024,52752,52752,52752,55056,55056,55056,55056,57336,57336,57336,57336,59256,59256,59256,
		552,1128,1736,2280,2856,3496,4008,4584,5160,5736,6200,6968,7480,7992,8504,9144,9912,10296,11064,11448,12216,12576,12960,13536,14112,14688,15264,15840,16416,16992,17568,18336,19080,19848,19848,20616,21384,22152,22152,22920,23688,24496,24496,25456,25456,26416,27376,27376,28336,28336,29296,29296,30576,30576,31704,31704,32856,32856,34008,34008,35160,35160,36696,36696,37888,37888,37888,39232,39232,40576,40576,40576,42368,42368,43816,43816,43816,45352,45352,45352,46888,46888,46888,48936,48936,48936,51024,51024,51024,51024,52752,52752,52752,55056,55056,55056,55056,57336,57336,57336,57336,59256,59256,59256,59256,61664,61664,61664,61664,63776,
		584,1192,1800,2408,2984,3624,4264,4968,5544,5992,6712,7224,7992,8504,9144,9912,10296,11064,11448,12216,12960,13536,14112,14688,15264,15840,16416,16992,17568,18336,19080,19848,19848,20616,21384,22152,22920,22920,23688,24496,25456,25456,26416,26416,27376,28336,28336,29296,29296,30576,31704,31704,32856,32856,34008,34008,35160,35160,36696,36696,36696,37888,37888,39232,39232,40576,40576,42368,42368,42368,43816,43816,45352,45352,45352,46888,46888,46888,48936,48936,48936,51024,51024,51024,52752,52752,52752,52752,55056,55056,55056,57336,57336,57336,57336,59256,59256,59256,61664,61664,61664,61664,63776,63776,63776,63776,66592,66592,66592,66592,
		616,1256,1864,2536,3112,3752,4392,5160,5736,6200,6968,7480,8248,8760,9528,10296,10680,11448,12216,12576,13536,14112,14688,15264,15840,16416,16992,17568,18336,19080,19848,20616,20616,21384,22152,22920,23688,24496,24496,25456,26416,26416,27376,28336,28336,29296,29296,30576,31704,31704,32856,32856,34008,34008,35160,35160,36696,36696,37888,37888,39232,39232,40576,40576,40576,42368,42368,43816,43816,43816,45352,45352,46888,46888,46888,48936,48936,48936,51024,51024,51024,52752,52752,52752,55056,55056,55056,55056,57336,57336,57336,59256,59256,59256,61664,61664,61664,61664,63776,63776,63776,63776,66592,66592,66592,66592,68808,68808,68808,71112,
		712,1480,2216,2984,3752,4392,5160,5992,6712,7480,8248,8760,9528,10296,11064,11832,12576,13536,14112,14688,15264,16416,16992,17568,18336,19080,19848,20616,21384,22152,22920,23688,24496,25456,25456,26416,27376,28336,29296,29296,30576,30576,31704,32856,32856,34008,35160,35160,36696,36696,37888,37888,39232,40576,40576,40576,42368,42368,43816,43816,45352,45352,46888,46888,48936,48936,48936,51024,51024,52752,52752,52752,55056,55056,55056,55056,57336,57336,57336,59256,59256,59256,61664,61664,61664,63776,63776,63776,66592,66592,66592,68808,68808,68808,71112,71112,71112,73712,73712,75376,75376,75376,75376,75376,75376,75376,75376,75376,75376,75376};

        uint16_t count = iTB * 110;
        while (tbList[count++] != tbSizeb);
      //  std::cout<<"count"<<count % 110<<std::endl;

        return (count % 110);

    }

    void
    MyGymEnv::GetPhyStats(Ptr < MyGymEnv > gymEnv,
        const PhyTransmissionStatParameters params) {
    
        
        if (gymEnv -> collect == 1) {
            NS_LOG_LOGIC(" ======= New Transmission =======");
            NS_LOG_LOGIC("Packed sent by" << " cell: " << params.m_cellId << " to UE: " << params.m_rnti);
            // Get size of TB
            uint32_t idx = params.m_cellId - 1;
            gymEnv -> m_cellFrequency.at(idx) = gymEnv -> m_cellFrequency.at(idx) + 1;
            gymEnv -> m_dlThroughputVec.at(idx) = gymEnv -> m_dlThroughputVec.at(idx) + (params.m_size) * 8.0 / 1024.0 / 1024.0 / gymEnv -> collecting_window;
            gymEnv -> m_dlThroughput = gymEnv -> m_dlThroughput + (params.m_size) * 8.0 / 1024.0 / 1024.0 / gymEnv -> collecting_window;
            NS_LOG_LOGIC("sent data at cell " << idx << " is " << gymEnv -> m_dlThroughputVec.at(idx) * gymEnv -> collecting_window);
            
            //add throughput per user
            gymEnv -> UserThrouput[params.m_cellId][params.m_rnti] = gymEnv -> UserThrouput[params.m_cellId][params.m_rnti] + (params.m_size) * 8.0 / 1024.0 / 1024.0 / gymEnv -> collecting_window;
            // std::cout<<"UserThrouput at cell " << UserThrouput[params.m_cellId][params.m_rnti]<<std::endl;
            // Get nRBs
            uint8_t nRBs = MyGymEnv::GetnRB(MyGymEnv::Convert2ITB(params.m_mcs), params.m_size);
          //  std::cout<<"params.m_size" << params.m_size<<std::endl;
           // std::cout<<"nRBs" << nRBs<<std::endl;
          //  std::cout<<"idx" <<idx<<std::endl;
            gymEnv -> m_rbUtil.at(idx) = gymEnv -> m_rbUtil.at(idx) + nRBs;
           

            // Get MCSPen
            gymEnv -> m_MCSPen.at(idx).at(params.m_mcs) = gymEnv -> m_MCSPen.at(idx).at(params.m_mcs) + 1;
            NS_LOG_LOGIC("Frequency at cell " << idx << " is " << gymEnv -> m_cellFrequency.at(idx));
            NS_LOG_LOGIC("DLThroughput at cell " << idx << " is " << gymEnv -> m_dlThroughputVec.at(idx));
            NS_LOG_LOGIC("NRB at cell " << idx << " is " << gymEnv -> m_rbUtil.at(idx));
            NS_LOG_LOGIC("MCS frequency at cell " << idx << " of MCS " << uint32_t(params.m_mcs) << " is " << gymEnv -> m_MCSPen[idx][params.m_mcs]);
        }
    }

} // ns3 namespace
