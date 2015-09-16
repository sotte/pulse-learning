#include <gtest/gtest.h>

#include <memory> // std::shared_ptr
#include <limits>

#include "TemporallyExtendedModel.h"

#define DEBUG_STRING "Unit Tests: "
#define DEBUG_LEVEL 0
#include "debug.h"

using std::cout;
using std::endl;
using std::shared_ptr;
using std::make_shared;

class TemporallyExtendedModelTest: public ::testing::Test {
protected:
    typedef TemporallyExtendedModel::data_t data_t;
    typedef TemporallyExtendedModel::DataPoint DataPoint;
    virtual void SetUp() {
        // use an implementation of the simple 2x2 world
        int one_step_observation = 0;
        int two_step_observation = 0;
        for(int i=0; i<data_n; ++i) {
            // perform transition
            int action = rand()%5; // 0:up, 1:down, 2:left, 3:right, 4:stay
            int observation = one_step_observation; // 0:upper-left, 1:upper-right, 2:lower-left, 3:lower-right
            switch(one_step_observation) {
            case 0:
                if(action==3) observation = 1;
                if(action==1) observation = 2;
                break;
            case 1:
                if(action==2) observation = 0;
                if(action==1) observation = 3;
                break;
            case 2:
                if(action==0) observation = 0;
                if(action==3) observation = 3;
                break;
            case 3:
                if(action==0) observation = 1;
                if(action==2) observation = 2;
                break;
            default:
                EXPECT_TRUE(false) << "This line should never be reached " << one_step_observation;
            }
            double reward = (two_step_observation==0 && observation==3)?1:0;
            // print transitions
            IF_DEBUG(2) {
                const char * action_char = "";
                switch(action) {
                case 0:
                    action_char = "↑";
                    break;
                case 1:
                    action_char = "↓";
                    break;
                case 2:
                    action_char = "←";
                    break;
                case 3:
                    action_char = "→";
                    break;
                case 4:
                    action_char = "•";
                    break;
                default:
                    EXPECT_TRUE(false) << "This line should never be reached " << action;
                }
                const char * observation_char = "";
                switch(observation) {
                case 0:
                    observation_char = "◰";
                    break;
                case 1:
                    observation_char = "◳";
                    break;
                case 2:
                    observation_char = "◱";
                    break;
                case 3:
                    observation_char = "◲";
                    break;
                default:
                    EXPECT_TRUE(false) << "This line should never be reached " << observation;
                }
                DEBUG_OUT(1,action_char << " --> " << observation_char << " / " << reward);
            }
            // store to data
            data.push_back(DataPoint(action, observation, reward));
            // update memory
            two_step_observation = one_step_observation;
            one_step_observation = observation;
        }
    }
    //virtual void TearDown();
    int data_n = 1000;
    data_t data;
};

TEST_F(TemporallyExtendedModelTest, DataTest) {
    // count occurrences in data
    std::map<int,int> action_map, observation_map, reward_map;
    std::map<DataPoint,int> transition_map;
    for(auto point : data) {
        ++action_map[point.action];
        ++observation_map[point.observation];
        ++reward_map[point.reward];
        ++transition_map[point];
    }
    // check number of unique value
    EXPECT_EQ(action_map.size(),5)
        << "Number of actions that were taken not as expected";
    EXPECT_EQ(observation_map.size(),4)
        << "Number of observations that were made not as expected";
    EXPECT_EQ(reward_map.size(),2)
        << "Number of rewards that were received not as expected";
    EXPECT_EQ(transition_map.size(),14)
        << "Number of transitions that occurred not as expected";
    // check whether counts are in reasonable bounds
    for(auto a : action_map)
        EXPECT_GT(a.second,data_n/10)
            << "Action " << a.first << " was only taken " << a.second << " times in " << data_n << " steps";
    for(auto o : observation_map)
        EXPECT_GT(o.second,data_n/8)
            << "Observation " << o.first << " was only make " << o.second << " times in " << data_n << " steps";
    EXPECT_GT(reward_map[0],data_n-data_n/10)
        << "A reward of 0 was only received " << reward_map[0] << " times in " << data_n << " steps";
    EXPECT_LT(reward_map[1],data_n/10)
        << "A reward of 1 was received " << reward_map[1] << " times in " << data_n << " steps, which is more than expected";
    for(auto t : transition_map) {
        if(t.first.reward==0) EXPECT_GT(t.second,data_n/28);
        else EXPECT_LT(t.second,data_n/28);
    }
}

TEST_F(TemporallyExtendedModelTest, FeatureTest) {
    // learn
    TemporallyExtendedModel TEM;
    double reg = 0.01;
    double obj = TEM.set_data(data).
        set_regularization(reg).
        set_horizon_extension(2).
        set_maximum_horizon(2).
        set_max_outer_loop_iterations(2).
        optimize();
    DEBUG_OUT(1, "likelihood=" << obj << ", regularization=" << reg);
    IF_DEBUG(2) TEM.print_feature_set();

    obj = TEM.set_regularization(0).optimize_weights();
    DEBUG_OUT(1, "likelihood=" << obj << ", regularization=" << 0);
    IF_DEBUG(2) TEM.print_feature_set();

    // check features
    for(int data_idx=0; data_idx<(int)data.size(); ++data_idx) {
        IF_DEBUG(2) {
            DEBUG_OUT(2,"data point " << data_idx << ":	" << data[data_idx].action << "	" << data[data_idx].observation << "	" << data[data_idx].reward);
        }
        DEBUG_INDENT;
        int feature_idx = 0;
        for(auto & feature : TEM.feature_set) {
            DEBUG_OUT(3,"Feature " << feature_idx);
            DEBUG_INDENT;
            bool is_true = true;
            for(auto & basis_feature : feature.first) {
                bool this_one_true = true;
                typedef TemporallyExtendedModel::FEATURE_TYPE type_t;
                type_t type;
                int time;
                double value;
                std::tuple<type_t&,int&,double&>(type,time,value) = basis_feature;
                if(data_idx+time<0) {
                    this_one_true = false;
                }
                switch(type) {
                case TemporallyExtendedModel::ACTION:
                    if(data[data_idx+time].action!=value) this_one_true = false;
                    DEBUG_OUT(3,"a(" << value << "," << time << ")=" << this_one_true);
                    break;
                case TemporallyExtendedModel::OBSERVATION:
                    if(data[data_idx+time].observation!=value) this_one_true = false;
                    DEBUG_OUT(3,"o(" << value << "," << time << ")=" << this_one_true);
                    break;
                case TemporallyExtendedModel::REWARD:
                    if(data[data_idx+time].reward!=value) this_one_true = false;
                    DEBUG_OUT(3,"r(" << value << "," << time << ")=" << this_one_true);
                    break;
                }
                is_true = is_true && this_one_true;
            }
            EXPECT_EQ(is_true,TEM.F_matrices[data_idx](feature_idx,TEM.outcome_indices[data_idx]));
            ++feature_idx;
        }
    }
}

TEST_F(TemporallyExtendedModelTest, Learn) {
    // learn
    TemporallyExtendedModel TEM;
    double reg = 0.001;
    double obj = TEM.set_data(data).
        set_regularization(reg).
        set_horizon_extension(2).
        set_maximum_horizon(2).
        set_max_outer_loop_iterations(2).
        optimize();
    DEBUG_OUT(1, "likelihood=" << obj << ", regularization=" << reg);
    IF_DEBUG(2) TEM.print_feature_set();

    obj = TEM.set_regularization(0).optimize_weights();
    DEBUG_OUT(1, "likelihood=" << obj << ", regularization=" << 0);
    IF_DEBUG(2) TEM.print_feature_set();

    // check prediction
    data_t data_copy;
    double min_pred = std::numeric_limits<double>::max();
    double max_pred = std::numeric_limits<double>::lowest();
    int data_idx = 1;
    for(auto d : data) {
        // ignore first two data points since they may be bad becaus of the
        // missing history
        if(data_idx<=2) continue;
        data_copy.push_back(d);
        double pred =  TEM.get_prediction(data_copy);
        min_pred = std::min(min_pred,pred);
        max_pred = std::max(min_pred,pred);
        EXPECT_GT(pred,0.99) << "predictive probability too small for data point " << data_idx;
        EXPECT_LT(pred,1);
        DEBUG_OUT(2,"predictive probability " << pred);
        ++data_idx;
    }
    DEBUG_OUT(1,"Predictive probabilities are in range [" << min_pred << "," << max_pred << "]");
}

TEST_F(TemporallyExtendedModelTest, Derivatives) {
    // expand twice (to get an acceptably large feature set) with only one
    // optimization step of the weights (to get weight to non-zero but not
    // at optimum, where the gradient is zero)
    TemporallyExtendedModel TEM;
    TEM.set_data(data).
        set_regularization(0.001).
        set_max_outer_loop_iterations(1).
        set_max_inner_loop_iterations(1);
    TEM.optimize();
    TEM.optimize();
    EXPECT_TRUE(TEM.check_derivatives());
}
