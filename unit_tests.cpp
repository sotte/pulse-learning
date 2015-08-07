#include <gtest/gtest.h>

#include <memory> // std::shared_ptr

#include "TemporallyExtendedModel.h"

#define DEBUG_STRING "Unit Tests: "
#define DEBUG_LEVEL 1
#include "debug.h"

using std::cout;
using std::endl;
using std::shared_ptr;
using std::make_shared;

class TemporallyExtendedModelTest: public ::testing::Test {
protected:
    typedef TemporallyExtendedModel::data_t data_t;
    typedef TemporallyExtendedModel::data_point_t data_point_t;
    virtual void SetUp() {
        // use an implementation of the simple 2x2 world
        int one_step_observation = 0;
        int two_step_observation = 0;
        for(int i=0; i<10000; ++i) {
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
            data.push_back(data_point_t(action, observation, reward));
            // update memory
            two_step_observation = one_step_observation;
            one_step_observation = observation;
        }
    }
    //virtual void TearDown();
    data_t data;
};

TEST_F(TemporallyExtendedModelTest, Learn) {
    auto TEM = new TemporallyExtendedModel();
    shared_ptr<AbstractTemporallyExtendedModel> model(TEM);
    double reg = 0.0001;
    double obj = model->set_data(data).
        set_regularization(reg).
        set_horizon_extension(1).
        set_maximum_horizon(2).
        set_max_outer_loop_iterations(2).
        optimize();
    DEBUG_OUT(1, "likelihood=" << obj << ", regularization=" << reg);
    obj = model->set_regularization(0).optimize_weights();
    DEBUG_OUT(1, "likelihood=" << obj << ", regularization=" << 0);
    TEM->print_feature_set();
    // double pred =  model->get_prediction(data);
    // cout << "Prediction: " << pred << endl;
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
