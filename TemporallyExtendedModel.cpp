#include "TemporallyExtendedModel.h"

#include <iostream>

#define DEBUG_STRING "TEM: "
#define DEBUG_LEVEL 3
#include "debug.h"

using std::cout;
using std::endl;

typedef TemporallyExtendedModel::action_t action_t;
typedef TemporallyExtendedModel::observation_t observation_t;
typedef TemporallyExtendedModel::reward_t reward_t;

#define DATA_POINT(name, action, observation, reward)                   \
    action_t action;                                                    \
    observation_t observation;                                          \
    reward_t reward;                                                    \
    std::tuple<action_t&,observation_t&,reward_t&> name(action,observation,reward);

#define FEATURE(name, type, time, value)                                \
    FEATURE_TYPE type;                                                  \
    int time;                                                           \
    double value;                                                       \
    std::tuple<FEATURE_TYPE&,int&,double&> name(type,time,value);

AbstractTemporallyExtendedModel & TemporallyExtendedModel::set_data(const data_t & data_) {
    DEBUG_OUT(1,"Set data");
    DEBUG_INDENT;
    data = data_;
    // update unique values
    unique_actions.clear();
    unique_observations.clear();
    unique_rewards.clear();
    for(auto & d : data) {
        DATA_POINT(point, action, observation, reward);
        point = d;
        unique_actions.insert(action);
        unique_observations.insert(observation);
        unique_rewards.insert(reward);
    }
    // debug output
    IF_DEBUG(3) {
        DEBUG_OUT(3,"Unique actions");
        {
            DEBUG_INDENT;
            for(auto action : unique_actions) {
                DEBUG_OUT(3, action);
            }
        }
        DEBUG_OUT(3,"Unique observations");
        {
            DEBUG_INDENT;
            for(auto observation : unique_observations) {
                DEBUG_OUT(3, observation);
            }
        }
        DEBUG_OUT(3,"Unique rewards");
        {
            DEBUG_INDENT;
            for(auto reward : unique_rewards) {
                DEBUG_OUT(3, reward);
            }
        }
    }
    return *this;
}

double TemporallyExtendedModel::optimize() {
    DEBUG_OUT(1,"PULSE optimization");
    DEBUG_INDENT;

    // return value after optimization
    double likelihood = 0;

    // outer optimization loop
    for(int outer_loop_iteration=0; outer_loop_iteration<max_outer_loop_iterations; ++outer_loop_iteration) {
        DEBUG_OUT(2,"Iteration " << outer_loop_iteration);
        DEBUG_INDENT;

        // expand --> optimize --> shrink
        expand_feature_set();
        double new_likelihook = optimize_feature_weights();
        shrink_feature_set();

        // checking terminal conditions
        if(new_likelihook-likelihood<likelihood_threshold) {
            likelihood = new_likelihook;
            break;
        } else {
            likelihood = new_likelihook;
        }
    }

    return likelihood;
}

double TemporallyExtendedModel::get_prediction(const data_t &) {
    DEBUG_OUT(1,"get_prediction()");
    return 0;
}

void TemporallyExtendedModel::expand_feature_set() {
    // first make a copy of the initial feature set which remains unchanged during expansion
    auto initial_feature_set = feature_set;
    // initialize if feature set is empty expand otherwise
    if(feature_set.empty()) {
        // add simple basis features
        for(auto action : unique_actions) {
            feature_set[feature_t({basis_feature_t(ACTION,0,action)})] = 0;
        }
        for(auto observation : unique_observations) {
            feature_set[feature_t({basis_feature_t(OBSERVATION,0,observation)})] = 0;
        }
        for(auto reward : unique_rewards) {
            feature_set[feature_t({basis_feature_t(REWARD,0,reward)})] = 0;
        }
    } else {
        // find maximum temporal extension
        int max_extension = 0;
        for(auto & feature : initial_feature_set) {
            for(auto & basis_feature : feature.first) {
                FEATURE(tuple,type,time,value);
                tuple = basis_feature;
                max_extension = std::max(max_extension,time);
            }
        }
        // go one step further (if permitted)
        DEBUG_EXPECT(max_extension<=0);
        DEBUG_EXPECT(horizon_extension>=0);
        DEBUG_EXPECT(maximum_horizon>=-1);
        if(max_extension-horizon_extension<-maximum_horizon && maximum_horizon!=-1) {
            // we cannot extend horizon by full amount an set it to the maximum
            // allowed horizon instead
            max_extension = -maximum_horizon;
        } else {
            // we have not reached maximum allowed horizon or it is infinite (-1) so
            // we extend by specified amount
            max_extension -= horizon_extension;
        }
        // go through all (initial) features, for all possible temporal delays
        // augment with simple basis features, and add to set
        for(auto & feature : initial_feature_set) {
            for(int t_idx = 0; t_idx>=max_extension; --t_idx) {
                for(auto action : unique_actions) {
                    feature_t f = feature.first;
                    f.insert(basis_feature_t(ACTION,t_idx,action));
                    feature_set[f] = 0;
                }
                for(auto observation : unique_observations) {
                    feature_t f = feature.first;
                    f.insert(basis_feature_t(OBSERVATION,t_idx,observation));
                    feature_set[f] = 0;
                }
                for(auto reward : unique_rewards) {
                    feature_t f = feature.first;
                    f.insert(basis_feature_t(REWARD,t_idx,reward));
                    feature_set[f] = 0;
                }
            }
        }
        // remove contradictory feature (same type, same time, different value)
        for(auto feature_it = feature_set.begin();
            feature_it!=feature_set.end();
            /*increment manually*/) {
            bool deleted = false;
            for(auto basis_feature_it = feature_it->first.begin();
                basis_feature_it!=feature_it->first.end();
                ++basis_feature_it) {
                // basis features are ordered first by type, second by time, and
                // third by value so that contradictory feature will always be
                // adjacent
                auto next_basis_feature_it = basis_feature_it;
                ++next_basis_feature_it;
                if(next_basis_feature_it!=feature_it->first.end()) {
                    FEATURE(tuple_1, type_1, time_1, value_1);
                    FEATURE(tuple_2, type_2, time_2, value_2);
                    tuple_1 = *basis_feature_it;
                    tuple_2 = *next_basis_feature_it;
                    if(type_1==type_2 && time_1==time_2 && value_1!=value_2) {
                        // this feature contains contradictory basis features
                        auto delete_feature_it = feature_it;
                        ++feature_it;
                        feature_set.erase(delete_feature_it);
                        deleted = true;
                        break;
                    }
                }
            }
            // increment
            if(!deleted) ++feature_it;
        }
        // restore values from initial feature set
        for(auto & feature : initial_feature_set) {
            feature_set[feature.first] = feature.second;
        }
    }
    // print
    DEBUG_OUT(3,"Expanded feature set (" << initial_feature_set.size() << " --> " << feature_set.size() << ")");
    IF_DEBUG(4) {
        print_feature_set();
    }
}

double TemporallyExtendedModel::optimize_feature_weights() {
    DEBUG_OUT(3,"Optimizting feature weights");
    DEBUG_INDENT;
    for(auto & feature : feature_set) {
        feature.second = rand()%2==0;
    }
    DEBUG_OUT(3,"likelihood = " << 1);
    return 1;
}

void TemporallyExtendedModel::shrink_feature_set() {
    int old_size = feature_set.size();
    for(auto feature_it = feature_set.begin(); feature_it!=feature_set.end(); /*increment manually*/) {
        if(feature_it->second==0) {
            auto delete_feature_it = feature_it;
            ++feature_it;
            feature_set.erase(delete_feature_it);
        } else {
            ++feature_it;
        }
    }
    // print
    DEBUG_OUT(3,"Shrunk feature set (" << old_size << " --> " << feature_set.size() << ")");
    IF_DEBUG(4) {
        print_feature_set();
    }
}

void TemporallyExtendedModel::print_feature_set() {
    int f_idx = 1;
    for(auto & feature : feature_set) {
        cout << "Feature " << f_idx << " (" << feature.second << ")" << endl;
        for(auto & basis_feature : feature.first) {
            FEATURE(tuple,type,time,value);
            tuple = basis_feature;
            cout << "    ";
            switch(type) {
            case ACTION:
                cout << "    a";
                break;
            case OBSERVATION:
                cout << "    o";
                break;
            case REWARD:
                cout << "    r";
                break;
            }
            cout << "(" << value << ", " << time << ")" << endl;
        }
        cout << endl;
        ++f_idx;
    }
    if(f_idx==1) cout << "    empty" << endl;
}
