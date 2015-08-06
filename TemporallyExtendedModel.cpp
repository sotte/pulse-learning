#include "TemporallyExtendedModel.h"

#include <iostream>

#include "lbfgs_codes.h"

#define DEBUG_STRING "TEM: "
#define DEBUG_LEVEL 3
#include "debug.h"

using std::cout;
using std::endl;
using arma::zeros;

typedef TemporallyExtendedModel::action_t action_t;
typedef TemporallyExtendedModel::observation_t observation_t;
typedef TemporallyExtendedModel::reward_t reward_t;

// some helper macros and functions

#define DATA_POINT(name, action, observation, reward)                   \
    TemporallyExtendedModel::action_t action;                           \
    TemporallyExtendedModel::observation_t observation;                 \
    TemporallyExtendedModel::reward_t reward;                           \
    std::tuple<action_t&,observation_t&,reward_t&> name(action,observation,reward);

#define BASIS_FEATURE(name, type, time, value)                          \
    TemporallyExtendedModel::FEATURE_TYPE type;                         \
    int time;                                                           \
    double value;                                                       \
    std::tuple<TemporallyExtendedModel::FEATURE_TYPE&,int&,double&> name(type,time,value);

std::ostream& operator<<(std::ostream & out,
                         const TemporallyExtendedModel::basis_feature_t & basis_feature) {
    BASIS_FEATURE(tuple,type,time,value);
    tuple = basis_feature;
    switch(type) {
    case TemporallyExtendedModel::ACTION:
        out << "a";
        break;
    case TemporallyExtendedModel::OBSERVATION:
        out << "o";
        break;
    case TemporallyExtendedModel::REWARD:
        out << "r";
        break;
    }
    out << "(" << value << ", " << time << ")";
    return out;
}

// member function definitions

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
    // compute outcome indices
    outcome_indices.assign(data.size(),-1);
    for(int data_idx=0; data_idx<(int)data.size(); ++data_idx) {
        int outcome_index = 0;
        bool found = false;
        for(auto & observation : unique_observations) {
            for(auto & reward : unique_rewards) {
                DATA_POINT(data_point, data_action, data_observation, data_reward);
                data_point = data[data_idx];
                if(data_observation==observation && data_reward==reward) {
                    outcome_indices[data_idx] = outcome_index;
                    found = true;
                    break;
                }
                ++outcome_index;
            }
            if(found) break;
        }
        DEBUG_EXPECT(found);
    }
    return *this;
}

double TemporallyExtendedModel::optimize() {
    DEBUG_OUT(1,"PULSE optimization");
    DEBUG_INDENT;

    // return value after optimization
    double likelihood = 0;

    // outer optimization loop
    for(int outer_loop_iteration=0;
        (max_outer_loop_iterations<=0 || outer_loop_iteration<max_outer_loop_iterations);
        ++outer_loop_iteration) {
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
                BASIS_FEATURE(tuple,type,time,value);
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
                    BASIS_FEATURE(tuple_1, type_1, time_1, value_1);
                    BASIS_FEATURE(tuple_2, type_2, time_2, value_2);
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
    // update F-matrices
    update_F_matrices();
    // optimize weights
    lbfgsfloatval_t objective_value;
    {
        // initialize the parameters
        lbfgs_parameter_t param;
        lbfgs_parameter_init(&param);
        param.orthantwise_c = regularization;
        param.delta = likelihood_threshold;
        param.epsilon = gradient_threshold;
        if(regularization!=0) {
            param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
        }
        if(max_inner_loop_iterations>0) {
            param.max_iterations = max_inner_loop_iterations;
        }
        // initialize variables
        lbfgsfloatval_t * weights = lbfgs_malloc(feature_set.size());
        // set weights
        {
            int feature_idx = 0;
            for(auto & feature : feature_set) {
                weights[feature_idx] = feature.second;
                ++feature_idx;
            }
        }
        // start the L-BFGS optimization
        auto ret = lbfgs(feature_set.size(),
                         weights,
                         &objective_value,
                         neg_log_likelihood,
                         progress,
                         this,
                         &param);
        DEBUG_OUT(1,"status code = " << ret << " ( " << lbfgs_code(ret) << " )");
        // get weights
        {
            int feature_idx = 0;
            for(auto & feature : feature_set) {
                feature.second = weights[feature_idx];
                ++feature_idx;
            }
        }
        // free weights
        lbfgs_free(weights);
    }
    // print likelihood
    DEBUG_OUT(3,"likelihood = " << exp(-objective_value));
    return exp(-objective_value);
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
            cout << "    " << basis_feature << endl;
        }
        cout << endl;
        ++f_idx;
    }
    if(f_idx==1) cout << "    empty" << endl;
}

void TemporallyExtendedModel::update_F_matrices() {
    DEBUG_OUT(4,"Update F-matrices");
    DEBUG_INDENT;
    F_matrices.assign(data.size(),
                      zeros<mat_t>(feature_set.size(),
                                   unique_observations.size()*unique_rewards.size()));
    for(int data_idx=0; data_idx<(int)data.size(); ++data_idx) {
        DEBUG_OUT(5,"data point " << data_idx);
        DEBUG_INDENT;
        int feature_idx = 0; // row index
        for(auto & feature : feature_set) {
            DEBUG_OUT(5,"Feature " << feature_idx);
            DEBUG_INDENT;
            int outcome_idx = 0; // column index
            for(auto & observation : unique_observations) {
                for(auto & reward : unique_rewards) {
                    DEBUG_OUT(5,"Outcome " << outcome_idx << " (" << observation << ", " << reward << ")");
                    DEBUG_INDENT;
                    bool is_true = true;
                    for(auto & basis_feature : feature.first) {
                        //-----------------------------------//
                        // all basis feature have to be true //
                        //-----------------------------------//
                        BASIS_FEATURE(tuple, type, time, value);
                        tuple = basis_feature;
                        DEBUG_EXPECT(time<=0);
                        DEBUG_OUT(5,"Basis feature " << basis_feature);
                        DEBUG_INDENT;
                        // is the required time index accessible?
                        if(data_idx+time<0) {
                            DEBUG_OUT(5,"time idx inaccessible");
                            is_true = false;
                            break;
                        }
                        // get data point from required time
                        DATA_POINT(data_point, data_action, data_observation, data_reward);
                        data_point = data[data_idx+time];
                        // does the value match?
                        switch(type) {
                        case ACTION:
                            if(data_action!=value) is_true = false;
                            break;
                        case OBSERVATION:
                            if(time==0 && observation!=value) is_true = false;
                            if(time!=0 && data_observation!=value) is_true = false;
                            break;
                        case REWARD:
                            if(time==0 && reward!=value) is_true = false;
                            if(time!=0 && data_reward!=value) is_true = false;
                            break;
                        }
                        // break
                        if(!is_true) {
                            DEBUG_OUT(5,"value mismatch");
                            break;
                        }
                    }
                    if(is_true) {
                        F_matrices[data_idx](feature_idx,outcome_idx) = 1;
                        DEBUG_OUT(5,"true");
                    } else {
                        DEBUG_OUT(5,"false");
                    }
                    ++outcome_idx;
                }
            }
            ++feature_idx;
        }
    }
}

lbfgsfloatval_t TemporallyExtendedModel::neg_log_likelihood(void * instance,
                                                            const lbfgsfloatval_t * weights,
                                                            lbfgsfloatval_t * gradient,
                                                            const int n,
                                                            const lbfgsfloatval_t /*step*/) {
    DEBUG_OUT(5,"Neg-Log-Likelihood");
    DEBUG_INDENT;

    // weights and gradient use given memory
    const col_vec_t w(weights,n); // unfortunately have to copy :-(
    col_vec_t grad(gradient,n,false);

    // get instance and number of data points
    auto TEM_instance = (TemporallyExtendedModel*)instance;
    int data_n = TEM_instance->data.size();

    // initialize
    lbfgsfloatval_t neg_log_like = 0;
    grad.zeros(TEM_instance->feature_set.size());

    // sum over data points
    for(int data_idx=0; data_idx<data_n; ++data_idx) {
        // use references to improve readability
        const auto & F = TEM_instance->F_matrices[data_idx];
        const int & outcome_idx = TEM_instance->outcome_indices[data_idx];
        // interim variables
        const row_vec_t lin = w.t()*F;
        const row_vec_t exp_lin = arma::exp(lin);
        const double z = arma::sum(exp_lin);
        // terms of objective and gradient
        double obj_term = lin(outcome_idx)-log(z);
        col_vec_t grad_term = F.col(outcome_idx) - F*exp_lin.t()/z;
        // update objective and gradient
        neg_log_like += obj_term;
        grad += grad_term;
    }

    // divide by number of data points and reverse sign
    if(data_n>0) {
        neg_log_like = -neg_log_like/data_n;
        grad = -grad/data_n;
    }

    // print weights and gradient
    IF_DEBUG(6) {
        {
            DEBUG_OUT(6,"weights");
            DEBUG_INDENT;
            for(int idx=0; idx<n; ++idx) {
                DEBUG_OUT(6,idx << ": " << w(idx));
            }
        }
        {
            DEBUG_OUT(6,"gradient");
            DEBUG_INDENT;
            for(int idx=0; idx<n; ++idx) {
                DEBUG_OUT(6,idx << ": " << grad(idx));
            }
        }
    }

    return neg_log_like;
}

int TemporallyExtendedModel::progress(void * instance,
                                      const lbfgsfloatval_t * weights,
                                      const lbfgsfloatval_t * gradient,
                                      const lbfgsfloatval_t objective_value,
                                      const lbfgsfloatval_t xnorm,
                                      const lbfgsfloatval_t gnorm,
                                      const lbfgsfloatval_t step,
                                      int nr_variables,
                                      int iteration_nr,
                                      int ls) {
    DEBUG_OUT(3,"Iteration " << iteration_nr << ", likelihood = " << exp(-objective_value));
    return 0;
}
