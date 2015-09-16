#include "TemporallyExtendedModel.h"

#include <iostream>

#include "lbfgs_codes.h"

#include <omp.h>
#define USE_OMP

#define DEBUG_STRING "TEM: "
#define DEBUG_LEVEL 0
#include "debug.h"

using std::cout;
using std::endl;
using arma::zeros;
using std::vector;

typedef TemporallyExtendedModel::action_t action_t;
typedef TemporallyExtendedModel::observation_t observation_t;
typedef TemporallyExtendedModel::reward_t reward_t;

// some helper macros and functions

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

TemporallyExtendedModel::DataPoint::DataPoint(action_t action,
                                              observation_t observation,
                                              reward_t reward):
    action(action),
    observation(observation),
    reward(reward)
{}

bool TemporallyExtendedModel::DataPoint::operator<(const DataPoint & other) const {
    if(action<other.action) return true;
    if(action>other.action) return false;
    if(observation<other.observation) return true;
    if(observation>other.observation) return false;
    if(reward<other.reward) return true;
    if(reward>other.reward) return false;
    return false;
}

TemporallyExtendedModel & TemporallyExtendedModel::set_data(const data_t & data_) {
    DEBUG_OUT(1,"Set data");
    DEBUG_INDENT;
    data = data_;
    // update unique values
    unique_actions.clear();
    unique_observations.clear();
    unique_rewards.clear();
    for(auto & point : data) {
        unique_actions.insert(point.action);
        unique_observations.insert(point.observation);
        unique_rewards.insert(point.reward);
    }
    // debug output
    IF_DEBUG(3) {
        DEBUG_OUT(3,"Unique actions");
        {
            DEBUG_INDENT;
            for(auto & action : unique_actions) {
                DEBUG_OUT(3, action);
            }
        }
        DEBUG_OUT(3,"Unique observations");
        {
            DEBUG_INDENT;
            for(auto & observation : unique_observations) {
                DEBUG_OUT(3, observation);
            }
        }
        DEBUG_OUT(3,"Unique rewards");
        {
            DEBUG_INDENT;
            for(auto & reward : unique_rewards) {
                DEBUG_OUT(3, reward);
            }
        }
    }
    // resize outcome indices
    outcome_indices.assign(data.size(),-1);
    return *this;
}

double TemporallyExtendedModel::optimize() {
    DEBUG_OUT(1,"PULSE optimization");
    DEBUG_INDENT;

    // return value after optimization
    double likelihood = 0;

    // outer optimization loop
    for(int outer_loop_iteration=1;
        (max_outer_loop_iterations<=0 || outer_loop_iteration<=max_outer_loop_iterations);
        ++outer_loop_iteration) {
        DEBUG_OUT(2,"Iteration " << outer_loop_iteration);
        DEBUG_INDENT;

        // expand --> optimize --> shrink
        expand_feature_set();
        double new_likelihook = optimize_weights();
        shrink_feature_set();

        // checking terminal conditions
        if((new_likelihook-likelihood)/likelihood<likelihood_threshold) {
            likelihood = new_likelihook;
            break;
        } else {
            likelihood = new_likelihook;
        }
    }

    return likelihood;
}

double TemporallyExtendedModel::get_prediction(const data_t & pred_data) const {
    DEBUG_OUT(5,"Computing prediction");
    DEBUG_EXPECT(pred_data.size()>0);
    // temporally add the given observation and reward to the unique sets in
    // case they did not occur in the training data
    auto unique_observations_copy = unique_observations;
    auto unique_rewards_copy = unique_rewards;
    unique_observations_copy.insert(pred_data.back().observation);
    unique_rewards_copy.insert(pred_data.back().reward);
    // comput F-matrix
    mat_t F = zeros<mat_t>(feature_set.size(),
                           unique_observations_copy.size()*unique_rewards_copy.size());
    int outcome_idx;
    fill_F_matrix(feature_set,
                  unique_actions,
                  unique_observations_copy,
                  unique_rewards_copy,
                  pred_data,
                  pred_data.size()-1,
                  F,
                  outcome_idx);
    DEBUG_EXPECT(outcome_idx>=0);
    // get weights
    col_vec_t w;
    {
        w.zeros(feature_set.size());
        int feature_idx = 0;
        for(auto & feature : feature_set) {
            w(feature_idx) = feature.second;
            ++feature_idx;
        }
    }
    // interim variables
    const row_vec_t lin = w.t()*F;
    const row_vec_t exp_lin = arma::exp(lin);
    const double z = arma::sum(exp_lin);
    return exp_lin(outcome_idx)/z;
}

double TemporallyExtendedModel::optimize_weights() {
    DEBUG_OUT(3,"Optimizting weights");
    DEBUG_INDENT;
    // update F-matrices
    update_F_matrices();
    // optimize weights
    lbfgsfloatval_t objective_value;
    {
        DEBUG_OUT(4,"optimize");
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
        IF_DEBUG(2) {cout << endl;}
        DEBUG_OUT(2,"status code = " << ret << " ( " << lbfgs_code(ret) << " )");
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

bool TemporallyExtendedModel::check_derivatives() {
    DEBUG_OUT(1,"Checking derivatives");
    DEBUG_INDENT;
    // initialize some vectors
    vector<double> weights, weights_copy, gradient(feature_set.size()), diffs(feature_set.size());
    for(auto & feature : feature_set) weights.push_back(feature.second);
    weights_copy = weights;
    // delta/epsilon
    double delta = 1e-5;
    double epsilon = 1e-5;
    // update F-matrices
    update_F_matrices();
    // compute numerical gradient by symmetric finite differences
    for(int dim=0; dim<(int)feature_set.size(); ++dim) {
        weights_copy[dim] = weights[dim] + delta;
        double plus = neg_log_likelihood(this,
                                         &(weights_copy.front()),
                                         &(gradient.front()),
                                         feature_set.size());
        weights_copy[dim] = weights[dim] - delta;
        double minus = neg_log_likelihood(this,
                                          &(weights_copy.front()),
                                          &(gradient.front()),
                                          feature_set.size());
        weights_copy[dim] = weights[dim];
        diffs[dim] = plus - minus;
    }
    // calculate gradient at center
    double objective_value = neg_log_likelihood(this,
                                                &(weights.front()),
                                                &(gradient.front()),
                                                feature_set.size());
    // compare numerical and analytical gradient accepting both a small absolute
    // difference as well as a small relative difference
    bool ok = true;
    DEBUG_OUT(2,"ERROR	dim	analytical	numerical	difference	weight	(fx=" << objective_value << ")");
    for(int dim=0; dim<(int)feature_set.size(); ++dim) {
        double grad = diffs[dim]/(2*delta);
        if(fabs(grad-gradient[dim])>epsilon &&
           fabs(grad-gradient[dim])/fabs(objective_value)>epsilon) {
            if(ok && DEBUG_LEVEL==1) {
                DEBUG_OUT(1,"ERROR	dim	analytical	numerical	difference	weight	(fx=" << objective_value << ")");
            }
            ok = false;
            DEBUG_OUT(1,"ERROR	" << dim << "	" << gradient[dim] << "	" << grad
                      << "	" << fabs(grad-gradient[dim]) << "	" << weights[dim]);
        } else {
            DEBUG_OUT(2,"	" << dim << "	" << gradient[dim] << "	" << grad
                      << "	" << fabs(grad-gradient[dim]) << "	" << weights[dim]);
        }
    }
    if(ok) DEBUG_OUT(1,"no errors");
    return ok;
}

void TemporallyExtendedModel::expand_feature_set() {
    // first make a copy of the initial feature set which remains unchanged during expansion
    auto initial_feature_set = feature_set;
    // initialize if feature set is empty expand otherwise
    if(feature_set.empty()) {
        // add simple basis features
        for(auto & action : unique_actions) {
            feature_set[feature_t({basis_feature_t(ACTION,0,action)})] = 0;
        }
        for(auto & observation : unique_observations) {
            feature_set[feature_t({basis_feature_t(OBSERVATION,0,observation)})] = 0;
        }
        for(auto & reward : unique_rewards) {
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
                for(auto & action : unique_actions) {
                    feature_t f = feature.first;
                    f.insert(basis_feature_t(ACTION,t_idx,action));
                    feature_set[f] = 0;
                }
                for(auto & observation : unique_observations) {
                    feature_t f = feature.first;
                    f.insert(basis_feature_t(OBSERVATION,t_idx,observation));
                    feature_set[f] = 0;
                }
                for(auto & reward : unique_rewards) {
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
    IF_DEBUG(6) {
        print_feature_set();
    }
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
    IF_DEBUG(6) {
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
    DEBUG_OUT(4,"update F-matrices");
    DEBUG_INDENT;
    F_matrices.assign(data.size(),
                      zeros<mat_t>(feature_set.size(),
                                   unique_observations.size()*unique_rewards.size()));
    int progress = 0;
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) collapse(1)
    #endif
    for(int data_idx=0; data_idx<(int)data.size(); ++data_idx) {
        DEBUG_OUT(6,"data point " << data_idx);
        DEBUG_INDENT;
        fill_F_matrix(feature_set,
                      unique_actions,
                      unique_observations,
                      unique_rewards,
                      data,
                      data_idx,
                      F_matrices[data_idx],
                      outcome_indices[data_idx]);
        DEBUG_EXPECT(outcome_indices[data_idx]>=0);
        #ifdef USE_OMP
        #pragma omp critical (TemporallyExtendedModel)
        #endif
        {
            ++progress;
            IF_DEBUG(4) {
                cout << "\r" << (100*progress)/data.size() << "%    " << std::flush;
                IF_DEBUG(6) cout << endl;
            }
        } // end critical
    } // end parallel
    IF_DEBUG(4) {
        IF_DEBUG(6);// nothing to do
        else cout << endl;
    }
}

void TemporallyExtendedModel::fill_F_matrix(const feature_set_t & feature_set,
                                            const std::set<int> & unique_actions,
                                            const std::set<int> & unique_observations,
                                            const std::set<double> & unique_rewards,
                                            const data_t & data,
                                            const int & data_idx,
                                            mat_t & F_matrix,
                                            int & matching_outcome_index) {
    matching_outcome_index = -1;
    int feature_idx = 0; // row index
    for(auto & feature : feature_set) {
        DEBUG_OUT(6,"Feature " << feature_idx);
        DEBUG_INDENT;
        int outcome_idx = 0; // column index
        for(auto & observation : unique_observations) {
            for(auto & reward : unique_rewards) {
                DEBUG_OUT(6,"Outcome " << outcome_idx
                          << " (" << observation << ", " << reward << ")");
                DEBUG_INDENT;
                // check for matching outcome index
                if(observation==data[data_idx].observation && reward==data[data_idx].reward)
                    matching_outcome_index = outcome_idx;
                // check basis features
                bool is_true = true;
                for(auto & basis_feature : feature.first) {
                    //-----------------------------------//
                    // all basis feature have to be true //
                    //-----------------------------------//
                    BASIS_FEATURE(tuple, type, time, value);
                    tuple = basis_feature;
                    DEBUG_EXPECT(time<=0);
                    DEBUG_OUT(6,"Basis feature " << basis_feature);
                    DEBUG_INDENT;
                    // is the required time index accessible?
                    if(data_idx+time<0) {
                        DEBUG_OUT(6,"time idx inaccessible");
                        is_true = false;
                        break;
                    }
                    // does the value match?
                    switch(type) {
                    case ACTION:
                        if(data[data_idx+time].action!=value) is_true = false;
                        break;
                    case OBSERVATION:
                        if(time==0 && observation!=value) is_true = false;
                        if(time!=0 && data[data_idx+time].observation!=value) is_true = false;
                        break;
                    case REWARD:
                        if(time==0 && reward!=value) is_true = false;
                        if(time!=0 && data[data_idx+time].reward!=value) is_true = false;
                        break;
                    }
                    // break
                    if(!is_true) {
                        DEBUG_OUT(6,"value mismatch");
                        break;
                    }
                }
                if(is_true) {
                    F_matrix(feature_idx,outcome_idx) = 1;
                    DEBUG_OUT(6,"true");
                } else {
                    DEBUG_OUT(6,"false");
                }
                ++outcome_idx;
            }
        }
        ++feature_idx;
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
    const col_vec_t w(weights,n); // do I really have to copy?!
    col_vec_t grad(gradient,n,false);

    // get instance and number of data points
    auto TEM_instance = (TemporallyExtendedModel*)instance;
    int data_n = TEM_instance->data.size();

    // initialize
    lbfgsfloatval_t neg_log_like = 0;
    grad.zeros(TEM_instance->feature_set.size());

    // sum over data points
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) collapse(1)
    #endif
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
        #ifdef USE_OMP
        #pragma omp critical (TemporallyExtendedModel)
        #endif
        {
            neg_log_like += obj_term;
            grad += grad_term;
        } // end critical
    } // end parallel

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
    IF_DEBUG(2) {
        IF_DEBUG(6) {
            cout << "Iteration " << iteration_nr
                 << ", likelihood = " << exp(-objective_value) << std::endl;
            cout << "weights:" << endl;
            for(int idx=0; idx<nr_variables; ++idx) cout << "    " << weights[idx] << endl;
        } else {
            cout << "\rIteration " << iteration_nr
                 << ", likelihood = " << exp(-objective_value) << "    " << std::flush;
        }
    }
    return 0;
}
