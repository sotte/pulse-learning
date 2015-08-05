#ifndef TEMPORALLY_EXTENDED_MODEL_H_
#define TEMPORALLY_EXTENDED_MODEL_H_

#include "AbstractTemporallyExtendedModel.h"

#include <set>
#include <map>

/**
 * Learn a feature set and predictive Conditional Random Field model.
 *
 * We learn a model of the form
 *
 * \f{align}
 * p(\mathbf{y}|\mathbf{x}) &= \frac{1}{Z(\mathbf{x})} \exp \sum_{f\in\mathcal{F}} \theta_{f} f(\mathbf{x},\mathbf{y}) \\
 * Z(\mathbf{x}) &= \sum_{\mathbf{y}} \exp \sum_{f\in\mathcal{F}} \theta_{f} f(\mathbf{x},\mathbf{y})~,
 * \f}
 *
 * where \f$\mathbf{x}\f$ are past observations and the next action, while
 * \f$\mathbf{y}\f$ are the next obervation and reward that are to be predicted.
 * Given data
 * \f$\{(\mathbf{x}^{(1)},\mathbf{y}^{(1)}),\ldots,(\mathbf{x}^{(N)},\mathbf{y}^{(N)})\}\f$
 * we need to maximize the log-likelihood
 *
 * \f{align}
 * \ell(\boldsymbol{\mathbf{\theta}}) &= \log \prod_{n=1}^N p(\mathbf{y}^{(n)}|\mathbf{x}^{(n)}) \\
 * &=  \sum_{n=1}^N \sum_{f\in\mathcal{F}} \theta_{f} f(\mathbf{x}^{(n)},\mathbf{y}^{(n)}) -
 * \sum_{n=1}^N \log \sum_{\mathbf{y}} \exp \sum_{f\in\mathcal{F}} \theta_{f} f(\mathbf{x}^{(n)},\mathbf{y})
 * \f}
 *
 * with the gradient
 *
 * \f{align}
 * \frac{\partial\ell(\boldsymbol{\mathbf{\theta}})}{\partial\theta_{g}}
 * &= \sum_{n=1}^N \left[
 * g(\mathbf{x}^{(n)},\mathbf{y}^{(n)})
 * -
 * \frac{1}{Z(\mathbf{x}^{(n)})}
 * \sum_{\mathbf{y}}
 * g(\mathbf{x}^{(n)},\mathbf{y}) \,
 * \exp \sum_{f\in\mathcal{F}} \theta_{f} f(\mathbf{x}^{(n)},\mathbf{y})
 * \right]~.
 * \f}
 *
 * <b>Writing the Objective and Gradient in Matrix Form</b>
 *
 * If \f$\boldsymbol{\mathbf{\theta}}\f$ is the the vector of feature weights
 * and \f$\mathbf{f}(\mathbf{x},\mathbf{y})\f$ is the vector of feature values
 * for a given \f$\mathbf{x}\f$ and \f$\mathbf{y}\f$ we can write the linear
 * combination \f$\sum_{f\in\mathcal{F}} \theta_{f} f(\mathbf{x},\mathbf{y})\f$
 * as dot product of these vectors
 *
 * \f{align}
 * \sum_{f\in\mathcal{F}} \theta_{f} f(\mathbf{x},\mathbf{y}) &= \boldsymbol{\mathbf{\theta}}^{\top}\mathbf{f}(\mathbf{x},\mathbf{y})~.
 * \f}
 *
 * If we write the vectors \f$\mathbf{f}(\mathbf{x},\mathbf{y}_i)\f$ for all
 * possible values of \f$\mathbf{y}_i\f$ in a matrix
 *
 * \f{align}
 * \mathbf{F}(\mathbf{x})
 * &= \Big( \mathbf{f}(\mathbf{x},\mathbf{y}_{1}), \mathbf{f}(\mathbf{x},\mathbf{y}_{2}), \ldots \Big) \\
 * \text{or equivalently} \qquad \mathbf{F}(\mathbf{x})_{ij}
 * &= \mathbf{f}_{i}(\mathbf{x},\mathbf{y}_{j})
 * \f}
 *
 * the product \f$\boldsymbol{\mathbf{\theta}}^{\top}\mathbf{F}(\mathbf{x})\f$
 * is a row vector containing the linear combinations of features for all
 * possible values of \f$\mathbf{y}\f$ for a given specific value of
 * \f$\mathbf{x}\f$.
 *
 * In the partial derivative of the objective \f$\ell\f$ with respect to the
 * \f$i^{th}\f$ feature, the sum over \f$\mathbf{y}\f$ can then be written as
 *
 * \f{align}
 * \ldots &= \sum_{k}
 * f_{i}(\mathbf{x}^{(n)},\mathbf{y}_{k}) \,
 * \exp \sum_{j} \theta_{j} f_{j}(\mathbf{x}^{(n)},\mathbf{y}_{k}) \\
 * &= \sum_{k} \mathbf{F}(\mathbf{x}^{(n)})_{ik} \, \exp\! \left[\,
 * \boldsymbol{\mathbf{\theta}}^{\top} \mathbf{F}(\mathbf{x}^{(n)})_{k}
 * \,\right] \\
 * &= \left[
 * \mathbf{F}(\mathbf{x}^{(n)})
 * \exp\! \left[\,
 * \boldsymbol{\mathbf{\theta}}^{\top} \mathbf{F}(\mathbf{x}^{(n)})
 * \,\right]^{\top}
 * \right]_{i}~,
 * \f}
 *
 * where the \f$\exp\f$ is to be taken elementwise. So for a data point
 * \f$(\mathbf{x}^{(n)},\mathbf{y}^{(n)})\f$ given
 * \f$\mathbf{F}=\mathbf{F}(\mathbf{x}^{(n)})\f$ and \f$y^{*}\f$ being the index
 * of \f$\mathbf{y}^{(n)}\f$ we can compute
 * \f$\ell(\boldsymbol{\mathbf{\theta}})\f$ and
 * \f$\nabla\ell(\boldsymbol{\mathbf{\theta}})\f$ as follows
 *
 * \f{align}
 * \mathtt{lin} &= \boldsymbol{\mathbf{\theta}}^{\top}\mathbf{F} \\
 * \mathtt{explin} &= \exp \left( \mathtt{lin} \right) \\
 * \mathtt{z} &= \sum_{i} \mathtt{explin}_{i} \\
 * \ell(\boldsymbol{\mathbf{\theta}}) &= \mathtt{lin}_{y^{*}} - \log(\mathtt{z}) \\
 * \nabla\ell(\boldsymbol{\mathbf{\theta}}) &= \left(\mathbf{F}^{\top}\right)_{y^{*}} -
 * \frac{
 * \mathbf{F} \,
 * \mathtt{explin}^{\top}
 * }{
 * \mathtt{z}
 * }
 * \f}
 *
 * where \f$\left(\mathbf{F}^{\top}\right)_{i}\f$ denotes the
 * \f$i^{th}\f$ column of \f$\mathbf{F}\f$. The sum over data
 * points can then be computed in parallel.
 *
 * <b>Partial Derivatives of New Features</b>
 *
 * All features from a new set \f$\widetilde{\mathcal{F}}\f$ are included with
 * an initial weight of zero. The vector \f$\mathtt{lin} =
 * \boldsymbol{\mathbf{\theta}}^{\top}\mathbf{F}\f$ (containing
 * all linear combinations for the different values of
 * \f$\mathbf{y}\f$) thus does not change. Correspondingly,
 * \f$\mathtt{explin}\f$ and \f$\mathtt{z}\f$ do not change either. To compute
 * the gradient for the new features only, the corresponding feature matrix
 *
 * \f{align}
 * \widetilde{\mathbf{F}}(\mathbf{x})_{ij} &= \widetilde{\mathbf{f}}_{i}(\mathbf{x},\mathbf{y}_{j})
 * \f}
 *
 * has to be calculated (excluding the old features). The gradient then is (as
 * above)
 *
 * \f{align}
 * \nabla\ell(\boldsymbol{\mathbf{\theta}}_{new}) &= \left(\widetilde{\mathbf{F}}^{\top}\right)_{y^{*}} -
 * \frac{
 * \widetilde{\mathbf{F}} \,
 * \mathtt{explin}^{\top}
 * }{
 * \mathtt{z}
 * }
 * \f}
 *
 * where \f$\widetilde{\mathbf{F}}\f$ is computed from the new
 * features and \f$\mathtt{explin}\f$ and \f$\mathtt{z}\f$ are computed from
 * the old features.
 */
class TemporallyExtendedModel: public AbstractTemporallyExtendedModel {
protected:
    enum FEATURE_TYPE { ACTION, OBSERVATION, REWARD };
    typedef std::tuple<FEATURE_TYPE,int,double> basis_feature_t;
    typedef std::set<basis_feature_t> feature_t;
    typedef std::map<feature_t,double> feature_set_t;
public:
    TemporallyExtendedModel() = default;
    virtual ~TemporallyExtendedModel() = default;
    virtual AbstractTemporallyExtendedModel & set_regularization(double d) override {regularization=d;return *this;}
    virtual AbstractTemporallyExtendedModel & set_data(const data_t &) override;
    virtual AbstractTemporallyExtendedModel & set_horizon_extension(int n) override {horizon_extension=n;return *this;}
    virtual AbstractTemporallyExtendedModel & set_maximum_horizon(int n) override {maximum_horizon=n;return *this;}
    virtual double optimize() override;
    virtual double get_prediction(const data_t & data) override;
    virtual AbstractTemporallyExtendedModel & set_gradient_threshold(double d) override {gradient_threshold=d;return *this;}
    virtual AbstractTemporallyExtendedModel & set_parameter_threshold(double d) override {parameter_threshold=d;return *this;}
    virtual AbstractTemporallyExtendedModel & set_max_inner_loop_iterations(int n) override {max_inner_loop_iterations=n;return *this;}
    virtual AbstractTemporallyExtendedModel & set_max_outer_loop_iterations(int n) override {max_outer_loop_iterations=n;return *this;}
    virtual AbstractTemporallyExtendedModel & set_likelihood_threshold(double d) override {likelihood_threshold=d;return *this;}
protected:
    double regularization = 0;
    int horizon_extension = 1;
    int maximum_horizon = -1;
    double  gradient_threshold = 1e-10;
    double parameter_threshold = 1e-10;
    int max_inner_loop_iterations = -1;
    int max_outer_loop_iterations = -1;
    double likelihood_threshold = 1e-10;
    data_t data;
    std::set<int> unique_actions;
    std::set<int> unique_observations;
    std::set<double> unique_rewards;
    feature_set_t feature_set;

    void expand_feature_set();
    double optimize_feature_weights();
    void shrink_feature_set();
    void print_feature_set();
};

#endif /* TEMPORALLY_EXTENDED_MODEL_H_ */
