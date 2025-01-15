import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro
from pyro.contrib.gp.kernels import RBF, Linear, Constant, Periodic, Polynomial
from pyro.contrib.gp.models import GPRegression
from pyro.contrib.gp.kernels import Sum, Product
import pyro.distributions as dist
import random

import scipy.linalg
import scipy.spatial
import scipy.optimize as opt

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    #pyro.set_rng_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def g(x):
    # defined on the domain x in [0,1]
    assert 0 <= x <= 1
    return -(np.sin(6*np.pi*x))**2 + 6*x**2 - 5*x**4 + 3/2

def sample_kernel_from_prior(theta_ranges, theta=None):

    # Define defaults or sampled values
    rbf_variance_dist = dist.Uniform(torch.tensor(theta_ranges[0][0] - 1e-4), torch.tensor(theta_ranges[0][1] + 1e-4))
    # if theta is not None:
    #     rbf_variance = theta[0]
    # else:
    rbf_variance = pyro.sample("rbf_variance", rbf_variance_dist)
    
    rbf_lengthscale_dist = dist.Uniform(torch.tensor(theta_ranges[1][0] - 1e-4), torch.tensor(theta_ranges[1][1] + 1e-4))
    
    # if theta is not None:
    #     rbf_lengthscale = theta[1]
    # else:
    rbf_lengthscale = pyro.sample("rbf_lengthscale", rbf_lengthscale_dist)
    

    # rbf_variance = (pyro.sample("rbf_variance", dist.LogNormal(torch.tensor(-0.5), torch.tensor(0.5))))

    # rbf_lengthscale = (pyro.sample("rbf_lengthscale", dist.LogNormal(torch.tensor(0.0), torch.tensor(0.5))))


    #rbf_lengthscale = (poly_offset_optim if poly_offset_optim is not None 
                       #else pyro.sample("rbf_lengthscale", dist.LogNormal(torch.tensor(0.0), torch.tensor(0.5))))
    
    # Hardcoded values for periodic kernel
    periodic_amplitude = torch.tensor(1.0)
    period = torch.tensor(1/6)
    lengthscale = torch.tensor(0.5)

    # Define kernels
    k1 = Periodic(
        input_dim=1, 
        period=period,
        variance=periodic_amplitude,
        lengthscale=lengthscale
    )

    k2 = RBF(
        input_dim=1,
        variance=rbf_variance,  # From argument or prior
        lengthscale=rbf_lengthscale  # From argument or prior
    )

    

    return Sum(k1, k2), [rbf_variance_dist, rbf_lengthscale_dist]



def neg_log_likelihood(X,y, params, theta_ranges):
    """
    Negative log likelihood function using only PyTorch operations
    -logp(y|X,theta)
    """
    noise_y = params[0]
    l = X.shape[0]
    
    K_theta_initialization, _ = sample_kernel_from_prior(theta_ranges[1:], params[1:])
    K_theta = K_theta_initialization(X, X)
    
    K = K_theta + noise_y * torch.eye(l)
    
    try:
        L = torch.linalg.cholesky(K + 1e-6 * torch.eye(l))
    except RuntimeError:
        print("RuntimeError")
        return torch.tensor(1e10, dtype=torch.float64)
        
    alpha = torch.triangular_solve(
        torch.triangular_solve(y.unsqueeze(1), L, upper=False)[0],
        L.T,
        upper=True
    )[0].squeeze()

    # print(y, alpha)
    
    data_fit = -(1/2) * torch.dot(y, alpha)
    complexity_penalty = -(1/2) * torch.sum(torch.log(torch.diagonal(L)))
    norm_constant = -(l/2) * torch.log(torch.tensor(2*torch.pi))

    print(data_fit, complexity_penalty, norm_constant)
    
    nll = -(data_fit + complexity_penalty + norm_constant)
    
    return nll.detach().item()  # Convert to Python scalar


def neg_log_joint_conditional_probability(params, X, y, theta_priors, theta_ranges):
    """
    -logP(y,theta|X)=-logP(y|theta,X) - p(theta) proportional to log posterior: p(theta|x,y) the quantity we min/max to obtain theta*
    """
    params_tensor = torch.tensor(params, dtype=torch.float64)
    nll = neg_log_likelihood(X, y, params_tensor, theta_ranges)
    
    # Compute prior terms
    prior_terms = 0
    for param_i, prior_i in zip(params, theta_priors):
        #print(prior_i, param_i, -prior_i.log_prob(torch.tensor(param_i)))
        prior_terms += -prior_i.log_prob(torch.tensor(param_i))
        
    return nll + prior_terms

def optimize_params(X_train, y_train, theta_priors, ranges, Ngrid):
    """
    Optimize parameters using grid search with PyTorch
    """
    
    def objective(params):
        return neg_log_joint_conditional_probability(params, X_train, y_train, theta_priors, ranges)
    

     # Create explicit grid points
    noise_points = np.linspace(ranges[0][0], ranges[0][1], Ngrid)
    rbf_var_points = np.linspace(ranges[1][0], ranges[1][1], Ngrid)
    rbf_offset_lengthscale = np.linspace(ranges[2][0], ranges[2][1], Ngrid)
    

    # Keep track of top 3 using a list of (value, params) tuples
    top_3 = [(float('inf'), None), (float('inf'), None), (float('inf'), None)]
    
    # Exhaustive search
    for noise in noise_points:
        for rbf_var in rbf_var_points:
            for rbf_offset in rbf_offset_lengthscale:
                params = [noise, rbf_var, rbf_offset]
                val = objective(params)
                
                # Update top 3 if needed
                if val < top_3[-1][0]:  # Better than worst of top 3
                    top_3[-1] = (val, params)
                    # Sort by value (first element of tuple)
                    top_3.sort(key=lambda x: x[0])
    
    print("\nTop 3 parameter combinations:")
    for i, (val, params) in enumerate(top_3, 1):
        print(f"{i}. params={params}, value={val:.3f}")
    
    # Return the best parameters as before
    return [torch.tensor(top_3[0][1][0], dtype=torch.float32)], \
           [torch.tensor(top_3[0][1][1], dtype=torch.float32), 
            torch.tensor(top_3[0][1][2], dtype=torch.float32)]





def conditional(X_train, X_test, y_train, theta_star, theta_ranges):
    #set_seed(42)

    noise_var_star = theta_star[0]
    
    # Reshape inputs to 2D tensors if they aren't already
    X_train = X_train.reshape(-1, 1) if len(X_train.shape) == 1 else X_train
    X_test= X_test.reshape(-1, 1) if len(X_test.shape) == 1 else X_test
    y_train = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train  # Added this line
    K_theta_initialization_star, _ = sample_kernel_from_prior(theta_ranges[1:], theta_star[1:])

    # Compute kernel matrices
    k_xtrain_xtrain = K_theta_initialization_star(X_train, X_train)
    k_xtrain_xtest = K_theta_initialization_star(X_train, X_test)
    k_xtest_xtest = K_theta_initialization_star(X_test, X_test)
    noise_matrix = noise_var_star * torch.eye(len(y_train), device=X_train.device)
    K = k_xtrain_xtrain + noise_matrix
    
    # Compute posterior mean and covariance
    # Using torch.linalg.solve for better numerical stability than inverse
    L = torch.linalg.cholesky(K)
    alpha = torch.linalg.solve_triangular(L.T, 
                                        torch.linalg.solve_triangular(L, y_train, upper=False),
                                        upper=True)
    
    mu_test = k_xtrain_xtest.T @ alpha
    
    # Compute posterior variance using Cholesky for stability
    v = torch.linalg.solve_triangular(L, k_xtrain_xtest, upper=False)
    sigma_test = k_xtest_xtest - v.T @ v
    
    
    return mu_test.detach(), sigma_test.detach(), torch.diag(sigma_test.detach())





def posterior_log_likelihood(y_test, mu_test, sigma_test):
    """Compute the posterior log-likelihood of test data."""
    y_test = y_test.reshape(-1)
    mu_test = mu_test.reshape(-1)
    
    # Add small jitter for numerical stability
    sigma_test = sigma_test + 1e-6 * torch.eye(sigma_test.shape[0])
    
    try:
        # Use multivariate normal distribution from PyTorch
        dist = torch.distributions.MultivariateNormal(mu_test, sigma_test)
        log_likelihood = dist.log_prob(y_test)
        return log_likelihood.item()
    except:
        print("Exception")
        return -float('inf')
    

def run_inference(X_train, y_train, X_test, y_test, Ngrid, i):

    # Preparing priors
    

    # Search parameter ranges
    theta_ranges = [
        (0.001, 0.1),    # noise_var_range
        (0.001, 2.0),    # poly_var_range
        (0.001, 2.0)     # poly_offset_range
    ]

    _, kernel_priors = sample_kernel_from_prior(theta_ranges=theta_ranges[1:])
    noise_prior = [dist.Uniform(torch.tensor(theta_ranges[0][0] - 1e-4), 
                                torch.tensor(theta_ranges[0][1] + 1e-4))]
    theta_priors = noise_prior + kernel_priors

    # Run optimization
    noise_var_star, kernel_param_star=  optimize_params(X_train, y_train, theta_priors, theta_ranges, Ngrid)
    theta_star = noise_var_star + kernel_param_star #[param1,param2,param3]
    mu_test, sigma_test, gp_posterior_var = conditional(X_train, X_test, y_train, theta_star, theta_ranges)
    log_likelihood_i = posterior_log_likelihood(y_test, mu_test, sigma_test)


    
    print(f"Seed: {seed}, Log-Likelihood: {log_likelihood_i:.3f}, Best Params: {theta_star}")
    plt.title("Performance on D_test")
    #training
    plt.plot(X_train, y_train, label="training function")
    plt.scatter(X_train ,y_train, label="training points")
    #ground truth
    plt.plot(X_test,y_test,label="ground truth function")
    plt.scatter(X_test,y_test,label="ground truth points")
    #prediction on test
    plt.plot(X_test, mu_test, label="mu_test")
    plt.fill_between(X_test.squeeze().detach().numpy(), 
                        (mu_test.squeeze() - 2 * torch.sqrt(gp_posterior_var)).detach().numpy(), 
                        (mu_test.squeeze() + 2 * torch.sqrt(gp_posterior_var)).detach().numpy(), 
                        alpha=0.2, zorder=2,label="mu_test +- 2 std")
    plt.legend()
    plt.savefig(rf"B_test_images\map\map_{i}bug_fix_rbf_loglike_{int(log_likelihood_i)}.png")
    plt.close()


    return log_likelihood_i




def initialize_dataset():
    # Generate data
    l = 30
    X_grid = np.array([(i-1)/(l-1) for i in range(1, l+1)])
    mu, var = 0, 0.01
    eps_grid = np.random.normal(loc=mu, scale=np.sqrt(var), size=l)
    y_grid = np.array([g(xi) + eps for xi, eps in zip(X_grid, eps_grid)])

    # Split into train/test
    all_indices = np.arange(30)
    train_indices = np.sort(np.random.choice(all_indices, size=20, replace=False))
    test_indices = np.setdiff1d(all_indices, train_indices)

    X_train = torch.from_numpy(X_grid[train_indices]).float().reshape(-1, 1)
    y_train = torch.from_numpy(y_grid[train_indices]).float()
    X_test = torch.from_numpy(X_grid[test_indices]).float().reshape(-1, 1)
    y_test = torch.from_numpy(y_grid[test_indices]).float()
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    posterior_log_likelihoods = []
    for seed in range(20):
        set_seed(seed)
        X_train, y_train, X_test, y_test = initialize_dataset()
        log_likelihood_i = run_inference(X_train, y_train, X_test, y_test, Ngrid=20, i = seed)
        posterior_log_likelihoods.append(log_likelihood_i)
        
        print(log_likelihood_i)

    
    mean_posterior_log_likelihood = np.mean(posterior_log_likelihoods)
    std_posterior_log_likelihood = np.std(posterior_log_likelihoods)
    print(f"mean_posterior_log_likelihood: {mean_posterior_log_likelihood}")
    print(f"std_posterior_log_likelihood: {std_posterior_log_likelihood}")
    print(posterior_log_likelihoods)


