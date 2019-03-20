import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from scipy.stats import norm, uniform, multivariate_normal
from statsmodels.regression.linear_model import yule_walker
from matplotlib import cm
import os
my_path = '/fslhome/fslcollab151/convergence_mcmc/Results/10000x2xy'
print(my_path)
class diagnostics:
    def __init__(self, samples):
        self.samples = samples
        self.m = len(samples)
        self.n = len(samples[0])
    def Gelman_Rubin(self):
        B_over_n = np.sum((np.mean(self.samples, axis = 1)-np.mean(self.samples))**2)/(self.m-1) #between-chains variances 
        W = np.sum(
        [(self.samples[i] - xbar) ** 2 for 
         i, xbar in enumerate(np.mean(self.samples, axis=1))]) / (self.m * (self.n - 1))
        s2 = W * (self.n - 1) / self.n + B_over_n # (over) estimate of variance
        V = s2 + B_over_n / self.m # Pooled posterior variance estimate
        R = V / W # Calculate PSRF
        print("Gelman-Rubin:", R)
    def geweke(self):
        def spec(x, order=2):
            beta, sigma = yule_walker(x, order)
            return sigma**2 / (1. - np.sum(beta))**2
        first=.1
        last=.5
        intervals = 20
        for j in range(len(self.samples)):
            x = self.samples[j]
            zscores = [None] * intervals  # Initialize list of z-scores
            starts = np.linspace(0, int(len(x)*(1.-last)), intervals).astype(int) # Starting points for calculations
            # Loop over start indices
            for i,s in enumerate(starts):
                # Size of remaining array
                x_trunc = x[s:]
                n = len(x_trunc)
                # Calculate slices
                first_slice = x_trunc[:int(first * n)]
                last_slice = x_trunc[int(last * n):]
                z = np.mean(first_slice) - np.mean(last_slice)
                z /= np.sqrt(spec(first_slice)/len(first_slice) +
                             spec(last_slice)/len(last_slice))
                zscores[i] = len(x) - n, z

            print("the geweke result of chain {} is: {}".format(j + 1, zscores))

para = (10.0, 28, 8.0/3.0)

# solve ODE
def f(state, t, para):
  x, y, z = state  # unpack the state vector
  sigma, rho, beta = para 
  return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # derivatives

state0 = [1.0, 1.0, 1.0]
t = np.arange(0.0, 40.0, 0.01)
states = odeint(f, state0, t, args = (para,))
x_true, y_true, z_true = np.mean(states, axis=0)
xy_true = np.mean(states[0]*states[1])
x2_true = np.mean(states[0]*states[0])

# uniform prior distribution of parameters
def pr(x):
    return uniform.pdf(x[0], 0, 30) * uniform.pdf(x[1], 0, 40) *uniform.pdf(x[2], 0, 10) 

#likelyhood of parameters given 1obeserbations-x
def li(x):
    para = x
    states = odeint(f, state0, t, args = (para,))
    u0=np.mean(states[0]*states[0])
    u1=np.mean(states[0]*states[1])    
    return norm.pdf(u0, x2_true, 1.0) * norm.pdf(u1, xy_true, 1.0) 


# plot the solutions of E-L ODE
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(states[:,0], states[:,1], states[:,2])
plt.show();


n =10000# number of samples for each variable
print(n)
all_samples = []

# j is the index of chains
# i is the index of the iterations of each chain

for j in range(10):
    samples = [np.random.uniform(0, 20, 3)] 
    #print(samples[0])
    gamma0 = np.array([[1,0,0], [0,1,0], [0,0,1]])

    for i in range(n):
        #gamma = gamma0 # without big step length
        x_current = samples[-1]
        v = np.random.uniform(0, 1)
        gamma = gamma0 if v > 0.1 else 10 * gamma0
        w = np.random.multivariate_normal([0,0,0], gamma)
        x_tuta = samples[-1] + w
        alpha = min(1, pr(x_tuta)*li(x_tuta)/(pr(x_current)*li(x_current)))
        u = np.random.uniform(0, 1)
        x_next = x_tuta if u < alpha else samples[-1]
        samples.append(x_next)
        
    samples = np.transpose(samples)
    all_samples.append(samples)
    x, y, z = samples[0], samples[1], samples[2]
    print("Accept ratio:", len(set(x))/len(x))
    fig1 = plt.figure()
    plt.hist(x, bins=100, density=True)
    myfile_x = 'x'+str(j) 
    fig1.savefig(os.path.join(my_path, myfile_x)) 
    plt.close()
    
    fig2 = plt.figure()
    plt.hist(y, bins=100, density=True)
    myfile_y = 'y'+str(j) 
    fig2.savefig(os.path.join(my_path, myfile_y))
    plt.close()
    
    fig3 = plt.figure()
    plt.hist(z, bins=100, density=True)
    myfile_z = 'z'+str(j) 
    fig3.savefig(os.path.join(my_path, myfile_z))
    plt.close()
    
    
    
all_samples_x = []
all_samples_y = []
all_samples_z = []

for i in range(len(all_samples)):
    all_samples_x.append(all_samples[i][0])
    all_samples_y.append(all_samples[i][1])
    all_samples_z.append(all_samples[i][2])
np.savetxt('x2xy_x.out',all_samples_x, delimiter=',')
np.savetxt('x2xy_y.out',all_samples_y, delimiter=',')
np.savetxt('x2xy_z.out',all_samples_y, delimiter=',')


# call diagnostics
diagno1 = diagnostics(all_samples_x)
diagno1.Gelman_Rubin()    
diagno1.geweke()
diagno2 = diagnostics(all_samples_y)
diagno2.Gelman_Rubin()
diagno2.geweke()
diagno3 = diagnostics(all_samples_z)
diagno3.Gelman_Rubin()
diagno3.geweke()
all_samples_x = np.asarray(all_samples_x)
all_samples_y = np.asarray(all_samples_y)
all_samples_z = np.asarray(all_samples_z)
# the diagonestic result when set burn in = 5000

all_samples_xb = all_samples_x[:, 5000:]
all_samples_yb = all_samples_y[:, 5000:]
all_samples_zb = all_samples_z[:, 5000:]
"""
for j in range(len(all_samples_x)):
    x, y, z = all_samples_x[j], all_samples_y[j], all_samples_z[j] 
    plt.hist(x, bins=100, density=True)
    myfile_x = 'x'+str(j)+'bi' 
    plt.savefig(os.path.join(my_path, myfile_x))   
    plt.hist(y, bins=100, density=True)
    myfile_y = 'y'+str(j)+'bi' 
    plt.savefig(os.path.join(my_path, myfile_y))
    plt.hist(z, bins=100, density=True)
    myfile_z = 'z'+str(j)+'bi'
    plt.savefig(os.path.join(my_path, myfile_z))
"""
diagno1 = diagnostics(all_samples_xb)
diagno1.Gelman_Rubin()    
diagno1.geweke()
diagno2 = diagnostics(all_samples_yb)
diagno2.Gelman_Rubin()    
diagno2.geweke()
diagno3 = diagnostics(all_samples_zb)
diagno3.Gelman_Rubin()    
diagno3.geweke()

    
