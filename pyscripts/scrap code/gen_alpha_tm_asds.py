# By Bart Ouwehand 12-04-2022

from my_functions import *

tight_range = False


# Defining the model used for MCMC fitting
def model(st,Amp0,Afunc,Efunc,s,Amp=Amp_true, phi0=phi0_true, freq=f_true, gw_beta=gw_beta_true, gw_lambda=gw_lambda_true, t0=orbits_t0+1/fs,orbits_ZP=0):
    
    # Create random filename to allow for multiprocessing
    gwfn = ext2+'gwtmp_'+str(int(1e20*np.random.rand(1)[0]))+'.h5'
    
    # Amp, phi0 = theta[0:Ngalbins], theta[Ngalbins:2*Ngalbins]
    
    # Generate GW signals
    skip = [0]
    for a, f, p, beta, lamb,i in zip(Amp, freq, phi0, gw_beta, gw_lambda,range(Ngalbins)):
        if i in skip:
            GalBin = GalacticBinary(A=Amp0/f, f=f, phi0=p, orbits=orbit_path, t0=t0, gw_beta=beta-orbits_ZP, gw_lambda=lamb, dt=1/fs, size=s+300)
        else:
            GalBin = GalacticBinary(A=a/f, f=f, phi0=p, orbits=orbit_path, t0=t0, gw_beta=beta-orbits_ZP, gw_lambda=lamb, dt=1/fs, size=s+300)
            # Amplitude is a/f since we have to convert from_gws to from_instr by differentiating and don't want extra factors of 2pi*f
        GalBin.write(gwfn)
    
    # rawdata = Data.from_gws( 'gw_tmp.h5', orbit_path)
    rawdata = Data.from_gws(gwfn, orbit_path,interpolate=True)
    mA = Afunc(rawdata.measurements)[discard:]
    mE = Efunc(rawdata.measurements)[discard:]
    # If we only fit signal parameters, we don't include T since it has by definition no signal.
    #T = Tfunc(rawdata.measurements)[discard:]
    
    mt = GalBin.t[discard:]
    
    os.remove(gwfn)
    
    # Generate correct amplitude to be compatible with sample data
    nmt = np.copy(mt)[:-1]
    nmA = dphi_to_dnu(fs,mA)
    nmE = dphi_to_dnu(fs,mE)

    # Make sure that the model generates data at the correct time
    time_indices = np.where(np.in1d(nmt, st))[0]
    nnmt, nnmA, nnmE = nmt[time_indices], nmA[time_indices], nmE[time_indices]
    
    return np.array([nnmt,nnmA,nnmE])
    # return np.array([t,A,E,T])


# Define the likelyhood functions
def lnL(theta, t, y1, y2,Afunc,Efunc,s):
    """
    The log likelihood of our simplistic Linear Regression. 
    """
    # Amp, f, phi0 = theta
    # beta_ZP, Amp = theta[0], theta[1:]
    # Amp_lnL, phi0_lnL = theta[:Ngalbins], theta[Ngalbins:2*Ngalbins]
    beta_ZP_lnL = np.copy(theta)
    
    # newt, y1_model, y2_model = model(t, Amp_lnL,phi0_lnL)
    newt, y1_model, y2_model = model(t, beta_ZP_lnL,Afunc,Efunc,s)
    
    return 0.5*(np.sum((y1-y1_model)**2)) + 0.5*(np.sum((y2-y2_model)**2))

def lnprior(theta):
    """
    Define a gaussian prior that preferences values near the observed values of the Galactic binaries     
    """
    # Amp, phi0 = theta[:Ngalbins], theta[Ngalbins:2*Ngalbins]
    
    # beta_ZP, Amp = theta[0], theta[1:]
    # Amp_lnprior, phi0_lnprior = theta[:Ngalbins], theta[Ngalbins:2*Ngalbins]
    beta_ZP_lnprior = np.copy(theta)
    
    # if (int(np.sum((1e-26 < Amp)*(Amp<1e-20))) == Ngalbins):# and -np.pi <= beta_ZP <= np.pi:
    # if (int(np.sum((1e-26 < Amp_lnprior)*(Amp_lnprior<1e-20))) == Ngalbins) and int(np.sum((-np.pi <= phi0_lnprior)* (phi0_lnprior<= np.pi))) == Ngalbins:
    if -np.pi <= beta_ZP_lnprior <= np.pi:
        return 0
    return np.inf
    
    # if 1e-18 < Amp < 1e-14:
    #     return gauss_prior(f, obs_q[1]*obs_qe[1],obs_q[1])
    # return -np.inf
    # if int(np.sum((-np.pi <= phi0) * (phi0 <= np.pi))) == Ngalbins:
    #     return np.sum(gauss_prior(Amp,Amp_prior[1],Amp_prior[0])) #gauss_prior(Amp, obs_q[0]*obs_qe[0], obs_q[0]) + gauss_prior(f, obs_q[1]*obs_qe[1],obs_q[1])
    # return -np.inf
    
def lnprob(theta, t, y1, y2,Afunc,Efunc,s):
    """
    The likelihood to include in the MCMC.
    """
    # global iters
    # iters +=1
    
    lp = lnprior(theta)
    if not np.isfinite(lp):
        # print (iters,'infty')
        return np.inf
    lnlikely = lp + lnL(theta,t,y1, y2,Afunc,Efunc,s)
    # print (iters,lnlikely)
    return lnlikely

# Define the parabula functions

def sig(x0,xpm,N,f0,fpm):
    return xpm / np.sqrt(2*np.log((N-f0) / (N-fpm)))

def parabula_err(x0, xpm, fsdata,L_on_range,f0,Afunc,Efunc,s):
    """Finds error of likelyhood parabula. Input must be the optimal value and the offset from this for which to calculate the parabula. The output is sigma"""
    med = np.median(L_on_range)
    # f0 = lnprob([x0],fsdata[0],fsdata[1], fsdata[2])
    fp, fm = lnprob([x0+xpm],fsdata[0],fsdata[1], fsdata[2],Afunc,Efunc,s),lnprob([x0-xpm],fsdata[0],fsdata[1], fsdata[2],Afunc,Efunc,s)
    # print (med, f0, fp, fm)
    return np.nanmean([sig(x0,xpm,med,f0,fp), sig(x0,xpm,med,f0,fm)])

def parabula(x0, L_on_range,f0,sig):
    N = np.median(L_on_range)
    # A = lnprob([x0],fsdata[0],fsdata[1], fsdata[2]) - N
    length = 2000
    xrange = np.linspace(-np.pi,np.pi,length)
    return xrange, N - (N-f0)*np.exp(-((xrange-x0)**2)/(2*(sig**2)))
    # return xrange, N + (f0-N)*np.exp(((xrange-x0)**2)/(sig**2))

# if tight_range:
#     zp_range = np.linspace(-0.25,0.25,N2)
# else:
#     zp_range = np.pi*np.linspace(-1,1,N2)
Amp0_range = Amp_true[0]*np.linspace(0.5,1.5,31)

calculate_again = True
sigmas = np.zeros((len(alpha),N1))
# for i,alph in enumerate(alpha):
for d,dr in enumerate(dur_range):
    # Build the TDI chanels for model data
    Afunc, Efunc = BuildModelTDI(orbit_path,fs,size[d],Amp_true, f_true, phi0_true, gw_beta_true, gw_lambda_true,discard,detailed=True)

    timing = time.time()
    for i,alph in enumerate(alpha):
        sigmas_1a = np.zeros(N1)
        # for j in tqdm(range(N)):
        fp = 'plots/'+str(dr)+'d/'+str(int(alph))+'/'
        L_range_1a = np.zeros((N1,N2))
        for j in range(N1):
            print('RAM memory used {} d, alpha {}, iter {}: {} %'.format(dr,alph,j,psutil.virtual_memory()[2]))
            inputf = ext+'measurements/tm_asds/'+str(int(dr))+'d/'+str(int(alph))+"/fs"+str(j)

            rawdata = ascii.read(inputf+'.txt')
            fsdata = np.array([rawdata['t'],rawdata['A'],rawdata['E']])

            # Generate model data, calculate likelyhood for entire range, and calculate error in region
            # print ("Calculating likelyhood range")
            L_range = np.zeros(N2)
            for l,Amp0 in enumerate(tqdm(Amp0_range)):
                L_range[l] = lnprob([Amp0],fsdata[0],fsdata[1], fsdata[2],Afunc,Efunc,size[d])

            L_range_1a[j] = L_range

            sigma = parabula_err(0,0.01, fsdata, L_range, L_range[N2//2],Afunc,Efunc,size[d])
            # x,y = parabula(0,L_range, L_range[N2//2],sigma)
            # print ("Time to calculate likelyhood range = {:.2f} s / {:.3f} hrs".format(time.time()-L_range_time0,(time.time()-L_range_time0)/3600))

            sigmas_1a[j] = sigma
#             # print ("For alpha={}, sigma={:.2f}".format(alph,sigma))

            plt.figure(figsize=(8,6))
            # plt.plot(x,y,c='black',alpha=.8)
            plt.plot(Amp0_range,L_range,marker='o',ls='--',c='r')
            plt.xlabel("Amp0")
            plt.ylabel("-ln(L)")
            plt.title("Likelyhood range over ZP range for alpha {} and iteration {}".format(alph,j))
            # plt.ylim(np.min(L_range),np.max(L_range))
            plt.savefig(fp+"L_range_over_ZP_"+str(j)+".jpg")
            
            
        
        if tight_range:
            filename = ext+"measurements/tm_asds/"+str(int(dr))+"d/L_range_a"+str(int(alph))+"tight.txt"
        else:
            filename = ext+"measurements/tm_asds/"+str(int(dr))+"d/L_range_a"+str(int(alph))+".txt"
        # filecontent = Table(sdata.T, names=['t','A','E','T'])
        filecontent = Table(L_range_1a.T, names=[j for j in range(N1)])
        ascii.write(filecontent, filename, overwrite=True)

        print ("sigmas for duration",dr,"d, alpha",alph,":",sigmas_1a)
        sigmas[i] = sigmas_1a

        # plt.figure(figsize=(8,6))
        # for j in range(i+1):
        #     plt.scatter([alpha[j]]*N1,sigmas[j],marker='.',c='b')
        # plt.xlabel("alpha")
        # plt.ylabel("sigmas")
        # plt.xscale("log")
        # plt.grid()
        # plt.title("alpha vs sigma")
        # plt.savefig("plots/AlphaVSigma"+str(i+1)+".jpg")
    print ("For duration:",dr,"d")
    print ("alpha values:",alpha)
    print ("Sigma values:",np.array(sigmas))
    print ("Time elapsed:",timing-time.time())
