# By Bart Ouwehand 12-04-2022
from my_functions import *

calc_sigs = False

if calc_sigs:
    # Defining the model used for MCMC fitting
    def model(st,orbits_ZP,Afunc,Efunc,s,Amp=Amp_true, phi0=phi0_true, freq=f_true, gw_beta=gw_beta_true, gw_lambda=gw_lambda_true, t0=orbits_t0+1/fs):

        # Create random filename to allow for multiprocessing
        gwfn = 'gws_spam/gwtmp_'+str(int(1e20*np.random.rand(1)[0]))+'.h5'

        # Amp, phi0 = theta[0:Ngalbins], theta[Ngalbins:2*Ngalbins]

        # Generate GW signals
        for a, f, p, beta, lamb in zip(Amp, freq, phi0, gw_beta, gw_lambda):
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

    # Define the parabula functions

    def sig(x0,xpm,N,f0,fpm):
        return xpm / np.sqrt(2*np.log((N-f0) / (N-fpm)))

    def parabula_err(x0, xpm, fsdata,L_on_range,f0,Afunc,Efunc,s):
        """Finds error of likelyhood parabula. Input must be the optimal value and the offset from this for which to calculate the parabula. The output is sigma"""
        med = np.median(L_on_range)
        # f0 = lnL([x0],fsdata[0],fsdata[1], fsdata[2])
        fp, fm = lnL([x0+xpm],fsdata[0],fsdata[1], fsdata[2],Afunc,Efunc,s),lnL([x0-xpm],fsdata[0],fsdata[1], fsdata[2],Afunc,Efunc,s)
        # print (med, f0, fp, fm)
        return np.nanmean([sig(x0,xpm,med,f0,fp), sig(x0,xpm,med,f0,fm)])

    def parabula(x0, L_on_range,f0,sig):
        N = np.median(L_on_range)
        # A = lnL([x0],fsdata[0],fsdata[1], fsdata[2]) - N
        length = 2000
        xrange = np.linspace(-np.pi,np.pi,length)
        return xrange, N - (N-f0)*np.exp(-((xrange-x0)**2)/(2*(sig**2)))
        # return xrange, N + (f0-N)*np.exp(((xrange-x0)**2)/(sig**2))

        
if calc_sigs:
    # Making plots for 
    fp = 'measurements/tm_asds/'
    zp_range = np.pi*np.linspace(-1,1,N2)
    # zp_range_tight = 0.25*np.linspace(-1,1,N2)

    sigmas = np.zeros((len(duration),len(alpha),N1))

    for d,dr in enumerate(duration):
        print ("Calculations for duraiton {} d".format(dr))
        if calc_sigs:
            Afunc, Efunc = BuildModelTDI(orbit_path,fs,size[d],Amp_true, f_true, phi0_true, gw_beta_true, gw_lambda_true,discard,detailed=True)
        
        sigmas_1d = np.zeros((len(alpha),N1))
        # for a,alph in enumerate(tqdm(alpha)):
        for a,alph in enumerate(alpha):
            # Start by reading in the likelyhood range
            file1 = fp+str(dr)+'d/L_range_a'+str(alph)+'.txt'
            rawdata1 = ascii.read(file1)
            L_range = np.zeros((N1,N2))
            for i in range(N1):
                L_range[i] = np.array(rawdata1['col'+str(i+1)])[1:]
                
            # if dr != 60:
            #     file2 = fp+str(dr)+'d/L_range_a'+str(alph)+'tight.txt'
            #     rawdata2 = ascii.read(file2)
            #     L_range_tight = np.zeros((N1,N2))
            #     for i in range(N1):
            #         L_range_tight[i] = np.array(rawdata2['col'+str(i+1)])[1:]
                    
            # for i in range(N1):
            #     plt.figure(figsize=(8,6))
            #     plt.plot(zp_range, L_range[i], marker='.', linestyle='--', linewidth=1, c='b')
            #     plt.plot(zp_range_tight, L_range_tight[i], marker='.', linestyle='-', linewidth=1, c='r')
            #     plt.title("Likelyhood range for {} d and {} alpha value".format(dr,alph))
            #     plt.xlabel("Orbit-ZP [rad]")
            #     plt.ylabel("-ln(L)")
            #     plt.yscale('log')
            #     plt.savefig('plots/tight/'+str(dr)+'d/'+str(alph)+'/'+str(i)+'.jpg')
                # plt.close()
            
            sigmas_1a = np.zeros(N1)
            fp2 = 'plots/'+str(dr)+'d/'+str(int(alph))+'/'

            for j in range(N1):
                # inputf = '../../data1/measurements/tm_asds/'+str(int(dr))+'d/'+str(int(alph))+"/fs"+str(j)
                inputf = 'measurements/tm_asds/'+str(int(dr))+'d/'+str(int(alph))+"/fs"+str(j)

                rawdata = ascii.read(inputf+'.txt')
                fsdata = np.array([rawdata['t'],rawdata['A'],rawdata['E']])

                sigma = parabula_err(0,0.01, fsdata, L_range[j], L_range[j,N2//2],Afunc,Efunc,size[d])
                x,y = parabula(0,L_range[j], L_range[j,N2//2],sigma)
                # print ("Time to calculate likelyhood range = {:.2f} s / {:.3f} hrs".format(time.time()-L_range_time0,(time.time()-L_range_time0)/3600))

                sigmas_1a[j] = sigma
                # print ("For alpha={}, sigma={:.2f}".format(alph,sigma))
                
                plt.figure(figsize=(8,6))
                plt.plot(x,y,c='black',alpha=.8)
                plt.plot(zp_range,L_range[j],marker='.',ls='--',c='r',linewidth=1)
                # if dr != 60:
                #     plt.plot(zp_range_tight,L_range_tight[j],marker='o',ls='--',c='indianred')
                plt.xlabel("zp [rad]")
                plt.ylabel("-ln(L)")
                plt.title("Likelyhood range over ZP range for alpha {} and iteration {}".format(alph,j))
                # plt.ylim(np.min(L_range),np.max(L_range))
                plt.savefig(fp2+"L_range_over_ZP_"+str(j)+".jpg")
                plt.close()

            print ("sigmas for duration",dr,"d, alpha",alph,":",sigmas_1a)
            sigmas_1d[a] = sigmas_1a
        sigmas[d] = sigmas_1d
        filename = fp+str(dr)+'d/sigmas.txt'
        
        print (sigmas_1d.shape,alpha.shape)
        filecontent = Table(sigmas_1d.T,names=alpha)
        ascii.write(filecontent, filename, overwrite=True)


sigmas = np.zeros((len(duration),len(alpha),N1))
for d,dr in enumerate(duration):
    filename= 'measurements/tm_asds/'+str(dr)+'d/sigmas.txt'
    rawdata = ascii.read(filename)
    sigmas_1d = np.zeros((len(alpha),N1))
    for a,alph in enumerate(alpha):
        sigmas_1d[a] = np.array(rawdata['col'+str(a+1)][1:])
    sigmas[d] = sigmas_1d

# Make the plots of the likelyhood range
def parabula(x0, L_on_range,f0,sig):
        N = np.median(L_on_range)
        # A = lnL([x0],fsdata[0],fsdata[1], fsdata[2]) - N
        length = 2000
        xrange = np.linspace(-np.pi,np.pi,length)
        return xrange, N - (N-f0)*np.exp(-((xrange-x0)**2)/(2*(sig**2)))
        # return xrange, N + (f0-N)*np.exp(((xrange-x0)**2)/(sig**2))

zp_range = np.pi*np.linspace(-1,1,N2)
for d,dr in enumerate(duration):
    fp = 'plots/'+str(dr)+'d/'
    print ("Making plots for {} d".format(dr))
    for a,alph in enumerate(tqdm(alpha)):
        # Start by reading in the likelyhood range
        file1 = 'measurements/tm_asds/'+str(dr)+'d/L_range_a'+str(alph)+'.txt'
        rawdata1 = ascii.read(file1)
        fpp = fp+str(alph)+'/'
        for i in range(N1):
            L_range = np.array(rawdata1['col'+str(i+1)])[1:]
            x,y = parabula(0,L_range, L_range[N2//2],sigmas[d,a,i])
            plt.figure(figsize=(8,6))
            plt.plot(x,y,c='black',alpha=.8)
            plt.plot(zp_range,L_range,marker='.',ls='--',c='r',linewidth=1)
            plt.xlabel("zp [rad]")
            plt.ylabel("-ln(L)")
            plt.title("Likelyhood range over ZP range for alpha {} and iteration {}".format(alph,i))
            # plt.ylim(np.min(L_range),np.max(L_range))
            plt.savefig(fp+str(alph)+"/L_range_over_ZP_"+str(i)+".jpg")
            plt.close()

skip = 0
plt.figure(figsize=(8,6))
for d,dr in enumerate(duration[skip:]):
    plt.errorbar(alpha,np.nanmean(sigmas[skip+d],axis=1),yerr = np.nanstd(sigmas[skip+d],axis=1),fmt='--o',label=str(dr)+'d',linewidth=1,capsize=5,capthick=1)
    print ("For",str(dr),"d, number of non-nan points = ",np.sum(~np.isnan(sigmas[skip+d]),axis=1))
plt.ylabel("$\sigma_{\delta}$")
plt.xscale('log')
plt.legend()
plt.xlabel("TM_acceleration noise amplification")
plt.title("Error in determining orbits ZP ($\delta$) for amplified tm acceleration noise")
plt.savefig('plots/masterplot.jpg')
plt.close()