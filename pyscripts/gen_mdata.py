# By Bart Ouwehand 09-06-2022

from my_functions import *

# Function to create individual model datastreams
def indiv_mdata(st,Afunc,Efunc,s,a, f, p, beta, lamb, t0=orbits_t0+1/fs):
    # Create random filename to allow for multiprocessing

    gwfn = ext2+'gwtmp_'+str(int(1e20*np.random.rand(1)[0]))+'.h5'
    GalBin = GalacticBinary(A=a/f, f=f, phi0=p, orbits=orbit_path, t0=t0, gw_beta=beta, gw_lambda=lamb, dt=1/fs, size=s+300)
    GalBin.write(gwfn)
    
    # rawdata = Data.from_gws( 'gw_tmp.h5', orbit_path)
    rawdata = Data.from_gws(gwfn, orbit_path,interpolate=False)
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

# We need the sample time to equal to the model time so:
# for dr,i in enumerate(dur_range):
d,dr = -1,dur_range[-1]
fp = ext+'measurements/tm_asds/'+str(dr)+'d/1/fs0.txt'
rawdata = ascii.read(fp)
fsdata = np.array([rawdata['t'],rawdata['A'],rawdata['E']])

# Generate AE chanels for mdata
Afunc, Efunc = BuildModelTDI(orbit_path,fs,size[d]+600,Amp_true, f_true, phi0_true, gw_beta_true, gw_lambda_true,discard,detailed=True,interpolate=False)

fp = ext+'measurements/tm_asds/'+str(dr)+'d/'
fp2 = 'plots/'+str(dr)+'d/mdata/'
# Generate the datafiles for the different binaries
for a, f, p, beta, lamb,i in zip(Amp_true, f_true, phi0_true, gw_beta_true, gw_lambda_true,range(Ngalbins)):
    
    # Amp_range = a*np.linspace(0.1,2.5,N2) # Range of amplitudes
    
    mdata_1b = np.zeros((1+N2*2,len(fsdata[0])))
    print ("Calculating mdata for binary {}".format(i+1))
    for j,Amp in enumerate(tqdm(Amp_range[i])):
        mdata_tmp = indiv_mdata(fsdata[0],Afunc,Efunc,size[d]+600,Amp, f, p, beta, lamb, t0=orbits_t0+1/fs)
        mdata_1b[0] = mdata_tmp[0]
        mdata_1b[1+2*j:3+2*j] = mdata_tmp[1:]
    
    names = ['t']
    for k in range(N2):
        names.extend(['A'+str(k),'E'+str(k)])
    print ("datashape =",mdata_1b.shape)
    
    filename = fp+"mdata/binary_"+str(i)+".txt"
    # filecontent = Table(sdata.T, names=['t','A','E','T'])
    filecontent = Table(mdata_1b.T, names=names)
    ascii.write(filecontent, filename, overwrite=True)


    fig, axs = plt.subplots(2,sharex=True,figsize=(10,8),gridspec_kw={'hspace':0})
    # plt.plot(x,y,c='black',alpha=.8)
    for k in range(2):
        for j,Amp in enumerate(Amp_range):
            axs[k].plot(mdata_1b[0]/day,mdata_1b[1+2*j+k],label='{:.3f} A$_0$'.format(Amp/a))
        axs[k].set_ylabel("{} amplitude".format(rec[k]))
    axs[0].set_title("Model data for binary {} at different amplitudes for duration {} d".format(i,dr))
    axs[0].legend(loc=1)
    axs[k].set_xlim(.2,.22)
    axs[k].set_xlabel("Time [d]")
    fig.savefig(fp2+"binary_"+str(i)+".jpg")
    plt.close(fig)

