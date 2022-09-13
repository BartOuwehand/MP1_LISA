# By Bart Ouwehand 12-04-2022

from my_functions import *

import gc

d,dr = -1,dur_range[-1]

gen_fsdata = True
gen_mdata = True

if run_new_simulation:
    # Generate the Galactic binaries
    GenerateGalbins(orbit_path,gw_path,fs,size[d],Amp_true, f_true, phi0_true_forinst, gw_beta_true, gw_lambda_true, iota_true, orbits_t0 + 1/fs)
    
    # Generate the instrument data
    # sAfunc, sEfunc, sTfunc = GenerateInstrumentAET(orbit_path, gw_path, fs, size, sample_outputf, discard)
    sdata, sAfunc, sEfunc = GenerateInstrumentAET(orbit_path, gw_path, fs, size[d], discard)

# Retreive A, E, T data
# rawdata = ascii.read(sample_outputf+'.txt')
# # sdata = np.array([rawdata['t'],rawdata['A'],rawdata['E'],rawdata['T']])
# sdata = np.array([rawdata['t'],rawdata['A'],rawdata['E']])

# Plot raw data retreived from AET datastream
if gen_plots:
    for i in range(2):
        plt.plot(sdata[0]/day,sdata[i+1],label=rec[i],alpha=.5)
    plt.title('AET datastreams for '+str(dr)+'d simulation with all noises')
    plt.legend(loc=4)
    plt.xlabel('Time (d)')
    plt.ylabel('Amplitude')
    plt.savefig("plots/Sample_rawAETdata.jpg")

def psd_func(data):
    return scipy.signal.welch(data,fs=fs,window='nuttall',nperseg=len(data),detrend=False)

# Create psd for data
tmp = []
for i in range(1,3):
    ftmp, psdtmp = psd_func(sdata[i])
    tmp.append(psdtmp)
psd = np.array([ftmp,tmp[0],tmp[1]])

# Create filtered data
cutoff = 100
tmp = []
#coeffs = scipy.signal.firls(73,bands=[0,1,1.2,2],desired=[1,1,0,0],fs=fs)
#coeffs = scipy.signal.firls(73, bands=[0,1e-2,3e-2,5e-2], desired=[1,1,0,0],fs=fs)
coeffs = scipy.signal.firls(73, bands=[0,1e-2,1.5e-2,fs/2], desired=[1,1,0,0],fs=fs)
for i in range(1,3):
    fdata_tmp = scipy.signal.filtfilt(coeffs,1., x=sdata[i],padlen=len(psd[0]))
    tmp.append(fdata_tmp[cutoff:-cutoff])
fsdata = np.array([sdata[0][cutoff:-cutoff],tmp[0],tmp[1]])

# Create psd for filtered data
tmp = []
for i in range(1,3):
    ftmp, psdtmp = psd_func(fsdata[i])
    tmp.append(psdtmp)
fpsd = np.array([ftmp,tmp[0],tmp[1]])

if gen_plots:
    fig, axs = plt.subplots(2, figsize=(16,4), sharex=True, gridspec_kw={'hspace':0})
    fig.suptitle("Filtered datastreams for a "+str(dr)+" day simulation")
    for i in range(2):
        axs[i].plot(fsdata[0]/day,fsdata[i+1],label='Channel '+rec[i])
        axs[i].legend(loc=1)
        axs[i].set_ylabel('Amplitude')
    axs[i].set_xlabel('Time (d)')
    axs[i].set_xlim(.25,.75)
    plt.savefig('plots/Sample_filteredAETdata.jpg')

    fig, axs = plt.subplots(2, figsize=(24,6), sharex=True, gridspec_kw={'hspace':0})
    for i in range(2):
        for f in f_true:
            if f == f_true[0]:
                axs[i].plot([f]*2,[1e-50,1e50],c='black',alpha=0.5,label="verbins")
            else:
                axs[i].plot([f]*2,[1e-50,1e50],c='black',alpha=0.5)
        axs[i].plot(psd[0],psd[i+1], label=rec[i]+' unfiltered',c='blue')
        axs[i].plot(fpsd[0],fpsd[i+1], label=rec[i]+' filtered',c='red')
        axs[i].legend()
        axs[i].set_yscale('log')
        axs[i].set_xscale('log')
        axs[i].set_xlabel('Freq [Hz]')
        axs[i].set_ylabel('ASD [Hz/sqrt(Hz)]?')
        axs[i].set_ylim(ymin=0.1*np.min(fpsd[i+1]),ymax=10*np.max(psd[i+1]))
        # axs[i].set_title("PSD of chanel "+rec[i]+' filtered and unfiltered')
        axs[i].grid()
    axs[0].set_title("PSD of chanels filtered and unfiltered")
    axs[i].set_xlim(xmin=9e-5,xmax=fs/1.9)#,xmax=1e-2)
    plt.savefig("plots/Sample_PSD.jpg")


# Generate the data for different alpha values
# alpha = np.array([1,10,100,1000,10000])
if gen_fsdata:
    for sample_alpha in ['acc','OMS']:
        for i,alph in enumerate(alpha):
            alph_nm = alpha_nm[i]
            # for j in tqdm(range(N)):
            fp = 'plots/{}d/{}/'.format(dr,alph_nm)
            print ("Start calculations for duration {} d, noise {}, alpha {}, (RAM usage = {} %)".format(dr,sample_alpha,alph,psutil.virtual_memory()[2]))
            for j in tqdm(range(10,N1)):

                # outputf = ext+'measurements/tm_asds/'+str(dr)+'d/'+alph_nm+"/s"+str(j)

                GenerateGalbins(orbit_path, gw_path, fs, size[d], Amp_true, f_true, phi0_true_forinst, gw_beta_true, gw_lambda_true, iota_true, orbits_t0 + 1/fs)

                # Enable or disable the noise for testing
                if sample_alpha == 'acc':
                    filepath = ext+'measurements/tm_asds/'+str(dr)+'d/'+alph_nm+"/fs"+str(j)+'.txt'
                    sdata = GenerateInstrumentAET(orbit_path, gw_path, fs, size[d], discard, False, sAfunc, sEfunc, tm_alpha=alph, noise=True, printing=False,
                                              turnoff=turnoff)

                elif sample_alpha == 'OMS':
                    filepath = ext+'measurements/oms_asds/'+str(dr)+'d/'+alph_nm+"/fs"+str(j)+'.txt'
                    sdata = GenerateInstrumentAET(orbit_path, gw_path, fs, size[d], discard, False, sAfunc, sEfunc, OMS_alpha=alph, noise=True, printing=False,
                                              turnoff=turnoff)

                elif sample_alpha == 'both':
                    filepath = ext+'measurements/tm_asds/'+str(dr)+'d/'+alph_nm+"/fs"+str(j)+'.txt'
                    sdata = GenerateInstrumentAET(orbit_path, gw_path, fs, size[d], discard, False, sAfunc, sEfunc, tm_alpha=alph, OMS_alpha=alph, noise=True, printing=False,
                                              turnoff=turnoff)
                else:
                    print ("sample_alpha = {}, is not an option".format(sample_alpha))
                    break

                # Filter sample data
                tmp = []
                for k in range(1,3):
                    fdata_tmp = scipy.signal.filtfilt(coeffs,1., x=sdata[k],padlen=len(psd[0]))
                    tmp.append(fdata_tmp[cutoff:-cutoff])
                fsdata = np.array([sdata[0][cutoff:-cutoff],tmp[0],tmp[1]])

                # # Make simple plot just to check if alpha makes sense
                # plt.plot(fsdata[0]/day,fsdata[1])
                # plt.xlabel("Time (d)")
                # plt.ylabel("GW amp")
                # plt.title("Datastream for {} d, alpha {}, iteration {}".format(dr,alph,j))
                # plt.savefig(fp+"Datastream_"+str(j))
                # plt.close()

                # Write fsdata to a file
                # filepath = ext+'measurements/tm_asds/'+str(dr)+'d/'+alph_nm+"/fs"+str(j)+'.txt'
                filecontent = Table(fsdata.T, names=['t','A','E'])
                ascii.write(filecontent, filepath, overwrite=True)

print ("Start mdata calculation")
if gen_mdata:
    # Ready to generate the model data!
    
    if sample_alpha == 'acc':
        fp = ext+'measurements/tm_asds/'+str(dr)+'d/'
    elif sample_alpha == 'OMS':
        fp = ext+'measurements/oms_asds/'+str(dr)+'d/'
    elif sample_alpha == 'both':
        fp = ext+'measurements/tm_asds/'+str(dr)+'d/'

    
    fp2 = 'plots/'+str(dr)+'d/mdata/'
    
    for a, f, p, beta, lamb, iota, i in tqdm(zip(Amp_true, f_true, phi0_true, gw_beta_true, gw_lambda_true, iota_true, range(Ngalbins)),total=Ngalbins):
        # a = 1 and p = 0 (for instrument parameters)
        GenerateGalbins(orbit_path, gw_path, fs, size[d], [1], [f], [0], [beta], [lamb], [iota], orbits_t0 + 1/fs)

        mdata = GenerateInstrumentAET(orbit_path, gw_path, fs, size[d], discard, False, sAfunc, sEfunc, noise=False, printing=False)


        # Filter model data to all the steps be equal the fsdata
        tmp = []
        for k in range(1,3):
            fmdata_tmp = scipy.signal.filtfilt(coeffs,1., x=mdata[k],padlen=len(psd[0]))
            tmp.append(fmdata_tmp[cutoff:-cutoff])
        fmdata = np.array([mdata[0][cutoff:-cutoff],tmp[0],tmp[1]])
        
        # Write mdata to a file
        filepath = fp+"mdata/binary_"+str(i)+".txt"
        # filecontent = Table(sdata.T, names=['t','A','E','T'])
        filecontent = Table(fmdata.T, names=['t','A','E'])
        ascii.write(filecontent, filepath, overwrite=True)

print ("Final RAM usage {} %".format(psutil.virtual_memory()[2]))
