from my_functions import *

calculate_again = True
colors = ['red','blue']

all_plots = False


def plot_single_alpha_Lrange(data,maxs,dr,im,b):
    plt.figure(figsize=(12,6))
    for j in range(N1):
        for ch in range(2):
            if j==0:
                plt.plot(Amp_range[b]/Amp_true[b],data[j,ch],label=rec[ch],c=colors[ch])
            else:
                plt.plot(Amp_range[b]/Amp_true[b],data[j,ch],c=colors[ch])
            if ch ==0:
                plt.axvline(maxs[ch,j],linestyle='--',alpha=0.5,label='Max, iter {}'.format(j+1), c=colors[ch])
            else:
                plt.axvline(maxs[ch,j],linestyle='--',alpha=0.5,label='Max, iter {}'.format(j+1), c=colors[ch])
    # print (maxs)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.ylim(ymin=0)
    plt.xlabel("Amplitude amplification")
    plt.ylabel("Likelyhood")
    plt.axvline(1,c='black',linestyle='dashed',label="Amp true")
    # plt.legend()
    # plt.ylim(1e-7*np.array([-1,1.5]))
    plt.grid()
    plt.savefig("plots/{}d/{}/AE_matchedfiltering_bin{}.jpg".format(dr,alpha_nm[i],b))
    plt.close()

def plot_single_day_maxvalues(data,dr,b):
    plt.figure(figsize=(8,6))
    plt.axhline(1,c='black',linestyle='--',alpha=0.8,label='True amplitude')
    for ch in range(2):
        mean = np.mean(data[:,ch],axis=1)
        std = np.std(data[:,ch],axis=1)

        plt.errorbar(alpha,mean,yerr=std,fmt='--o',label="Channel {}".format(rec[ch]),c=colors[ch])

    plt.xscale('log')
    #plt.yscale('log')
    plt.legend()
    plt.xticks(alpha)
    plt.grid()
    plt.xlabel("Total alpha")
    plt.ylabel("Amplitude/A$_0$")
    plt.savefig("plots/{}d/Matched_filtering_trueAmp_bin{}.jpg".format(dr,b))
    plt.close()

def plot_single_day_maxvalues_sig(data,dr,b):
    plt.figure(figsize=(8,6))
    for ch in range(2):
        mean = np.mean(data[:,ch],axis=1)
        std = np.std(data[:,ch],axis=1)

        plt.errorbar(alpha,std,fmt='--o',label="Channel {}".format(rec[ch]),c=colors[ch])

    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xticks(alpha)
    plt.grid()
    plt.xlabel("Total alpha")
    plt.ylabel("Amplitude/A$_0$")
    plt.savefig("plots/{}d/Matched_filtering_trueAmp_sig_bin{}.jpg".format(dr,b))
    plt.close()



amp_max_all = np.zeros((2,len(dur_range),Ngalbins,len(alpha),2,N1))
L_range_tot = np.zeros((2,len(dur_range),Ngalbins,len(alpha),N1,2,N2))
if calculate_again:
    for ns_i,ns_src in enumerate(noise_source):
        for d,dr in enumerate(dur_range):
            
            model_fftA_tot = []
            model_fftE_tot = []
            print ("Import mdata and make FFTs for noise {}, duration {} d".format(ns_src,dr))
            for b in tqdm(range(Ngalbins)):
                fp = ext+'measurements/'+ns_src+'/'+str(dr)+'d/'    
                fpm = fp+'mdata/binary_{}.txt'.format(b)

                iopt = N2//2
                rawdata = ascii.read(fpm)
                mdata_0 = np.array([rawdata['t'], rawdata['A'],rawdata['E']])

                # mdata_all = np.zeros((N2,2,mdata_0.shape[-1]))
                # for k,Amp in enumerate(Amp_range[b]):
                #     mdata_all[k] = np.copy(mdata_0[1:])*Amp
                    
                # Make the FFT's of the sample data
                model_fftA = np.zeros((N2,mdata_0.shape[-1]),dtype=complex)
                model_fftE = np.zeros((N2,mdata_0.shape[-1]),dtype=complex)
                for k,Amp in enumerate(Amp_range[b]):
                    model_fftA[k] = FFT(mdata_0[1]*Amp)
                    model_fftE[k] = FFT(mdata_0[2]*Amp)
                m_fft_freq = np.fft.fftshift(np.fft.fftfreq(len(mdata_0[1]), d=1/fs))
                
                model_fftA_tot.append(model_fftA)
                model_fftE_tot.append(model_fftE)
            
            # Save the optimal amplitude values
            amp_max = np.zeros((Ngalbins,len(alpha),2,N1))
            L_range = np.zeros((Ngalbins,len(alpha),N1,2,N2))
            print ("Start calculations for duration {} d, (RAM usage = {} %)".format(dr,psutil.virtual_memory()[2]))
            
            for i,alph in enumerate(tqdm(alpha)):
                fp_plot = 'plots/{}d/{}/'.format(dr,alpha_nm[i])

                for j in range(N1):
                    # Calc FFTs
                    rawdata = ascii.read(fp + alpha_nm[i] + '/fs'+str(j)+'.txt')
                    # fsdata = np.array([rawdata['t'],rawdata['A'],rawdata['E']])

                    fs_fft2A = FFT(rawdata['A'])
                    fs_fft2E = FFT(rawdata['E'])

                    # Create large array over which we can iterate the binaries
                    fs_fft2A_tot = np.tile(fs_fft2A,N2).reshape(N2,len(fs_fft2A))
                    fs_fft2E_tot = np.tile(fs_fft2E,N2).reshape(N2,len(fs_fft2E))

                    for b in range(Ngalbins):
                        L_range[b,i,j,0] = np.sum(np.abs(fs_fft2A_tot-model_fftA_tot[b])**2,axis=1)
                        L_range[b,i,j,1] = np.sum(np.abs(fs_fft2E_tot-model_fftE_tot[b])**2,axis=1)

                for b in range(Ngalbins):
                    for ch in range(2):
                        filename = 'measurements/'+ns_src+'/{}d/{}/L_range_{}_bin{}.txt'.format(dr,alpha_nm[i],rec[ch],b)
                        filecontent = Table(L_range[b,i,:,ch].T)
                        ascii.write(filecontent, filename, overwrite=True)


                    # plot the L_ranges for different iterations of noise
                    for j in range(N1):
                        for ch in range(2):
                            # maxi = np.where(L_range[j,ch] == np.max(L_range[j,ch]))[0][0]
                            maxi = np.where(L_range[b,i,j,ch] == np.min(L_range[b,i,j,ch]))[0][0]
                            amp_max[b,i,ch,j] = Amp_range[b][maxi]/Amp_true[b]
                    if all_plots:
                        plot_single_alpha_Lrange(L_range[b,i],amp_max[b,i],dr,i,b)
            
            L_range_tot[ns_i,d] = L_range
            amp_max_all[ns_i,d] = amp_max
            
            for b in range(Ngalbins):
                for ch in range(2):
                    filename = ext+'measurements/'+ns_src+'/{}d/Matched_filtering_result_{}_bin{}.txt'.format(dr,rec[ch],b)
                    filecontent = Table(amp_max[b,:,ch].T)
                    ascii.write(filecontent, filename, overwrite=True)
            # Plot values of optimal amplitude
            # amp_max = np.zeros((len(alpha),2,N1))
            if all_plots:
                plot_single_day_maxvalues(amp_max[ns_i],dr,b)
                plot_single_day_maxvalues_sig(amp_max[ns_i],dr,b)

            
else:
    print ("Not recalculating everything but importing the data")
    for ns_i,ns_src in enumerate(noise_source):
        for d,dr in enumerate(dur_range):
            for b in range(Ngalbins):
                for ch in range(2):
                    filename = ext+'measurements/'+ns_src+'/{}d/Matched_filtering_result_{}_bin{}.txt'.format(dr,rec[ch],b)
                    rawdata = ascii.read(filename)

                    filecontent = np.zeros((len(alpha),N1))
                    for i in range(len(alpha)):
                        filecontent[i] = rawdata['col'+str(i)]

                        filename2 = 'measurements/'+ns_src+'/{}d/{}/L_range_{}_bin{}.txt'.format(dr,alpha_nm[i],rec[ch],b)
                        rawdata2 = ascii.read(filename2)

                        filecontent2 = np.zeros((N1,N2))
                        for j in range(N1):
                            filecontent2[j] = rawdata2['col'+str(j)]
                        L_range_tot[ns_i,d,b,i,:,ch] = filecontent2
                    amp_max_all[ns_i,d,b,:,ch] = filecontent
                if all_plots:
                    # Make amplitude plots
                    print ("Making plots")
                    for d,dr in enumerate(tqdm(dur_range)):
                        for i,alph in enumerate(alpha):
                            plot_single_alpha_Lrange(L_range_tot[ns_i,d,b,i],amp_max_all[ns_i,d,b,i],dr,i,b)

                        plot_single_day_maxvalues(amp_max_all[ns_i,d,b],dr,b)

for ns_i,ns_src in enumerate(noise_source):
    for d,dr in enumerate(dur_range):
        stds = np.zeros((len(dur_range),2,len(alpha)))
        fig,axs = plt.subplots(2,sharex=True, figsize=(12,8), gridspec_kw={"hspace":0})
        for ch in range(2):
            axs[ch].axhline(1,c='black',linestyle='--',alpha=0.8,label='True amplitude')
            for d,dr in enumerate(dur_range):
                    mean = np.mean(amp_max_all[ns_i,d,b,:,ch],axis=1)
                    std = np.std(amp_max_all[ns_i,d,b,:,ch],axis=1)
                    stds[d,ch] = std
                    axs[ch].errorbar(alpha,mean,yerr=std, label="{} d".format(dr), fmt='o--', capsize=5, capthick=1)

            axs[ch].set_xscale('log')
            axs[ch].set_yscale('log')
            axs[ch].legend()
            axs[ch].grid()
            axs[ch].set_ylabel("Amp/A$_0$ chan {}".format(rec[ch]))
        axs[ch].set_xlabel("Alpha of acc and OMS noise")
        axs[ch].set_xticks(alpha)

        plt.savefig("plots/masterplots/MF_plot1_bin{}.jpg".format(b))
        plt.close()

        fig,axs = plt.subplots(2,sharex=True, figsize=(12,8), gridspec_kw={"hspace":0})
        for ch in range(2):
            for d,dr in enumerate(dur_range):
                    mean = np.mean(amp_max_all[ns_i,d,b,:,ch],axis=1)

                    axs[ch].plot(alpha,stds[d,ch], label="{} d".format(dr), marker='o', linestyle='--')

            axs[ch].set_xscale('log')
            axs[ch].set_yscale('log')
            axs[ch].legend()
            axs[ch].grid()
            axs[ch].set_ylabel("$\sigma_{A/A_0}$ channel "+rec[ch])
        axs[ch].set_xlabel("Amplification of acc and OMS noise")
        axs[ch].set_xticks(alpha)

        plt.savefig("plots/masterplots/MF_plot3_bin{}.jpg".format(b))
        plt.close()

print ("Finished!")
