from my_functions import *

# Define the parabula functions
def sig_func(x0,xpm,f0,fpm):
    return np.abs(xpm-x0) / np.sqrt(fpm-f0)

def parabula_err(x0, xpm, L_range, iopt, N3):
    """Finds error of likelyhood parabula. Input must be the optimal value and the offset from this for which to calculate the parabula. The output is sigma"""
    f0, fm, fp = L_range[N3], L_range[:N3], L_range[N3+1:]
    xm, xp = xpm[:N3], xpm[N3+1:]
    # print (f0,fm,fp)
    # print (xm,xp)
    
    sigs_m = sig_func(x0,xm,f0,fm)
    sigs_p = sig_func(x0,xp,f0,fp)
    # return sigs_m, sigs_p
    sigs_tot = list(sigs_m)
    sigs_tot.extend(list(sigs_p))
    return np.nanmean(sigs_tot)

# Import the data
for b in usebinaries:
    for ns_src in noise_source:
        amp_max_all = np.zeros((len(dur_range),len(alpha),2,N1))
        L_range_tot = np.zeros((len(dur_range),len(alpha),N1,2,N2))
        for d,dr in enumerate(tqdm(dur_range)):
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
                    L_range_tot[d,i,:,ch] = filecontent2
                amp_max_all[d,:,ch] = filecontent


        print ("Calculating sigmas for binary {}".format(b))
        sigmas_tot = np.zeros((len(dur_range),len(alpha),N1,2))
        for d,dr in enumerate(dur_range):
            sigmas = np.zeros((len(alpha),N1,2))
            for i,alph in enumerate(alpha):        
                for j in range(N1):
                    for ch in range(2):
                        iopt = np.where(Amp_range[0]/Amp_true[0] == amp_max_all[d,i,ch,j])[0][0]

                        # N3 defines how many samples next to the minimum are taken into account
                        # This routine makes sure the lists work as they are supposed to
                        if iopt == 0 or iopt == N2:
                            sig = np.nan
                        else:
                            if N2-iopt >= N2//4 and iopt >= N2//4:
                                N3 = N2//4
                            elif iopt < N2//4:
                                N3 = iopt
                            else:
                                N3 = N2-iopt-1

                            # define min and max over which to iterate
                            mi,ma = iopt - N3, iopt + N3 + 1
                            sig = parabula_err(1,(Amp_range[0]/Amp_true[0])[mi:ma],L_range_tot[d,i,j,ch][mi:ma],iopt,N3)

                        sigmas[i,j,ch] = sig
            sigmas_tot[d] = sigmas

        print ("Sigmas calculated")
        fig,axs = plt.subplots(2,sharex=True, figsize=(12,8), gridspec_kw={"hspace":0})
        for ch in range(2):
            for d,dr in enumerate(dur_range):
                avg = np.nanmean(sigmas_tot[d,:,:,ch],axis=1)
                stdv = np.nanstd(sigmas_tot[d,:,:,ch],axis=1)
                #axs[ch].errorbar(alpha, avg/avg[4], yerr=stdv/avg[4], label="{} d".format(dr), fmt='o--', capsize=5, capthick=1)
                axs[ch].errorbar(alpha, avg, yerr=stdv, label="{} d".format(dr), fmt='o--', capsize=5, capthick=1)
            axs[ch].axvline(1,c='black',alpha=0.8,linestyle='--',label="Nominal noise")
            axs[ch].legend()
            axs[ch].grid()
            axs[ch].set_yscale("log")
            axs[ch].set_ylabel("Sigma in {}".format(rec[ch]))
        axs[ch].set_xlabel("Alpha")
        axs[ch].set_xscale("log")
        axs[ch].set_xticks(alpha)
        axs[0].set_title("Error in Amplitudes for different noise levels")
        plt.savefig("plots/masterplots/MF_plot2_bin{}.jpg".format(b))
        plt.close()
