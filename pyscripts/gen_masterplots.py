from my_functions import *


d,dr = -1, dur_range[-1]

# prepare arrays
# ([acc,oms],[alpha],[A,E],[N1 iterations])
amp_max_all = np.zeros((2,Ngalbins,len(alpha),2,N1))
# ([acc,oms],[alpha],[N1 iterations],[A,E],[N2 likelyhood range values])
L_range_tot = np.zeros((2,Ngalbins,len(alpha),N1,2,N2))


# import the data
for ns,ns_src in enumerate(noise_source):
    print ("Importing {} data".format(ns_src))
    for b in tqdm(usebinaries):
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
                L_range_tot[ns,int(b),i,:,ch] = filecontent2
            amp_max_all[ns,int(b),:,ch] = filecontent

# Calculate the standard deviations
stdv_all = np.std(amp_max_all,axis=4)

print ("Making plot 1")
# Create the sick plots
import matplotlib.pylab as pl

lab2 = ['acc','OMS']

sorted_galbins = np.argsort(f_true)[::-1]

def gb_influplot(a,data,N,axs,i,j,skip=1,label=False,ymin=8e-2):
    for k in sorted_galbins[::skip]:
        axs[i,j].loglog(a,data[:,k],label="{:.2f} mHz".format(f_true[k]*1e3),c=pl.cm.viridis(1-f_true[k]/f_true[0]))
        #c=times/1e9,cmap='jet'
    # Enable title to see what is plotted
    # axs[i,j].set_title(title)
    axs[i,j].axvline(1,c='black',linestyle='--',alpha=0.8)
    axs[i,j].grid()
    axs[i,j].tick_params(axis='both',which='both',top=True,right=True)
    # axs[i,j].axhline(1,c='black',linestyle='--',alpha=.5)
    if i == 1:
        if j == 0:
            axs[i,j].set_xlabel("$\\alpha_a$")
        else:
            axs[i,j].set_xlabel("$\\alpha_o$")
            axs[i,j].legend(bbox_to_anchor=(1,1),loc='center left')
        axs[i,j].set_xticks(np.logspace(-4,4,5))
        
        
    if j == 0:
        axs[i,j].set_ylabel("$\sigma_{A_i}/\sigma_{A_{i,0}}$ channel "+rec[i])
        axs[i,j].set_ylim(ymin,1.5*np.max(data))
    if label:
        axs[i,j].legend()

fig,axs = plt.subplots(2,2,figsize=(14,8),sharex=True,sharey=True,gridspec_kw={"hspace":0,"wspace":0})
for noise_index in range(2):
    for AE_index in range(2):
        gb_influplot(alpha,stdv_all[noise_index,:,:,AE_index].T/stdv_all[noise_index,:,len(alpha)//2,AE_index],Ngalbins,axs,AE_index,noise_index)
    # gb_influplot(alpha,cov_matrix_diag[i]/cov_matrix_diag[i,0],l,Ngalbins,axs,x,y)
# fig.suptitle("Sigma vs alpha plot with normalisation")
plt.savefig("plots/sigma_alpha_simulated_normalised.pdf")
plt.close()


# Mask away all binaries that have stdv's of 0
print ("Making plot 2")
bin_mask = (np.sum(stdv_all < 1e-5,axis=2) == 0) #

# Use interpolate function or something...
from scipy.interpolate import interp1d

N_iters = 10001
x_testvalues = np.logspace(np.log10(alpha[0]),np.log10(alpha[-1]),N_iters)
i_increase = np.zeros((2,2,Ngalbins),dtype=int)
a_increase = np.zeros((2,2,Ngalbins))

for ns_i in range(2):
    for AE in range(2):
        for b in np.arange(Ngalbins)[bin_mask[ns_i,:,AE]]:
            # ([acc,oms],[Ngalbins],[alpha],[A,E])
            y_testvalues = np.interp(x_testvalues,alpha,stdv_all[ns_i,b,:,AE])
            
            arr = abs(y_testvalues-2*y_testvalues[N_iters//2])
            i_increase[ns_i,AE,b] = np.where(arr == np.min(arr))[0][0]
            a_increase[ns_i,AE,b] = x_testvalues[i_increase[ns_i,AE,b]]


# make second plots
lab = [["A acc", "E acc"],["A OMS", "E OMS"]]
lab2 = ["AE acc","AE OMS"]
cols = ['red','blue']
lnstls = ['-','--']

def sigdouble_freq_plot():
    plt.legend(loc=5)
    plt.xlabel("Frequency [mHz]")
    plt.ylabel("$\\alpha$ where $\sigma_{A_i}$ doubles")
    # plt.title("VB's to sample acceleration and OMS noise")
    # plt.xlim(4e-1,1e1)
    plt.xscale('log')
    plt.grid()


plt.figure(figsize=(10,6))
for ns_i in range(2):
    for AE in range(2):
        srtd_gb2 = (np.argsort(f_true[bin_mask[ns_i,:,AE]])[::-1])
        plt.errorbar(f_true[bin_mask[ns_i,:,AE]][srtd_gb2]*1e3, a_increase[ns_i,AE][bin_mask[ns_i,:,AE]][srtd_gb2]/np.max(a_increase[ns_i]), label=lab[ns_i][AE],
                     c=cols[ns_i], linestyle=lnstls[AE], marker='o', alpha=.8, linewidth=1)
sigdouble_freq_plot()
plt.savefig("plots/sigdouble_freq.pdf")
plt.close()

plt.figure(figsize=(10,6))
for ns_i in range(2):
    avg = np.mean(a_increase[ns_i],axis=0)
    std = np.std(a_increase[ns_i],axis=0)
    mask2 = bin_mask[ns_i,:,0] * bin_mask[ns_i,:,1]
    srtd_gb3 = (np.argsort(f_true[mask2])[::-1])
    
    plt.errorbar(f_true[mask2][srtd_gb3]*1e3, avg[mask2][srtd_gb3]/np.max(avg), yerr= std[mask2][srtd_gb3]/np.max(avg), label=lab2[ns_i], c=cols[ns_i], linestyle='--', marker='o', linewidth=1.5)

sigdouble_freq_plot()
plt.savefig("plots/sigdouble_freq2.pdf")
plt.close()

print ("All done!")