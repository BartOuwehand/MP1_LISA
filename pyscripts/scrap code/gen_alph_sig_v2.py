# By Bart Ouwehand 10-09-2022

from my_functions import *

calc_L_again = True
# calc_L_again = False

print('RAM memory used: {} %'.format(psutil.virtual_memory()[2]))
# Define the likelyhood functions
def lnL(sdata,mt,mdata,bin_i, amp_i):
    """
    The log likelihood of our simplistic Linear Regression. 
    """
    st, sA, sE = sdata
    mA, mE = mdata
    # if st != mt:
        # print ("Time arrays do not equal for binary {}, amplitude {}".format(bin_i,amp_i))
    
    return (np.sum((sA-mA)**2)) + (np.sum((sE-mE)**2))


# Define the parabula functions
def sig_func(x0,xpm,f0,fpm):
    return np.abs(xpm-x0) / np.sqrt(fpm-f0)

def parabula_err(x0, xpm, L_range,N3):
    """Finds error of likelyhood parabula. Input must be the optimal value and the offset from this for which to calculate the parabula. The output is sigma"""
    f0, fm, fp = L_range[N3//2], L_range[:N3//2], L_range[(N3-N3//2):]
    xm, xp = xpm[:N3//2], xpm[(N3-N3//2):]
    
    sigs_m = sig_func(x0,xm,f0,fm)
    sigs_p = sig_func(x0,xp,f0,fp)
    return np.nanmean([sigs_m,sigs_p])

def plot_bin_L(x,y,b,j,fptmp):
    plt.figure(figsize=(8,6))
    for j in range(N1):
        plt.plot(x/Amp_true[b],y,label="iter {}".format(j))
    plt.xlabel("Amplitude amplification [%]")
    plt.ylabel("-ln(Likelyhood)")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc=1)
    # plt.legend(bbox_to_anchor=(0.95,1), loc="upper left")
    plt.savefig(fptmp+'Amp_Lrange_b'+str(b)+'.jpg')
    plt.close()

def plot_sig_v_alph_indiv(alpha,data,b,dr):
    plt.figure(figsize=(12,8))
    for b in range(Ngalbins):
        plt.errorbar(alpha, np.nanmean(data[:,:,b],axis=1), yerr=np.nanstd(data[:,:,b],axis=1),label="bin {}".format(b),fmt='o--',c=(f_true[b]/f_true[0],0,0))
    plt.xlabel("Alpha")
    plt.ylabel("Sigma$_A$")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.savefig("plots/Sig_vs_alph_indiv_"+str(dr)+"d.jpg")
    plt.close()


# Create the model signals for all possibilities of amplitudes for each binary.
iopt = N2
fp1 = ext+'measurements/tm_asds/'
sigmas = np.zeros((len(dur_range),len(alpha),N1,Ngalbins))
L_range_tot = np.zeros((len(dur_range),len(alpha),N1,Ngalbins,N2))

if calc_L_again:
    # for d,dr in enumerate(dur_range):
    for d,dr in enumerate(dur_range):
        print ("Starting calculations for {} d".format(dr))
        rawdata = ascii.read(fp1+str(dr)+'d/1/fs0.txt')
        tmp_fsdata = np.array([rawdata['t'],rawdata['A'],rawdata['E']])
        len_data = len(ascii.read(fp1+str(dr)+'d/1/fs0.txt')['t'])

        mdata_tot = np.zeros((Ngalbins,N2,2,len_data))

        sigmas_1d = np.zeros((len(alpha),N1,Ngalbins))

        fp2 = fp1+str(dr)+"d/mdata/binary_"
        all_mdata = np.zeros((Ngalbins,1+2*N2,len_data)) #[binary,[t,A0,E0,A1,E1]]
        for j in range(Ngalbins):
            mdata_1b = np.zeros((1+2*N2,len_data))
            rawdata = ascii.read(fp2+str(j)+'.txt')
            mdata_1b[0] = rawdata['t']
            for i in range(N2):
                mdata_1b[1+i*2:3+i*2] = np.array([rawdata['A'+str(i)],rawdata['E'+str(i)]])
            all_mdata[j] = mdata_1b

        model_time = all_mdata[0][0]
        for i in range(Ngalbins):
            for j in range(N2):
                mdata_tot[i,j,0] = all_mdata[i,1+2*j] + (np.sum(all_mdata[:,1+iopt],axis=0)-all_mdata[i,1+iopt])
                mdata_tot[i,j,1] = all_mdata[i,2+2*j] + (np.sum(all_mdata[:,2+iopt],axis=0)-all_mdata[i,2+iopt])

        print ("Start likelyhood calculations")

        L_range = np.zeros((len(alpha),N1,Ngalbins,N2))
        for i,alph in enumerate(alpha):
            fp3 = 'plots/{}d/{}/'.format(dr,alph)
            for j in range(N1):
                rawdata = ascii.read(fp1+str(dr)+'d/'+str(alph)+'/fs'+str(j)+'.txt')
                fsdata = np.array([rawdata['t'],rawdata['A'],rawdata['E']])

                L_range_1a_1i = np.zeros((Ngalbins,N2))
                for b in range(Ngalbins):
                    for k in range(N2):
                        L_range_1a_1i[b,k] = lnL(fsdata,model_time,mdata_tot[b,k],b,k)
                    
                # Calculate the parabula using the inner N3 points (N3 \leq N2)
                N3 = 5
                i_in, i_out = N2//2-(N3//2), N2//2+(N3-N3//2)
                if N3 > N2:
                    print ("N3>N2 so L-space not sampled enough")

                sigs_1a_1i = np.zeros(Ngalbins)
                for b in range(Ngalbins):    
                    sig_tmp = parabula_err(Amp_true[b],Amp_range[b,i_in:i_out],L_range_1a_1i[b,i_in:i_out],N3)
                    sigs_1a_1i[b] = sig_tmp

                L_range[i,j] = L_range_1a_1i
                sigmas_1d[i,j] = sigs_1a_1i

            # Plot the likelyhoodranges for different binaries
            for b in range(Ngalbins):
                plot_bin_L(Amp_range[b],L_range[i,j,b],b,j,fp3)
                
                # Save the plotted data to a file
                filename = fp1+str(dr)+'d/'+str(alph)+'/L_range_b'+str(b)+'.txt'
                filecontent = Table(L_range[i,:,b].T,names=np.arange(N1,dtype=str))
                ascii.write(filecontent, filename, overwrite=True)
            
            # Save the sigmas to a file
            filename = fp1+str(dr)+'d/sigmas_alpha'+str(alph)+'.txt'
            filecontent = Table(sigmas_1d[i].T,names=np.arange(N1,dtype=str))
            ascii.write(filecontent,filename,overwrite=True)
        sigmas[d] = sigmas_1d
        L_range_tot[d] = L_range
        
        # Plot individual change in sigmas over alpha
        plot_sig_v_alph_indiv(alpha,sigmas_1d,b,dr)

else:
    for d,dr in enumerate(dur_range):
        for i,alph in enumerate(alpha):
            for b in range(Ngalbins):
                filename = fp1+str(dr)+'d/'+str(alph)+'/L_range_b'+str(b)+'.txt'
                rawdata = ascii.read(filename)
                for j in range(N1):
                    L_range_tot[d,i,j,b] = np.array(rawdata['col'+str(1+j)])[1:]
            
            filename = fp1+str(dr)+'d/sigmas_alpha'+str(alph)+'.txt'
            rawdata = ascii.read(filename)
            for j in range(N1):
                sigmas[d,i,j] = np.array(rawdata['col'+str(1+j)])[1:]
            
            # plot the binary likelyhood ranges
            fp3 = 'plots/{}d/{}/'.format(dr,alph)
            plot_bin_L(Amp_range[b],L_range_tot[d,i,j,b],b,j,fp3)
        
        # Plot the individual binary sigmas over the alpha range
        plot_sig_v_alph_indiv(alpha,sigmas[d],b,dr)

# print (L_range_tot)
print (sigmas)



plt.figure(figsize=(12,8))
for d,dr in enumerate(dur_range):
    avg = np.nanmean(np.nanmean(sigmas[d],axis=2),axis=1)
    stdv = np.nanstd(np.nanmean(sigmas[d],axis=2),axis=1)
    plt.errorbar(alpha,avg,yerr=stdv,label="{} d".format(dr),fmt='o--',capsize=5,capthick=1)
plt.xlabel("Alpha")
plt.ylabel("sig")
plt.xscale("log")
plt.title("Error in Amplitudes for different noise levels")
plt.legend()
plt.savefig("plots/masterplot2.jpg")
plt.close()

print('RAM memory used: {} %'.format(psutil.virtual_memory()[2]))