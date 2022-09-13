# By Bart Ouwehand 22-04-2022
from my_functions import *

orig_duration = dur_range[-1]
new_durations = dur_range[:-1]
#fp = 'measurements/tm_asds/'
create_fsdata = False

trunc_fsdata = True
trunc_mdata = True


if trunc_fsdata:
    for ns_src in noise_source:
        fp = 'measurements/'+ns_src+'/'
        print ("Exporting sample data "+ns_src)
        for i,alph in enumerate(alpha):
            print ("Sample with alpha = {}".format(alph))
            alph_nm = alpha_nm[i]
            for j in tqdm(range(N1)):
                # If the original dataset only has sample data and not filtered sample data
                if create_fsdata:
                    inputf = fp+str(orig_duration)+'d/'+alph_nm+"/s"+str(j)+'.txt'
                    rawdata = ascii.read(inputf)
                    old_sdata = np.array([rawdata['t'],rawdata['A'],rawdata['E']])

                    tmp = []
                    coeffs = scipy.signal.firls(73, bands=[0,1e-2,1.5e-2,fs/2], desired=[1,1,0,0],fs=fs)
                    for k in range(1,3):
                        fdata_tmp = scipy.signal.filtfilt(coeffs,1., x=old_sdata[k],padlen=(old_sdata.shape[1]//2)+1)
                        tmp.append(fdata_tmp[cutoff:-cutoff])
                    old_fsdata = np.array([old_sdata[0][cutoff:-cutoff],tmp[0],tmp[1]])

                    filepath = fp+str(orig_duration)+'d/'+alph_nm+"/fs"+str(j)+'.txt'
                    filecontent = Table(old_fsdata.T, names=['t','A','E'])
                    ascii.write(filecontent, filepath, overwrite=True)

                inputf = fp+str(orig_duration)+'d/'+alph_nm+"/fs"+str(j)+'.txt'
                rawdata = ascii.read(inputf)
                old_fsdata = np.array([rawdata['t'],rawdata['A'],rawdata['E']])

                for k,nd in enumerate(new_durations):
                    new_fsdata = old_fsdata[:,:int(size[k]-(discard+2*cutoff))]

                    outputf = fp+str(int(nd))+'d/'+alph_nm+"/fs"+str(j)+'.txt'
                    filecontent = Table(new_fsdata.T, names=['t','A','E'])
                    ascii.write(filecontent, outputf, overwrite=True)

        print ("Sample data {} truncated!".format(ns_src))

if trunc_mdata:
    for ns_src in noise_source:
        fp = 'measurements/'+ns_src+'/'
        print ("Exporting model data "+ns_src)
        names = ['t','A','E']

        inputf = fp+str(orig_duration)+'d/1/fs0.txt'
        rawdata = ascii.read(inputf)
        old_fsdata = np.array([rawdata['t'],rawdata['A'],rawdata['E']])

        for i in tqdm(range(Ngalbins)):
            inputf = fp+str(orig_duration)+'d/mdata/binary_'+str(i)+'.txt'
            
            rawdata = ascii.read(inputf)
            old_mdata = np.array([rawdata['t'],rawdata['A'],rawdata['E']])

            for k,nd in enumerate(new_durations):
                new_mdata = old_mdata[:,:int(size[k]-(discard+2*cutoff))]

                outputf = fp+str(int(nd))+'d/mdata/binary_'+str(i)+'.txt'
                filecontent = Table(new_mdata.T, names=names)
                ascii.write(filecontent, outputf, overwrite=True)

        print ("Model data {} truncated!".format(ns_src))
print ("All done")

