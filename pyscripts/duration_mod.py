# By Bart Ouwehand 22-04-2022
from my_functions import *

orig_duration = dur_range[-1]
new_durations = dur_range[:-1]
fp = ext+'measurements/tm_asds/'
create_fsdata = False

trunc_fsdata = True
trunc_mdata = True

if trunc_fsdata:
    print ("Exporting sample data")
    for i,alph in enumerate(tqdm(alpha)):
        for j in range(N1):
            # If the original dataset only has sample data and not filtered sample data
            if create_fsdata:
                inputf = fp+str(orig_duration)+'d/'+str(int(alph))+"/s"+str(j)+'.txt'
                rawdata = ascii.read(inputf)
                old_sdata = np.array([rawdata['t'],rawdata['A'],rawdata['E']])

                tmp = []
                coeffs = scipy.signal.firls(73, bands=[0,1e-2,1.5e-2,fs/2], desired=[1,1,0,0],fs=fs)
                for k in range(1,3):
                    fdata_tmp = scipy.signal.filtfilt(coeffs,1., x=old_sdata[k],padlen=(old_sdata.shape[1]//2)+1)
                    tmp.append(fdata_tmp[cutoff:-cutoff])
                old_fsdata = np.array([old_sdata[0][cutoff:-cutoff],tmp[0],tmp[1]])

                filepath = fp+str(orig_duration)+'d/'+str(int(alph))+"/fs"+str(j)+'.txt'
                filecontent = Table(old_fsdata.T, names=['t','A','E'])
                ascii.write(filecontent, filepath, overwrite=True)

            inputf = fp+str(orig_duration)+'d/'+str(int(alph))+"/fs"+str(j)+'.txt'
            rawdata = ascii.read(inputf)
            old_fsdata = np.array([rawdata['t'],rawdata['A'],rawdata['E']])

            for k,nd in enumerate(new_durations):
                new_fsdata = old_fsdata[:,:int(size[k])]

                outputf = fp+str(int(nd))+'d/'+str(int(alph))+"/fs"+str(j)+'.txt'
                filecontent = Table(new_fsdata.T, names=['t','A','E'])
                ascii.write(filecontent, outputf, overwrite=True)

    print ("Sample data truncated!")

if trunc_mdata:
    print ("Exporting model data")
    names = ['t']
    for k in range(N2):
        names.extend(['A'+str(k),'E'+str(k)])

    for i in tqdm(range(Ngalbins)):
        inputf = fp+str(orig_duration)+'d/mdata/binary_'+str(i)+'.txt'

        rawdata = ascii.read(inputf)
        # print (rawdata)
        old_mdata_1b = np.zeros((1+N2*2,len(old_fsdata[0])))
        old_mdata_1b[0] = rawdata['t']
        for j in range(N2):
            old_mdata_1b[1+j*2:3+j*2] = np.array([rawdata['A'+str(j)],rawdata['E'+str(j)]])

        for k,nd in enumerate(new_durations):
            new_mdata_1b = old_mdata_1b[:,:int(size[k])]

            outputf = fp+str(int(nd))+'d/mdata/binary_'+str(i)+'.txt'
            filecontent = Table(new_mdata_1b.T, names=names)
            ascii.write(filecontent, outputf, overwrite=True)

    print ("Model data truncated!")
print ("All done")

