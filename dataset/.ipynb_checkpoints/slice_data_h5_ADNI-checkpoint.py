import os
import glob
import numpy as np
import torch
from tqdm.notebook import tqdm
from monai.data import ITKReader,NibabelReader
from monai.transforms import LoadImage, LoadImaged
from monai.transforms import (
    Orientationd, AddChanneld, Compose, ToTensord, Spacingd,Resized,ScaleIntensityD,ResizeWithPadOrCropd
    # ScaleIntensityD, ScaleIntensityRangeD, AdjustContrastD, RandAffined, ToNumpyd,RepeatChannelD
)
from monai.data import Dataset
import h5py
import threading


#         datalist_filenum = self.datalist[filenum]
#         loc_data = datalist_filenum['image']
#         masklist_filenum = self.masklist[filenum]
#         loc_mask = masklist_filenum['label']


# +
class H5CachedDataset(Dataset):
    def __init__(self, niilist_dict,masklist_dict,
                 transforms_offline, transforms_online=ToTensord(('image',)),
                 nslices_per_image = 362 ,
                 start_slice = 61,
                 end_slice = 91,
                 h5cachedir=None,
                padding_values = ((18,18), (0, 0), (0,0)) ):
        #### nslices_per_image ---> total slice in the volume
        #### h5cachedir ---> directory to save one .h5 files for each volume & it would act like a cache directory
        #### if h5cachedir does not exist create one 
#         self.lock = threading.Lock()
        if h5cachedir is not None:
            if not os.path.exists(h5cachedir):
                os.mkdir(h5cachedir)
            self.cachedir = h5cachedir
        self.padding_values = padding_values
        #### datalist ---> a list [{'image': volume_1_path},......,{'image': volume_n_path}]
        #### masklist ---> a list [{'mask': mask_1_path},......,{'image': mask_n_path}]
        self.datalist_dict = niilist_dict
        self.masklist_dict = masklist_dict
#         print("self.masklist_dict.keys()",self.masklist_dict.keys())
        self.xfms = transforms_offline
        self.xfms2 = ToTensord(tuple([str(key)+'_image' for key in self.datalist_dict]+\
                                     [str(key)+'_label' for key in self.masklist_dict]))
        #### 3d image loader from monai
        self.loader = LoadImage()
        self.loader.register(NibabelReader())  
        #### start_slice & end_slice---> slices to be truncated in each volume vol[:,:,start_slice:-end_slice]
        self.start_slice = start_slice
        self.end_slice = end_slice
        #### nslices ---> nslices_per_image - end_slice i.e. slice value after end truncation
        #            ---> nslices is kept flexible so that index is obtained by adding front truncation value &
        #            ---> total length of the loder is caluclated considering subtracting front truncation value
        self.nslices = nslices_per_image - self.end_slice
        
    def __len__(self):
        #### total number of slices in all the volumes
        key_list = [key for key in self.datalist_dict]
        return len(self.datalist_dict[key_list[0]])*(self.nslices - self.start_slice)
    
    def clear_cache(self):
        #### function to clear the directory storing h5 files (used for caching the h5 files)
        for fn in os.listdir(self.cachedir):
            os.remove(self.cachedir+'/'+fn)
            
    def __getitem__(self,index):
        #### ditionary to store data slicewise
        
        data = {}
        #### index can take values from 
                # 0 to (total number of volumes * (len(datalist)*(nslices - start_slice)))  
        #### filenum can take values from 0 to total number of volumes
        #### slicenum can take values from 0 to (len(datalist)*(nslices - start_slice))
        filenum = index // (self.nslices - self.start_slice)
        slicenum = index % (self.nslices - self.start_slice)
        slicenum += self.start_slice
#         print(slicenum,index)
        #### Extract the datafile location & mask file location based on filenum
        loc_data = {}
        
        for keys_ in self.datalist_dict.keys():
            loc_data.update({keys_:self.datalist_dict[keys_][filenum]['image']})
        
        ##### Include other information required for ADNI
        #'Subject','Aquisition Date','Age','Description','nii path'
        Subject = self.datalist_dict[keys_][filenum]['Subject']
        Aquisition_Date = self.datalist_dict[keys_][filenum]['Aquisition Date']
        Age = self.datalist_dict[keys_][filenum]['Age']
        Description = self.datalist_dict[keys_][filenum]['Description']
        nii_path = self.datalist_dict[keys_][filenum]['nii path']
        Sex = self.datalist_dict[keys_][filenum]['Sex']
        Health_status = self.datalist_dict[keys_][filenum]['Health status']
                            
        loc_mask = {}
        for keys_ in self.masklist_dict.keys():            
            loc_mask.update({keys_:self.masklist_dict[keys_][filenum]['label']})        
    ##### if h5 exists for the current volume fill data dictionary with current slice number
        if self.cachedir is not None:
            h5name = self.cachedir+'/%d.h5' % filenum
            ptname = self.cachedir+'/%d.pt' % filenum

            if os.path.exists(h5name):
                
                with h5py.File(h5name,'r',libver='latest', swmr=True) as itm:
                    for key in itm.keys():                       
                        data[key]=torch.from_numpy(itm[key][:,:,:,slicenum])
                data['image_meta_dict']={'affine':np.eye(3)} # FIXME: loses spacing info - can read from pt file


        ##### if data dictionary is empty
        if len(data)==0:
            data_i = {}
#             print("loc_data.keys()",loc_data.keys())
            #### Read image & mask data, meta data
            for keys_ in loc_data.keys():
                imgdata_, meta = self.loader(loc_data[keys_])
#                 print("imgdata shape",imgdata.shape)
                # Pad the array
                imgdata = imgdata_[11:-10,5:-4,:]
#                 imgdata = np.pad(imgdata_, self.padding_values, mode='constant')
                data_i.update({str(keys_)+'_image':imgdata})
                data_i.update({str(keys_)+'_image_meta':meta})
#             print(loc_mask)
            for keys_ in loc_mask.keys():
                mask_data_, mask_meta = self.loader(loc_mask[keys_])  
                # Pad the array
                mask_data = np.pad(mask_data_, self.padding_values, mode='constant')
                data_i.update({str(keys_)+'_label':mask_data})
#                 print(keys_)
                data_i.update({str(keys_)+'_label_meta':mask_meta})
#             mask_data, mask_meta = self.loader(loc_mask)
            #### store volume wise image & mask data,metadata in a dictionary 
#             data_i = {'image':imgdata,'label':mask_data, 'image_meta_dict':meta, 'label_meta_dict':mask_meta}
            #### transform the data dictionary
            data3d = self.xfms(data_i)
#             print(data3d.keys())
#             print(data_i.keys())
            #### Create h5 file for the volume by chunking into the slice shape for data & mask 
            #### Create a .pt file for meta data
            if self.cachedir is not None:
                other = {}

                with h5py.File(h5name,'w',libver='latest') as itm:
                    itm.swmr_mode = True
                    for key in data3d:
                        if key.split('_')[-1]=='image' or key.split('_')[-1]=='label':                             
                            img_npy = data3d[key].numpy()
                            if key.split('_')[-1]=='label':
                                img_npy = (img_npy>0).astype('uint8')
                            shp = img_npy.shape
                            chunk_size = list(shp[:-1])+[1]
                            ds = itm.create_dataset(key,shp,chunks=tuple(chunk_size),dtype=img_npy.dtype)
                            ds[:]=img_npy[:]
                            ds.flush()
                    else:
                        other[key]=data3d[key]
                torch.save(other,ptname)


            #### fill the data dictionary
            data = {}
            for keys_ in data3d.keys():
                if keys_.split('_')[-1]=='image' or keys_.split('_')[-1]=='label':
                    data.update({keys_:data3d[keys_][:,:,:,slicenum]})
                
            data.update({'image_meta_dict':{
                    'affine':np.eye(3)
                }})
#             data = {
#                 'image':data3d['image'][:,:,:,slicenum],
#                 'label':data3d['label'][:,:,:,slicenum],
#                 'image_meta_dict':{
#                     'affine':np.eye(3)
#                 }
#             }

            
        if len(data)>0:
#             print("**",data.keys())
#             res = self.xfms2(data)
            res = data
            for key in res:
#                 print(key,key.split('_')[-1]=='image')
                if key.split('_')[-1]=='image':
                    res[key]=res[key].float()
#                     print('hi')
                elif key.split('_')[-1]=='label':
                    res[key]=res[key].to(torch.int64)
#             res['filenum'] = filenum
            res['slicenum'] = slicenum
            res['idx'] = index
            res['Subject'] = Subject
            res['Aquisition Date'] = Aquisition_Date
            res['Age'] = Age
            res['Description'] = Description
            res['nii path'] = nii_path
            res['Sex'] = Sex
            res['Health status'] = Health_status
            
#             print("res keys",res.keys())
            return res
# -

#         else:
#             # replace with random
#             return self.__getitem__(np.random.randint(len(self.datalist)))


