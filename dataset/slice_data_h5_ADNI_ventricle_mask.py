import os
import glob
import numpy as np
import torch
from tqdm.notebook import tqdm
from monai.data import ITKReader,NibabelReader
from monai.transforms import LoadImage, LoadImaged
from monai.transforms import (
    Orientationd, EnsureChannelFirst, Compose, ToTensord, Spacingd,Resized,ScaleIntensityD,ResizeWithPadOrCropd
    # ScaleIntensityD, ScaleIntensityRangeD, AdjustContrastD, RandAffined, ToNumpyd,RepeatChannelD
)
from monai.data import Dataset
import h5py
import threading
import os
# from metrics_custom import normalize_tensor
import cv2
from skimage.segmentation import chan_vese


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
                padding_values = ((18,18), (0, 0), (0,0)),\
                ventricle_mask_root_path = '/storage/ayantika/Ayantika/Data_final/ADNI_ventricle_mask/'):
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
        self.loader = LoadImage(image_only=False)
        self.loader.register(NibabelReader())  
        #### start_slice & end_slice---> slices to be truncated in each volume vol[:,:,start_slice:-end_slice]
        self.start_slice = start_slice
        self.end_slice = end_slice
        #### nslices ---> nslices_per_image - end_slice i.e. slice value after end truncation
        #            ---> nslices is kept flexible so that index is obtained by adding front truncation value &
        #            ---> total length of the loder is caluclated considering subtracting front truncation value
        self.nslices = nslices_per_image - self.end_slice
        
        self.ventricle_mask_root_path = ventricle_mask_root_path
        if not os.path.exists(self.ventricle_mask_root_path):
            os.mkdir(self.ventricle_mask_root_path)
        
    def get_ventricle(self,img_):
        cv = chan_vese(img_, mu=0.2, lambda1=1, lambda2=1, tol=1e-3,
                       max_num_iter=200, dt=0.5, init_level_set="checkerboard",
                       extended_output=True)

        img_mask = (img_>0)
        kernel = np.ones((11, 11), np.uint8) 
        eroded_image = cv2.erode(img_mask.astype(np.uint8), kernel, iterations=1)
        ventricle_mask = np.logical_not(cv[0]) * eroded_image
        return ventricle_mask,cv
    
    def normalise_(self,img):
        return ((img- img.min())/(img.max() - img.min()))
    
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
        loc_data_baseline = {}
        loc_data = {}

        for keys_ in self.datalist_dict.keys():
            loc_data_baseline.update({keys_:self.datalist_dict[keys_][filenum]['baseline_image']})
            loc_data.update({keys_:self.datalist_dict[keys_][filenum]['image']})
            

        ##### Include other information required for ADNI
        #'Subject','Aquisition Date','Age','Description','nii path'
        Subject = self.datalist_dict[keys_][filenum]['Subject']
        Aquisition_Date = self.datalist_dict[keys_][filenum]['Aquisition Date']
        baseline_Aquisition_Date = self.datalist_dict[keys_][filenum]['baseline_Aquisition Date']
        Age = self.datalist_dict[keys_][filenum]['Age']
        baseline_Age = self.datalist_dict[keys_][filenum]['baseline_Age']
        Description = self.datalist_dict[keys_][filenum]['Description']
        nii_path = self.datalist_dict[keys_][filenum]['nii path']
        baseline_nii_path = self.datalist_dict[keys_][filenum]['baseline nii path']
        Sex = self.datalist_dict[keys_][filenum]['Sex']
        Health_status = self.datalist_dict[keys_][filenum]['Health status']
        
        ################## Added the path to ventricle mask addition
        store_name = Subject+'_'+str(round(float(Age),2))+'_'+str(round(float(baseline_Age),2))+'_'+str(int(slicenum))
        
        ventricle_mask_path = self.ventricle_mask_root_path+store_name+'.pt'
        ######################
                            
        loc_mask = {}
        loc_mask_baseline = {}
        for keys_ in self.masklist_dict.keys():            
            loc_mask.update({keys_:self.masklist_dict[keys_][filenum]['label']})  
            loc_mask_baseline.update({keys_:self.masklist_dict[keys_][filenum]['baseline_label']})  
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
                imgdata_baseline_, meta_baseline = self.loader(loc_data_baseline[keys_])
#                 print("imgdata shape",imgdata.shape)
                # Pad the array
                imgdata = imgdata_[11:-10,5:-4,:]
                imgdata_baseline = imgdata_baseline_[11:-10,5:-4,:]
#                 imgdata = np.pad(imgdata_, self.padding_values, mode='constant')
                data_i.update({str(keys_)+'_image':imgdata})
                data_i.update({str(keys_)+'_image_meta':meta})
                data_i.update({str(keys_)+'_baseline_image':imgdata_baseline})
                data_i.update({str(keys_)+'_baseline_image_meta':meta_baseline})
#             print(loc_mask)
            for keys_ in loc_mask.keys():
                mask_data_, mask_meta = self.loader(loc_mask[keys_]) 
                mask_data_baseline_, mask_meta_baseline_ = self.loader(loc_mask_baseline[keys_]) 
                mask_data = mask_data_[11:-10,5:-4,:]
                mask_data_baseline = mask_data_baseline_[11:-10,5:-4,:]
                # Pad the array
#                 mask_data = np.pad(mask_data_, self.padding_values, mode='constant')
                data_i.update({str(keys_)+'_label':mask_data})
                data_i.update({str(keys_)+'_label_meta':mask_meta})
                data_i.update({str(keys_)+'_baseline_label':mask_data_baseline})
                data_i.update({str(keys_)+'_baseline_label_meta':mask_meta_baseline_})
#             mask_data, mask_meta = self.loader(loc_mask)
            #### store volume wise image & mask data,metadata in a dictionary 
#             data_i = {'image':imgdata,'label':mask_data, 'image_meta_dict':meta, 'label_meta_dict':mask_meta}
            #### transform the data dictionary
#             print("inside",data_i.keys())
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
#                             print("data3d[key],key",data3d[key],key)
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
            if not os.path.exists(ventricle_mask_path):
                print(ventricle_mask_path)
                for keys__ in data.keys():
                    if '_train_' in keys__:
                        mode_key = '_train_'
                    elif '_test_' in keys__:
                        mode_key = '_test_'
                followup_ = data[mode_key+'image']
                img_numpy = self.normalise_(followup_[0,:,:]).numpy()
                ventricle_mask,cv  = self.get_ventricle(img_numpy)
                ventricle_mask_tensor = torch.tensor(ventricle_mask).unsqueeze(dim=0).to(torch.float32)
                torch.save(ventricle_mask_tensor,ventricle_mask_path)
            ventricle_mask = torch.load(ventricle_mask_path)
            data.update({'ventricle_mask':ventricle_mask})
            
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
            if not os.path.exists(ventricle_mask_path):
                print(ventricle_mask_path)
                for keys__ in data.keys():
                    if '_train_' in keys__:
                        mode_key = '_train_'                        
                    elif '_test_' in keys__:
                        mode_key = '_test_'
                followup_ = data[mode_key+'image']
                img_numpy = self.normalise_(followup_[0,:,:]).numpy()
                ventricle_mask,cv  = self.get_ventricle(img_numpy)
                ventricle_mask_tensor = torch.tensor(ventricle_mask).unsqueeze(dim=0).to(torch.float32)
                torch.save(ventricle_mask_tensor,ventricle_mask_path)
            ventricle_mask = torch.load(ventricle_mask_path)
            data.update({'ventricle_mask':ventricle_mask})
            
            res = data
            for key in res:
#                 print(key,key.split('_')[-1]=='image')
                if key.split('_')[-1]=='image':
                    res[key]=res[key].float()
#                     print('hi')
                elif key.split('_')[-1]=='label':
                    res[key]=res[key].to(torch.int64)

            res['ventricle_mask'] = data['ventricle_mask'].to(torch.float32)
            res['slicenum'] = slicenum
            res['idx'] = index
            res['Subject'] = Subject
            res['Aquisition Date'] = Aquisition_Date
            res['baseline Aquisition Date'] = baseline_Aquisition_Date
            
            res['Age'] = Age
            res['baseline Age'] = baseline_Age
            
            res['Description'] = Description
            res['nii path'] = nii_path
            res['baseline nii path'] = baseline_nii_path
            
            res['Sex'] = Sex
            res['Health status'] = Health_status
            
#             print("res keys",res.keys())
            return res
# -

#         else:
#             # replace with random
#             return self.__getitem__(np.random.randint(len(self.datalist)))


