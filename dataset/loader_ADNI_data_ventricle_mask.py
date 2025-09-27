# import sys
# import monai.transforms as transforms
# from monai.transforms import (
# Orientationd, EnsureChannelFirst, Compose, ToTensord, Spacingd,Resized,ScaleIntensityD,ResizeWithPadOrCropd
# # ScaleIntensityD, ScaleIntensityRangeD, AdjustContrastD, RandAffined, ToNumpyd,RepeatChannelD
# )
# import pandas as pd
# from omegaconf import OmegaConf
# import numpy as np
# import os
# # sys.path.insert(0,'/storage/Ayantika/Diffusion_AE_medical/')
# # import dataset.slice_data_h5_ADNI_with_baseline as sdl
# import sys
# sys.path.insert(0,'/storage/ayantika/Ayantika/Diff_AE_xstart_w_xbsln_disentangle_unsup/dataset/')
# import slice_data_h5_ADNI_ventricle_mask as sdl


import pandas as pd
import csv
from omegaconf import OmegaConf
import os
import sys
import monai.transforms as transforms
from monai.transforms import (
Orientationd, EnsureChannelFirst, Compose, ToTensord, Spacingd,Resized,ScaleIntensityD,ResizeWithPadOrCropd
# ScaleIntensityD, ScaleIntensityRangeD, AdjustContrastD, RandAffined, ToNumpyd,RepeatChannelD
)
import pandas as pd
from omegaconf import OmegaConf
import numpy as np
import os
# sys.path.insert(0,'/storage/Ayantika/Diffusion_AE_medical/')
# import dataset.slice_data_h5_ADNI_with_baseline as sdl
import sys
sys.path.insert(0,'./dataset/')
import slice_data_h5_ADNI_ventricle_mask as sdl


# Custom MapTransform to work on dictionaries
class CustomEnsureChannelFirst(transforms.MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        self.ensure_channel_first = EnsureChannelFirst()

    def __call__(self, data):
        for key in self.keys:
            print(key)
            data[key] = self.ensure_channel_first(data[key])
        return data

class NormalizeIntensityByClippingD(transforms.MapTransform):
    def __init__(self, keys, percentile=99.5):
        super().__init__(keys)
        self.percentile = percentile

    def __call__(self, data):
        for key in self.keys:
            image = data[key]
            v_99_5 = np.percentile(image, self.percentile)
            image = np.clip(image, 0, v_99_5)
            data[key] = image
        return data

class ADNI_dataloader():
    def __init__(self,csv_path,path,h5_save_path):
        self.csv_path = csv_path
        self.path = path
        self.h5_save_path = h5_save_path

    # Define the custom transform

    def get_transform_nii_vols(self,key_list_image,key_list_label):
#         print("in fuction",key_list_image+key_list_label)
#         tuple(key_list_image+key_list_label)
        transforms_ = Compose(
            [
             CustomEnsureChannelFirst(tuple(key_list_image+key_list_label)),
             Orientationd(tuple(key_list_image+key_list_label),'RAS'),
        #      Spacingd(('image','label'),(1,1,1)),        
             Resized(keys = (tuple(key_list_image)),spatial_size = (self.config.dataloader.img_height, self.config.dataloader.img_width,-1),\
                     mode = 'trilinear' ,align_corners = True),
             Resized(keys = (tuple(key_list_label)),spatial_size = (self.config.dataloader.img_height, self.config.dataloader.img_width,-1),\
                     mode = 'nearest' ),
             NormalizeIntensityByClippingD(tuple(key_list_image)),
             ScaleIntensityD(tuple(key_list_image), minv=-1, maxv=1),
             ToTensord(tuple(key_list_image+key_list_label)),
            ]
        )

        return transforms_
    
    
    def get_list_for_dataloader(self,tuple_,MCI_M_dict,key_='image'):
        datalist = {}
        list_for_dict = []
        for i,key in enumerate(tuple_):
        #     list_image_paths = glob.glob(conf.config.dataloader.training_path[i])
        #     list_image_paths.sort()
            for indx in range(0,len(MCI_M_dict['Subject'])):

                sub_indx_list = []
                for sub_indx in range(0,len(MCI_M_dict['Subject'])):
                    if (MCI_M_dict['Subject'][sub_indx] == MCI_M_dict['Subject'][indx]) and (MCI_M_dict['Age'][sub_indx] > MCI_M_dict['Age'][indx]):

                        sub_indx_list.append(sub_indx)

                if len(sub_indx_list)!=0:
                    indx_final = np.random.randint(0,len(sub_indx_list),1)[0]
                    indx_next_ = sub_indx_list[indx_final]
    #                 print(MCI_M_dict['Subject'][indx_next_] , MCI_M_dict['Subject'][indx])
    #                 print(MCI_M_dict['Age'][indx_next_] , MCI_M_dict['Age'][indx])

                else:
                    indx_next_ = indx

                key_final ='baseline_' + key_
                
                list_for_dict.append( \
                                     {key_final :MCI_M_dict['nii path'][indx],\
                                        key_:MCI_M_dict['nii path'][indx_next_],\
                                        'Subject': MCI_M_dict['Subject'][indx],\
                                        'baseline_Aquisition Date':MCI_M_dict['Aquisition Date'][indx],\
                                        'Aquisition Date':MCI_M_dict['Aquisition Date'][indx_next_],\
                                        'baseline_Age':MCI_M_dict['Age'][indx],\
                                        'Age':MCI_M_dict['Age'][indx_next_],\
                                        'Health status':MCI_M_dict['Health status'][indx],\
                                        'Description':MCI_M_dict['Description'][indx],\
                                        'Sex':MCI_M_dict['Sex'][indx],\
                                        'baseline nii path':MCI_M_dict['nii path'][indx],\
                                        'nii path':MCI_M_dict['nii path'][indx_next_],\
                                     })

            datalist.update({key:list_for_dict})
            return datalist
        
    def dict_list_csv(self,datalist,csv_file_name):    
        field_names = datalist['_train'][0].keys()

        with open(csv_file_name, mode='w',newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names)

            # Write the header
            writer.writeheader()

            # Write the data rows
            for row in datalist['_train']:
                writer.writerow(row)
        
    def create_pair_info_csv(self, csv_file_name, csv_mask_name):
    
    
        MCI_M_csv = pd.read_csv( self.csv_path)
        MCI_M_dict = MCI_M_csv.to_dict()
        self.config_path = self.path
        self.config = OmegaConf.load(self.config_path)
#         self.config.dataloader.h5cachedir_ = '/storage/Ayantika/h5data_store_with_DiffAE_output'
            
        datalist = self.get_list_for_dataloader(self.config.dataloader.key_tuple_image,MCI_M_dict,key_='image')
        masklist = self.get_list_for_dataloader(self.config.dataloader.key_tuple_mask,MCI_M_dict,key_='label')
        
        self.dict_list_csv(datalist,csv_file_name)
        self.dict_list_csv(masklist,csv_mask_name)
        
    def make_dict_list_from_csv(self, csv_file_name,mode='train'):
        csv_file = pd.read_csv(csv_file_name)
        data_list = []

        # Open the CSV file for reading
        with open(csv_file_name, mode='r', newline='') as csv_file:
            reader = csv.DictReader(csv_file)

            # Iterate through the rows and append each row dictionary to the list
            for row in reader:
                data_list.append(row)



        return {'_'+mode:data_list}
        
    def main_call(self,csv_file_name,csv_mask_name,ventricle_mask_root_path,mode='train'):
        self.ventricle_mask_root_path = ventricle_mask_root_path
        
        MCI_M_csv = pd.read_csv( self.csv_path)
        MCI_M_dict = MCI_M_csv.to_dict()
        self.config_path = self.path
        self.config = OmegaConf.load(self.config_path)
#         self.config.dataloader.h5cachedir_ = '/storage/Ayantika/h5data_store_with_DiffAE_output'
            
        datalist = self.make_dict_list_from_csv(csv_file_name,mode)
#         print('datalist',datalist)
        masklist = self.make_dict_list_from_csv(csv_mask_name,mode)
        key_list_image = []
        key_list_label = []
        for keys_ in datalist.keys():
            key_list_image.append(str(keys_)+'_image')
            key_list_image.append(str(keys_)+'_baseline_image')
        for keys_ in masklist.keys():
            key_list_label.append(str(keys_)+'_label')
            key_list_label.append(str(keys_)+'_baseline_label')
#         print(key_list_image,key_list_label)

        transforms_ = self.get_transform_nii_vols(key_list_image,key_list_label)
        self.config.dataloader.h5cachedir_ = self.h5_save_path
        ## The loader is such that it would create h5 files if they are not created when the loader is called and executed
        if not os.path.exists(self.config.dataloader.h5cachedir_):
            os.mkdir(self.config.dataloader.h5cachedir_)
        h5cacheds = sdl.H5CachedDataset(datalist,masklist,transforms_,h5cachedir= self.config.dataloader.h5cachedir_,\
                                           nslices_per_image=self.config.dataloader.total_slices_per_vol,\
                                           start_slice = self.config.dataloader.start_slice_per_vol,\
                                        end_slice = self.config.dataloader.end_slice_per_vol,\
                                       ventricle_mask_root_path = self.ventricle_mask_root_path)
        return h5cacheds
        
        
        


# class NormalizeIntensityByClippingD(transforms.MapTransform):
#     def __init__(self, keys, percentile=99.5):
#         super().__init__(keys)
#         self.percentile = percentile

#     def __call__(self, data):
#         for key in self.keys:
#             image = data[key]
#             v_99_5 = np.percentile(image, self.percentile)
#             image = np.clip(image, 0, v_99_5)
#             data[key] = image
#         return data

# class ADNI_dataloader():
#     def __init__(self,csv_path,path,h5_save_path):
#         self.csv_path = csv_path
#         self.path = path
#         self.h5_save_path = h5_save_path

#     # Define the custom transform

#     def get_transform_nii_vols(self,key_list_image,key_list_label):
#         transforms_ = Compose(
#             [
#              EnsureChannelFirst(tuple(key_list_image+key_list_label)),
#              Orientationd(tuple(key_list_image+key_list_label),'RAS'),
#         #      Spacingd(('image','label'),(1,1,1)),        
#              Resized(keys = (tuple(key_list_image)),spatial_size = (self.config.dataloader.img_height, self.config.dataloader.img_width,-1),\
#                      mode = 'trilinear' ,align_corners = True),
#              Resized(keys = (tuple(key_list_label)),spatial_size = (self.config.dataloader.img_height, self.config.dataloader.img_width,-1),\
#                      mode = 'nearest' ),
#              NormalizeIntensityByClippingD(tuple(key_list_image)),
#              ScaleIntensityD(tuple(key_list_image), minv=-1, maxv=1),
#              ToTensord(tuple(key_list_image+key_list_label)),
#             ]
#         )

#         return transforms_
    
    
#     def get_list_for_dataloader(self,tuple_,MCI_M_dict,key_='image'):
#         datalist = {}
#         list_for_dict = []
#         for i,key in enumerate(tuple_):
#         #     list_image_paths = glob.glob(conf.config.dataloader.training_path[i])
#         #     list_image_paths.sort()
#             for indx in range(0,len(MCI_M_dict['Subject'])):

#                 sub_indx_list = []
#                 for sub_indx in range(0,len(MCI_M_dict['Subject'])):
#                     if (MCI_M_dict['Subject'][sub_indx] == MCI_M_dict['Subject'][indx]) and (MCI_M_dict['Age'][sub_indx] > MCI_M_dict['Age'][indx]):

#                         sub_indx_list.append(sub_indx)

#                 if len(sub_indx_list)!=0:
#                     indx_final = np.random.randint(0,len(sub_indx_list),1)[0]
#                     indx_next_ = sub_indx_list[indx_final]
#     #                 print(MCI_M_dict['Subject'][indx_next_] , MCI_M_dict['Subject'][indx])
#     #                 print(MCI_M_dict['Age'][indx_next_] , MCI_M_dict['Age'][indx])

#                 else:
#                     indx_next_ = indx

#                 key_final ='baseline_' + key_
                
#                 list_for_dict.append( \
#                                      {key_final :MCI_M_dict['nii path'][indx],\
#                                         key_:MCI_M_dict['nii path'][indx_next_],\
#                                         'Subject': MCI_M_dict['Subject'][indx],\
#                                         'baseline_Aquisition Date':MCI_M_dict['Aquisition Date'][indx],\
#                                         'Aquisition Date':MCI_M_dict['Aquisition Date'][indx_next_],\
#                                         'baseline_Age':MCI_M_dict['Age'][indx],\
#                                         'Age':MCI_M_dict['Age'][indx_next_],\
#                                         'Health status':MCI_M_dict['Health status'][indx],\
#                                         'Description':MCI_M_dict['Description'][indx],\
#                                         'Sex':MCI_M_dict['Sex'][indx],\
#                                         'baseline nii path':MCI_M_dict['nii path'][indx],\
#                                         'nii path':MCI_M_dict['nii path'][indx_next_],\
#                                      })

#             datalist.update({key:list_for_dict})
#             return datalist
        
        
        
    
    

#     def main_call(self):
        
#         MCI_M_csv = pd.read_csv( self.csv_path)
#         MCI_M_dict = MCI_M_csv.to_dict()
#         self.config_path = self.path
#         self.config = OmegaConf.load(self.config_path)
# #         self.config.dataloader.h5cachedir_ = '/storage/Ayantika/h5data_store_with_DiffAE_output'
            
#         datalist = self.get_list_for_dataloader(self.config.dataloader.key_tuple_image,MCI_M_dict,key_='image')
#         masklist = self.get_list_for_dataloader(self.config.dataloader.key_tuple_mask,MCI_M_dict,key_='label')
#         key_list_image = []
#         key_list_label = []
#         for keys_ in datalist.keys():
#             key_list_image.append(str(keys_)+'_image')
#             key_list_image.append(str(keys_)+'_baseline_image')
#         for keys_ in masklist.keys():
#             key_list_label.append(str(keys_)+'_label')
#             key_list_label.append(str(keys_)+'_baseline_label')
#         print(key_list_image,key_list_label)

#         transforms_ = self.get_transform_nii_vols(key_list_image,key_list_label)
#         self.config.dataloader.h5cachedir_ = self.h5_save_path
#         ## The loader is such that it would create h5 files if they are not created when the loader is called and executed
#         if not os.path.exists(self.config.dataloader.h5cachedir_):
#             os.mkdir(self.config.dataloader.h5cachedir_)
#         h5cacheds = sdl.H5CachedDataset(datalist,masklist,transforms_,h5cachedir= self.config.dataloader.h5cachedir_,\
#                                            nslices_per_image=self.config.dataloader.total_slices_per_vol,\
#                                            start_slice = self.config.dataloader.start_slice_per_vol,\
#                                         end_slice = self.config.dataloader.end_slice_per_vol)
#         return h5cacheds


