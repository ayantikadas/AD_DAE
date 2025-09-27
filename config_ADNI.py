from model.unet import ScaleAt
from model.latentnet import *
from diffusion.resample import UniformSampler
from diffusion.diffusion import space_timesteps
from typing import Tuple


from torch.utils.data import DataLoader

from config_base import BaseConfig
from dataset import *
from diffusion import *
from diffusion.base import GenerativeType, LossType, ModelMeanType, ModelVarType, get_named_beta_schedule
from model import *
from choices import *
from multiprocessing import get_context
import os
from dataset_util import *
from torch.utils.data.distributed import DistributedSampler

############### BratsReg data loader ##############################
from monai.transforms import (
Orientationd, EnsureChannelFirst, Compose, ToTensord, Spacingd,Resized,ScaleIntensityD,ResizeWithPadOrCropd
# ScaleIntensityD, ScaleIntensityRangeD, AdjustContrastD, RandAffined, ToNumpyd,RepeatChannelD
)
# EnsureChannelFirst
import sys
import glob
import torch
import os

# +
# import sys
# sys.path.insert(0,'/storage/Ayantika/Diff_AE_xstart_w_xbsln_disentangle_unsup/dataset/')
import dataset.slice_data_h5_ADNI_ventricle_mask as sdl
from omegaconf import OmegaConf
import pandas as pd

import dataset.loader_ADNI_data_ventricle_mask as load_
# -

import monai.transforms as transforms
# Define the custom transform
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


# +
# ############### 
# data_paths = {
#     'ffhqlmdb256':
#     os.path.expanduser('datasets/ffhq256.lmdb'),
#     # used for training a classifier
#     'celeba':
#     os.path.expanduser('/home/ayantika/Diffusion/diffae/datasets/celeba/'),
#     # used for training DPM models
#     'celebalmdb':
#     os.path.expanduser('datasets/celeba.lmdb'),
#     'celebahq':
#     os.path.expanduser('/home/ayantika/Diffusion/diffae/celebahq256.lmdb'),
#     'horse256':
#     os.path.expanduser('datasets/horse256.lmdb'),
#     'bedroom256':
#     os.path.expanduser('datasets/bedroom256.lmdb'),
#     'celeba_anno':
#     os.path.expanduser('datasets/celeba_anno/list_attr_celeba.txt'),
#     'celebahq_anno':
#     os.path.expanduser('/home/ayantika/Diffusion/diffae/datasets/celeba_anno/CelebAMask-HQ-attribute-anno.txt'
#         ),
#     'celeba_relight':
#     os.path.expanduser('datasets/celeba_hq_light/celeba_light.txt'),
# }
# -

@dataclass
class PretrainConfig(BaseConfig):
    name: str
    path: str



# +
@dataclass
class TrainConfig(BaseConfig):
    # random seed
    seed: int = 0
    train_mode: TrainMode = TrainMode.diffusion
    train_cond0_prob: float = 0
    train_pred_xstart_detach: bool = True
    train_interpolate_prob: float = 0
    train_interpolate_img: bool = False
    manipulate_mode: ManipulateMode = ManipulateMode.celebahq_all
    manipulate_cls: str = None
    manipulate_shots: int = None
    manipulate_loss: ManipulateLossType = ManipulateLossType.bce
    manipulate_znormalize: bool = False
    manipulate_seed: int = 0
    accum_batches: int = 1
    autoenc_mid_attn: bool = True
    batch_size: int = 16
    batch_size_eval: int = None
    beatgans_gen_type: GenerativeType = GenerativeType.ddim
    beatgans_loss_type: LossType = LossType.latent_code_mse
    beatgans_model_mean_type: ModelMeanType = ModelMeanType.eps
    beatgans_model_var_type: ModelVarType = ModelVarType.fixed_large
    beatgans_rescale_timesteps: bool = False
    latent_infer_path: str = None
    latent_znormalize: bool = False
    latent_gen_type: GenerativeType = GenerativeType.ddim
    latent_loss_type: LossType = LossType.mse
    latent_model_mean_type: ModelMeanType = ModelMeanType.eps
    latent_model_var_type: ModelVarType = ModelVarType.fixed_large
    latent_rescale_timesteps: bool = False
    latent_T_eval: int = 1_000
    latent_clip_sample: bool = False
    latent_beta_scheduler: str = 'linear'
    beta_scheduler: str = 'linear'
    data_name: str = ''
    data_val_name: str = None
    diffusion_type: str = None
    dropout: float = 0.1
    ema_decay: float = 0.9999
    eval_num_images: int = 5_000
    eval_every_samples: int = 200_000
    eval_ema_every_samples: int = 200_000
    fid_use_torch: bool = True
    fp16: bool = False
    grad_clip: float = 1
    img_size: int = 64
    img_size_height: int = 64
    img_size_width: int = 64
    lr: float = 0.0001
    optimizer: OptimizerType = OptimizerType.adam
    weight_decay: float = 0
    model_conf: ModelConfig = None
    model_name: ModelName = None
    model_type: ModelType = None
    net_attn: Tuple[int] = None
    net_beatgans_attn_head: int = 1
    # not necessarily the same as the the number of style channels
    net_beatgans_embed_channels: int = 512
    net_resblock_updown: bool = True
    net_enc_use_time: bool = False
    net_enc_pool: str = 'adaptivenonzero'
    net_beatgans_gradient_checkpoint: bool = False
    net_beatgans_resnet_two_cond: bool = False
    net_beatgans_resnet_use_zero_module: bool = True
    net_beatgans_resnet_scale_at: ScaleAt = ScaleAt.after_norm
    net_beatgans_resnet_cond_channels: int = None
    net_ch_mult: Tuple[int] = None
    net_ch: int = 64
    net_enc_attn: Tuple[int] = None
    net_enc_k: int = None
    # number of resblocks for the encoder (half-unet)
    net_enc_num_res_blocks: int = 2
    net_enc_channel_mult: Tuple[int] = None
    net_enc_grad_checkpoint: bool = False
    net_autoenc_stochastic: bool = False
    net_latent_activation: Activation = Activation.silu
    net_latent_channel_mult: Tuple[int] = (1, 2, 4)
    net_latent_condition_bias: float = 0
    net_latent_dropout: float = 0
    net_latent_layers: int = None
    net_latent_net_last_act: Activation = Activation.none
    net_latent_net_type: LatentNetType = LatentNetType.none
    net_latent_num_hid_channels: int = 1024
    net_latent_num_time_layers: int = 2
    net_latent_skip_layers: Tuple[int] = None
    net_latent_time_emb_channels: int = 64
    net_latent_use_norm: bool = False
    net_latent_time_last_act: bool = False
    net_num_res_blocks: int = 2
    # number of resblocks for the UNET
    net_num_input_res_blocks: int = None
    net_enc_num_cls: int = None
    num_workers: int = 4
    parallel: bool = False
    postfix: str = ''
    sample_size: int = 64
    sample_every_samples: int = 20_000
    save_every_samples: int = 100_000
    style_ch: int = 512
    T_eval: int = 1_000
    T_sampler: str = 'uniform'
    T: int = 1_000
    total_samples: int = 10_000_000
    warmup: int = 0
    pretrain: PretrainConfig = None
    continue_from: PretrainConfig = None
    eval_programs: Tuple[str] = None
    # if present load the checkpoint from this path instead
    eval_path: str = None
    base_dir: str = 'checkpoints'
    use_cache_dataset: bool = False
    data_cache_dir: str = os.path.expanduser('~/cache')
    work_cache_dir: str = os.path.expanduser('~/mycache')
    # to be overridden
    name: str = ''
    csv_path: str = ''
    csv_path_test: str = ''
    h5_save_path_train: str = ''
    h5_save_path_test: str = ''
    data_config_path: str = ''
        
    #####
    csv_file_name_train: str = ''
#     = '/home/projects/medimg/ayantika/Ayantika/Data_final/ADNI_Data_loader_csv/ADNI_train_pair_data_info.csv'
    csv_mask_name_train: str = '' 
#     = '/home/projects/medimg/ayantika/Ayantika/Data_final/ADNI_Data_loader_csv/ADNI_train_pair_mask_info.csv'

    csv_file_name_test: str = '' 
#     = '/home/projects/medimg/ayantika/Ayantika/Data_final/ADNI_Data_loader_csv/ADNI_test_pair_data_info.csv'
    csv_mask_name_test: str = '' 
#     = '/home/projects/medimg/ayantika/Ayantika/Data_final/ADNI_Data_loader_csv/ADNI_test_pair_mask_info.csv'
    
    ventricle_mask_root_path: str = ''
#     = '/home/projects/medimg/ayantika/Ayantika/Data_final/ADNI_ventricle_mask/',\
    mode_train: str = ''
    mode_test: str = ''
    

    
    def __post_init__(self):
        self.batch_size_eval = self.batch_size_eval or self.batch_size
        self.data_val_name = self.data_val_name or self.data_name

        

    def scale_up_gpus(self, num_gpus, num_nodes=1):
        self.eval_ema_every_samples *= num_gpus * num_nodes
        self.eval_every_samples *= num_gpus * num_nodes
        self.sample_every_samples *= num_gpus * num_nodes
        self.batch_size *= num_gpus * num_nodes
        self.batch_size_eval *= num_gpus * num_nodes
        return self

    @property
    def batch_size_effective(self):
        return self.batch_size * self.accum_batches

    @property
    def fid_cache(self):
        # we try to use the local dirs to reduce the load over network drives
        # hopefully, this would reduce the disconnection problems with sshfs
        return f'{self.work_cache_dir}/eval_images/{self.data_name}_size{self.img_size}_{self.eval_num_images}'

    @property
    def data_path(self):
        # may use the cache dir
        path = data_paths[self.data_name]
        if self.use_cache_dataset and path is not None:
            path = use_cached_dataset_path(
                path, f'{self.data_cache_dir}/{self.data_name}')
        return path

    @property
    def logdir(self):
        return f'{self.base_dir}/{self.name}'

    @property
    def generate_dir(self):
        # we try to use the local dirs to reduce the load over network drives
        # hopefully, this would reduce the disconnection problems with sshfs
        return f'{self.work_cache_dir}/gen_images_cond/{self.name}'

    def _make_diffusion_conf(self, T=None):
        if self.diffusion_type == 'beatgans':
            # can use T < self.T for evaluation
            # follows the guided-diffusion repo conventions
            # t's are evenly spaced
            if self.beatgans_gen_type == GenerativeType.ddpm:
                section_counts = [T]
            elif self.beatgans_gen_type == GenerativeType.ddim:
                section_counts = f'ddim{T}'
            else:
                raise NotImplementedError()

            return SpacedDiffusionBeatGansConfig(
                gen_type=self.beatgans_gen_type,
                model_type=self.model_type,
                betas=get_named_beta_schedule(self.beta_scheduler, self.T),
                model_mean_type=self.beatgans_model_mean_type,
                model_var_type=self.beatgans_model_var_type,
                loss_type=self.beatgans_loss_type,
                rescale_timesteps=self.beatgans_rescale_timesteps,
                use_timesteps=space_timesteps(num_timesteps=self.T,
                                              section_counts=section_counts),
                fp16=self.fp16,
            )
        else:
            raise NotImplementedError()

    def _make_latent_diffusion_conf(self, T=None):
        # can use T < self.T for evaluation
        # follows the guided-diffusion repo conventions
        # t's are evenly spaced
        if self.latent_gen_type == GenerativeType.ddpm:
            section_counts = [T]
        elif self.latent_gen_type == GenerativeType.ddim:
            section_counts = f'ddim{T}'
        else:
            raise NotImplementedError()

        return SpacedDiffusionBeatGansConfig(
            train_pred_xstart_detach=self.train_pred_xstart_detach,
            gen_type=self.latent_gen_type,
            # latent's model is always ddpm
            model_type=ModelType.ddpm,
            # latent shares the beta scheduler and full T
            betas=get_named_beta_schedule(self.latent_beta_scheduler, self.T),
            model_mean_type=self.latent_model_mean_type,
            model_var_type=self.latent_model_var_type,
            loss_type=self.latent_loss_type,
            rescale_timesteps=self.latent_rescale_timesteps,
            use_timesteps=space_timesteps(num_timesteps=self.T,
                                          section_counts=section_counts),
            fp16=self.fp16,
        )

    @property
    def model_out_channels(self):
        return 1

    def make_T_sampler(self):
        if self.T_sampler == 'uniform':
            return UniformSampler(self.T)
        else:
            raise NotImplementedError()

    def make_diffusion_conf(self):
        return self._make_diffusion_conf(self.T)

    def make_eval_diffusion_conf(self):
        return self._make_diffusion_conf(T=self.T_eval)

    def make_latent_diffusion_conf(self):
        return self._make_latent_diffusion_conf(T=self.T)

    def make_latent_eval_diffusion_conf(self):
        # latent can have different eval T
        return self._make_latent_diffusion_conf(T=self.latent_T_eval)
   
    
# len(datalist['_train'])
    #################################### 

    def make_dataset(self, path = ' ',csv_path = ' ', h5_save_path = '', csv_file_name = '', csv_mask_name = '' , ventricle_mask_root_path = '', mode_ = '' , **kwargs):
        print("self.data_name",self.data_name)
        if self.data_name == 'ffhqlmdb256':
            return FFHQlmdb(path=path or self.data_path,
                            image_size=self.img_size,
                            **kwargs)
        elif self.data_name == 'horse256':
            return Horse_lmdb(path=path or self.data_path,
                              image_size=self.img_size,
                              **kwargs)
        elif self.data_name == 'bedroom256':
            return Horse_lmdb(path=path or self.data_path,
                              image_size=self.img_size,
                              **kwargs)
        elif self.data_name == 'celebalmdb':
            # always use d2c crop
            return CelebAlmdb(path=path or self.data_path,
                              image_size=self.img_size,
                              original_resolution=None,
                              crop_d2c=True,
                              **kwargs)
        #################################################
        elif self.data_name == 'celeba':
            # always use d2c crop
            return CelebAttrDataset(folder=path or self.data_path,
                                  ext = 'jpg',
                                  image_size=self.img_size,
    #                               original_resolution=None,
    #                               crop_d2c=True,
                                  **kwargs)
        
        elif self.data_name ==  'brats_reg':
#             print("config.dataloader.imagesize",self.config.dataloader.img_height)
            self.config_path = self.data_config_path
            self.config = OmegaConf.load(self.config_path)
            datalist = {}
            for i,key in enumerate(self.config.dataloader.key_tuple_image):
                list_image_paths = glob.glob(self.config.dataloader.training_path[i])
                list_image_paths.sort()
                datalist.update({key:[{'image':x} for x in list_image_paths]})


            masklist = {}
            for i,key in enumerate(self.config.dataloader.key_tuple_mask):
                list_mask_paths = glob.glob(self.config.dataloader.mask_path[i])
                list_mask_paths.sort()
                masklist.update({key:[{'label':x} for x in list_mask_paths]})
            key_list_image = []
            key_list_label = []
            for keys_ in datalist.keys():
                key_list_image.append(str(keys_)+'_image')
            for keys_ in masklist.keys():
                key_list_label.append(str(keys_)+'_label')


            transforms_ = self.get_transform_nii_vols(key_list_image,key_list_label)

            ### The loader is such that it would create h5 files if they are not created when the loader is called and executed
            if not os.path.exists(self.config.dataloader.h5cachedir_):
                os.mkdir(self.config.dataloader.h5cachedir_)
            h5cacheds = sdl_h5.H5CachedDataset(datalist,masklist,transforms_,h5cachedir=self.config.dataloader.h5cachedir_,\
                                               nslices_per_image=self.config.dataloader.total_slices_per_vol,\
                                               start_slice = self.config.dataloader.start_slice_per_vol,\
                                            end_slice = self.config.dataloader.end_slice_per_vol)
            
        
            
            return h5cacheds
        
        elif self.data_name ==  'ADNI':
            
#             '/data/Ayantika/Data_final/AD_data/AD_info_train_final.csv'


#         make_dataset(self, path = ' ',csv_path = ' ', h5_save_path = '', csv_file_name = '', csv_mask_name = '' , ventricle_mask_root_path = '', mode_ = '' , **kwargs)
            
#             class_loader = load_.ADNI_dataloader(csv_path, path, h5_save_path)
#             h5cacheds = class_loader.main_call()
            
            
            
            loader_cls = load_.ADNI_dataloader(csv_path, path, h5_save_path)

            # loader_cls.create_pair_info_csv(csv_file_name ='/storage/ayantika/Ayantika/Data_final/ADNI_Data_loader_csv/ADNI_train_pair_data_info.csv',\
            #                                 csv_mask_name = '/storage/ayantika/Ayantika/Data_final/ADNI_Data_loader_csv/ADNI_train_pair_mask_info.csv')


            h5cacheds = loader_cls.main_call(csv_file_name=csv_file_name,\
                                 csv_mask_name = csv_mask_name,\
                                ventricle_mask_root_path= ventricle_mask_root_path ,\
                                 mode = mode_)

            return h5cacheds
        ##############################################
        else:
            raise NotImplementedError()

    def make_loader(self,
                    dataset,
                    shuffle: bool,
                    num_worker: bool = None,
                    drop_last: bool = True,
                    batch_size: int = None,
                    parallel: bool = False):
        if parallel and distributed.is_initialized():
            # drop last to make sure that there is no added special indexes
            sampler = DistributedSampler(dataset,
                                         shuffle=shuffle,
                                         drop_last=True)
        else:
            sampler = None
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            sampler=sampler,
            # with sampler, use the sample instead of this option
            shuffle=False if sampler else shuffle,
            num_workers=num_worker or self.num_workers,
            pin_memory=True,
            drop_last=drop_last,
#             multiprocessing_context=get_context('fork'),
        )

    def make_model_conf(self):
        if self.model_name == ModelName.beatgans_ddpm:
            self.model_type = ModelType.ddpm
            self.model_conf = BeatGANsUNetConfig(
                attention_resolutions=self.net_attn,
                channel_mult=self.net_ch_mult,
                conv_resample=True,
                dims=2,
                dropout=self.dropout,
                embed_channels=self.net_beatgans_embed_channels,
                image_size=self.img_size,
                in_channels=1,#channel added
                model_channels=self.net_ch,
                num_classes=None,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.net_beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                use_checkpoint=self.net_beatgans_gradient_checkpoint,
                use_new_attention_order=False,
                resnet_two_cond=self.net_beatgans_resnet_two_cond,
                resnet_use_zero_module=self.
                net_beatgans_resnet_use_zero_module,
            )
        elif self.model_name in [
                ModelName.beatgans_autoenc,
        ]:
            cls = BeatGANsAutoencConfig
            # supports both autoenc and vaeddpm
            if self.model_name == ModelName.beatgans_autoenc:
                self.model_type = ModelType.autoencoder
            else:
                raise NotImplementedError()

            if self.net_latent_net_type == LatentNetType.none:
                latent_net_conf = None
            elif self.net_latent_net_type == LatentNetType.skip:
                latent_net_conf = MLPSkipNetConfig(
                    num_channels=self.style_ch,
                    skip_layers=self.net_latent_skip_layers,
                    num_hid_channels=self.net_latent_num_hid_channels,
                    num_layers=self.net_latent_layers,
                    num_time_emb_channels=self.net_latent_time_emb_channels,
                    activation=self.net_latent_activation,
                    use_norm=self.net_latent_use_norm,
                    condition_bias=self.net_latent_condition_bias,
                    dropout=self.net_latent_dropout,
                    last_act=self.net_latent_net_last_act,
                    num_time_layers=self.net_latent_num_time_layers,
                    time_last_act=self.net_latent_time_last_act,
                )
            else:
                raise NotImplementedError()

            self.model_conf = cls(
                attention_resolutions=self.net_attn,
                channel_mult=self.net_ch_mult,
                conv_resample=True,
                dims=2,
                dropout=self.dropout,
                embed_channels=self.net_beatgans_embed_channels,
                enc_out_channels=self.style_ch,
                enc_pool=self.net_enc_pool,
                enc_num_res_block=self.net_enc_num_res_blocks,
                enc_channel_mult=self.net_enc_channel_mult,
                enc_grad_checkpoint=self.net_enc_grad_checkpoint,
                enc_attn_resolutions=self.net_enc_attn,
                image_size=self.img_size,
                in_channels=1,### channel addded
                model_channels=self.net_ch,
                num_classes=None,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.net_beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                use_checkpoint=self.net_beatgans_gradient_checkpoint,
                use_new_attention_order=False,
                resnet_two_cond=self.net_beatgans_resnet_two_cond,
                resnet_use_zero_module=self.net_beatgans_resnet_use_zero_module,
                latent_net_conf=latent_net_conf,
                resnet_cond_channels=self.net_beatgans_resnet_cond_channels,
            )
        else:
            raise NotImplementedError(self.model_name)

        return self.model_conf
# +
################################### Comment out all these before saving ###################################
# -




# +
# # !rm -r '/storage/Ayantika/h5data_store/ADNI_cond_test_ventricle_mask'

# +
################################### Comment out all these before saving ###################################

# +
# ### 
# tr = TrainConfig()
# path = '/storage/Ayantika/Diffusion_AE_medical_cond/config_file_ADNI.yaml'
# tr.data_name = 'ADNI'
# tr.num_workers=1
# datset_ = tr.make_dataset(path = path,\
#                 csv_path = '/storage/Ayantika/Data_final/ADNI_info_test_final.csv', \
#                 h5_save_path = '/storage/Ayantika/h5data_store/ADNI_cond_test_ventricle_mask')


# ADNI_data = tr.make_loader(datset_,shuffle=False,batch_size=10)
# for i,batch in tqdm(enumerate(ADNI_data)):
#     print(batch.keys())

# +
# from tqdm.notebook import tqdm

# +
# # # !rm -r '/data/Ayantika/h5data_store/ADNI_cond'
# # conf.csv_path = 
# # conf.csv_path_test = '/storage/Ayantika/Data_final/ADNI_info_test_final.csv'
# # conf.h5_save_path_train = 
# # conf.h5_save_path_test = '/storage/Ayantika/h5data_store/ADNI_cond_test'
# tr = TrainConfig()
# path = '/storage/Ayantika/Diffusion_AE_medical_cond/config_file_ADNI.yaml'
# tr.data_name = 'ADNI'
# tr.num_workers=1
# datset_ = tr.make_dataset(path = path,\
#                 csv_path = '/storage/Ayantika/Data_final/ADNI_info_train_final_subset.csv', \
#                 h5_save_path = '/storage/Ayantika/h5data_store/ADNI_cond_train_ventricle_mask')


# ADNI_data = tr.make_loader(datset_,shuffle=True,batch_size=10)
# for i,batch in tqdm(enumerate(ADNI_data)):
#     print(batch.keys())
# #     if i==0:
# #         break


# +
# from matplotlib import pyplot as plt
# i=8
# plt.imshow(batch['_train_baseline_image'][i,0,:,:],cmap='gray')
# plt.colorbar()
# plt.figure()
# plt.imshow(batch['ventricle_mask'][i,0,:,:],cmap='gray')
# plt.colorbar()

# +
# tr = TrainConfig()
# path = '/storage/Ayantika/Diffusion_AE_medical_cond/config_file_ADNI.yaml'
# tr.data_name = 'ADNI'
# tr.num_workers=1
# datset_ = tr.make_dataset(path=path,csv_path = '/storage/Ayantika/Data_final/ADNI_info_train_final.csv')
# ADNI_data = tr.make_loader(datset_,shuffle=True,batch_size=10)
# for i,batch in enumerate(ADNI_data):
#     print(batch.keys())
#     if i==0:
#         break

# +
# 
# 
# print(tensor_batch)

# +
# batch['_train_baseline_image'].shape

# +
# from matplotlib import pyplot as plt
# plt.imshow(batch['_train_baseline_image'][0,0,:,:],cmap='gray')
# plt.colorbar()

# +
# plt.imshow(batch['_train_image'][0,0,:,:] ,cmap='gray')
# plt.colorbar()

# +
# config = OmegaConf.load(path)
# MCI_M_csv = pd.read_csv('/data/Ayantika/Data_final/ADNI_info_train_final.csv')
# # config
# config.dataloader.key_tuple_image

# +
# MCI_M_csv = pd.read_csv(csv_path)
# MCI_M_dict = MCI_M_csv.to_dict()
# # self.config_path = path
# config = OmegaConf.load(path)
# tuple_ = config.dataloader.key_tuple_image
# datalist = get_list_for_dataloader(tuple_,MCI_M_dict)
# tuple_ = config.dataloader.key_tuple_mask
# masklist = get_list_for_dataloader(tuple_,MCI_M_dict)

# +

# MCI_M_dict = MCI_M_csv.to_dict()
# self.config_path = path
# self.config = OmegaConf.load(self.config_path)
# datalist = {}
# for i,key in enumerate(self.config.dataloader.key_tuple_image):
# #     list_image_paths = glob.glob(conf.config.dataloader.training_path[i])
# #     list_image_paths.sort()
#     datalist.update({key:[ {'image':MCI_M_dict['nii path'][indx],'Subject': MCI_M_dict['Subject'][indx],\
#                             'Aquisition Date':MCI_M_dict['Aquisition Date'][indx],\
#                             'Health status':MCI_M_dict['Health status'][indx],\
#                             'Age':MCI_M_dict['Age'][indx],'Sex':MCI_M_dict['Sex'][indx],\
#                             'Description':MCI_M_dict['Description'][indx],'nii path':MCI_M_dict['nii path'][indx]}\
#                            for indx in range(0,len(MCI_M_dict['Subject'])) 
#                          ]})


# masklist = {}
# for i,key in enumerate(self.config.dataloader.key_tuple_mask):
# #     list_mask_paths = glob.glob(conf.config.dataloader.mask_path[i])
# #     list_mask_paths.sort()
#     masklist.update({key:[ {'label':MCI_M_dict['nii path'][indx],'Subject': MCI_M_dict['Subject'][indx],\
#                             'Aquisition Date':MCI_M_dict['Aquisition Date'][indx],\
#                             'Health status':MCI_M_dict['Health status'][indx],\
#                             'Age':MCI_M_dict['Age'][indx],'Sex':MCI_M_dict['Sex'][indx],\
#                             'Description':MCI_M_dict['Description'][indx],'nii path':MCI_M_dict['nii path'][indx]}\
#                            for indx in range(0,len(MCI_M_dict['Subject'])) 
#                          ]})
# key_list_image = []
# key_list_label = []
# for keys_ in datalist.keys():
#     key_list_image.append(str(keys_)+'_image')
# for keys_ in masklist.keys():
#     key_list_label.append(str(keys_)+'_label')


# transforms_ = self.get_transform_nii_vols(key_list_image,key_list_label)

# ## The loader is such that it would create h5 files if they are not created when the loader is called and executed
# if not os.path.exists(self.config.dataloader.h5cachedir_):
#     os.mkdir(self.config.dataloader.h5cachedir_)
# h5cacheds = sdl.H5CachedDataset(datalist,masklist,transforms_,h5cachedir= self.config.dataloader.h5cachedir_,\
#                                    nslices_per_image=self.config.dataloader.total_slices_per_vol,\
#                                    start_slice = self.config.dataloader.start_slice_per_vol,\
#                                 end_slice = self.config.dataloader.end_slice_per_vol)
# -





# for indx in range(0,len(MCI_M_dict['Subject'])):
#     for sub_indx in range(0,len(MCI_M_dict['Subject'])):
#         if (MCI_M_dict['Subject'][sub_indx] == MCI_M_dict['Subject'][indx]) and (MCI_M_dict['Age'][sub_indx] > MCI_M_dict['Age'][indx]):
#             print(MCI_M_dict['Subject'][sub_indx] , MCI_M_dict['Subject'][indx])
#             print(MCI_M_dict['Age'][sub_indx] , MCI_M_dict['Age'][indx])
#     print(MCI_M_dict['Age'][indx])
#     break






# +
# # config.dataloader.h5cachedir_
# # # !rm -r '/data/Ayantika/h5data_store/ADNI/'
# ADNI_data = DataLoader(h5cacheds,shuffle=False,batch_size=50) 

# +
# len(ADNI_data)

# +
# # '/data/Ayantika/h5data_store/AD'
# # ADNI_data = tr.make_loader(h5cacheds,shuffle=False,batch_size=1)
# for i,batch in enumerate(ADNI_data):
# #     if i==10:
#     print(batch.keys(),batch['_train_image'].shape,batch['_train_image'].min(),batch['_train_image'].max())
# #         break

# # dict_keys(['_train_image', '_train_label', 'image_meta_dict', \
# #            'slicenum', 'idx', 'Subject', 'Aquisition Date', 'Age', 'Description', \
# #            'nii path', 'Sex', 'Health status']) torch.Size([1, 1, 128, 128]) tensor(-1.) tensor(1.)

# +
# batch['nii path']

# +
# import ants
# img = ants.image_read(batch['nii path'][0])
# batch['_train_image'][20,0,:,:].shape, img[11:-10,5:-4,:].shape

# +
# # !rm -r '/data/Ayantika/h5data_store/ADNI/'

# +
# batch['Health status']
# batch['_train_image'][20,0,:,:].shape

# +
# from matplotlib import pyplot as plt
# plt.imshow(batch['_train_image'][20,0,:,:],cmap='gray')

# +
# from matplotlib import pyplot as plt
# plt.imshow(batch['_train_image'][20,0,:,:],cmap='gray')
# -


