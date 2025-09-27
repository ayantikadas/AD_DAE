import os
import shutil
import torch
import torchvision
from pytorch_fid import fid_score
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.autonotebook import tqdm, trange
from torch.utils.data import Dataset
from renderer_cond import *
from config_ADNI import *
from diffusion import Sampler
from dist_utils import *
import lpips
from ssim import ssim


class SubsetDataset(Dataset):
    def __init__(self, dataset, size):
        assert len(dataset) >= size
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        assert index < self.size
        return self.dataset[index]


def make_subset_loader(conf: TrainConfig,
                       dataset: Dataset,
                       batch_size: int,
                       shuffle: bool,
                       parallel: bool,
                       drop_last=True):
    dataset = SubsetDataset(dataset, size=conf.eval_num_images)
    if parallel and distributed.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = None
    return DataLoader(
        dataset,
        batch_size=batch_size,
#         sampler=sampler,
        # with sampler, use the sample instead of this option if sampler else shuffle
        shuffle=True,
        num_workers=conf.num_workers,
#         pin_memory=True,
        drop_last=drop_last,
        prefetch_factor=5,
#         multiprocessing_context=get_context('fork'),
    )

def age_vector(age):
    age_ranges = [(55, 65), (65, 75), (75, 85), (85, float('inf'))]
    label_vectors = torch.eye(4)
    label = torch.zeros(4)
    for idx, (lower, upper) in enumerate(age_ranges):
        if lower <= age < upper:
            label = label_vectors[idx]
            break

    return label

def slice_vector(slicenum):
    slice_ranges = [(60, 70), (70, 80), (80, 90), (90, 100), (100, 110)]
    label_vectors = torch.eye(5)
    label = torch.zeros(5)
    for idx, (lower, upper) in enumerate(slice_ranges):
        if lower <= slicenum < upper:
            label = label_vectors[idx]
            break

    return label

# def get_data_elements(batch,age_diff,target_indices):
#     health_encoding ={'AD': [1, 0, 0], 'CN': [0, 1, 0], 'MCI': [0, 0, 1]}
# #         batch_dict = {'DDIM_reverse':[], 'DiffAE_pred':[], 'cond':[]}
#     for ii in range(0,batch['_train_image'].shape[0]):

#         ###
#         if age_diff[ii]!=0:
#             target_indices[ii][3] = torch.tensor(1)

# #             target_indices[ii][0:4] = age_vector(age=batch['Age'][ii])
#         target_indices[ii][0:3] = torch.tensor(health_encoding[batch['Health status'][ii]])

# #             target_indices[ii][4:8] = age_vector(age=batch['Age'][ii])
#         target_indices[ii][4:10] = slice_vector(slicenum=batch['slicenum'][ii])


# #         print(dict_['cond'].shape)

# #         target_indices.cuda()
#     target_indices = target_indices.to(batch['_train_image'].dtype)
#     shifts = age_diff
# #         shifts.cuda()
#     basis_shift = target_indices.clone()
# #         basis_shift[:,5] = shifts
#     basis_shift[:,3] = shifts

#     return target_indices,shifts,basis_shift

def age_gap_vectors(age_gap):
    age_gap_ranges = [(0, 0.5), (0.5, 1),\
                    (1, 1.5), (1.5, 2), \
                    (2, 2.5), (2.5, 3),\
                   (3, 3.5), (3.5, 4),(4, 4.5)]
    label_vectors = torch.eye(9)
    label = torch.zeros(9)
    for idx, (lower, upper) in enumerate(age_gap_ranges):
        if lower < age_gap <= upper:
            label = label_vectors[idx]
            break

    return label


def str_list_tensor(age_):
    if type(age_ == list):
        if type(age_[0]) == str:
            return torch.tensor([torch.tensor(float(age)) for age in age_]) 
        else:
            return print('list element not str')
    else:
        return print('list of str')
def get_data_elements(batch,age_diff,target_indices):
    health_encoding ={'AD': [1, 0, 0], 'CN': [0, 1, 0], 'MCI': [0, 0, 1]}
    
    if '_train_image' in batch.keys():
        mode_ = '_train'
    elif '_test_image' in batch.keys():
        mode_ = '_test'

    for ii in range(0,batch[mode_+'_image'].shape[0]):
        
        if age_diff[ii]!=0:
            target_indices[ii][3] = torch.tensor(1)


        target_indices[ii][0:3] = torch.tensor(health_encoding[batch['Health status'][ii]])
        target_indices[ii][3:3+9] = age_gap_vectors((str_list_tensor(batch['Age']) - str_list_tensor(batch['baseline Age']))[ii])

    target_indices = target_indices.to(batch[mode_+'_image'].dtype)
    target_indices = target_indices.to(batch[mode_+'_image'].device)
    shifts = age_diff
#         shifts.cuda()
    basis_shift = target_indices.clone()
#         basis_shift[:,5] = shifts
#         basis_shift[:,3] = shifts

    return target_indices,shifts,basis_shift



def evaluate_lpips(
    sampler: Sampler,
    model: Model,
    conf: TrainConfig,
    device,
    val_data: Dataset,
    latent_sampler: Sampler = None,
    use_inverted_noise: bool = False,
    epoch=0,
):
    """
    compare the generated images from autoencoder on validation dataset

    Args:
        use_inversed_noise: the noise is also inverted from DDIM
    """
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    val_loader = make_subset_loader(conf,
                                    dataset=val_data,
                                    batch_size=conf.batch_size_eval,
                                    shuffle=False,
                                    parallel=True)

    model.eval()
    with torch.no_grad():
        scores = {
            'lpips': [],
            'mse': [],
            'ssim': [],
            'psnr': [],
        }
        # dict_keys(['_train_image', '_train_baseline_image', '_train_label', \
        # '_train_baseline_label', 'image_meta_dict', 'slicenum', 'idx', 'Subject', \
        # 'Aquisition Date', 'baseline Aquisition Date', 'Age', 'baseline Age', 'Description',\
        # 'nii path', 'baseline nii path', 'Sex', 'Health status'])
        
        for batch in tqdm(val_loader, desc='lpips'):
            if '_train_image' in batch.keys():
                mode_ = '_train'
            elif '_test_image' in batch.keys():
                mode_ = '_test'
                
            imgs = batch[mode_+'_image'].to(device)
            imgs_baseline = batch[mode_+'_baseline_image'].to(device)
            
            age_diff = (str_list_tensor(batch['Age']) - str_list_tensor(batch['baseline Age'])).to(device)
            age_diff = age_diff.to(imgs.device)
            ###### for health state vector
            labels = batch[ 'Health status']

            label_encoding = {'AD': [1, 0, 0], 'CN': [0, 1, 0], 'MCI': [0, 0, 1]}

            # Create tensors with label encodings
            target_indices = torch.zeros(batch[mode_+'_image'].shape[0],12)

            target_indices,shifts,basis_shift = get_data_elements(batch,age_diff,target_indices)

            # Convert the list of tensors into a single tensor
            #             health_state = torch.stack(encoded_tensors)
            health_state = basis_shift
            health_state = health_state.to(age_diff.dtype)
            health_state = health_state.to(age_diff.device)
            #########################
            cond = model.encode(imgs_baseline)
            if use_inverted_noise:
                # inverse the noise
                # with condition from the encoder
                model_kwargs = {}
                if conf.model_type.has_autoenc():
                    with torch.no_grad():
                        model_kwargs = model.encode(imgs)
                x_T = sampler.ddim_reverse_sample_loop(
                    model=model,
                    x=imgs,
                    clip_denoised=True,
                    x_start_baseline=imgs_baseline,
                    age_diff=age_diff, 
                    model_kwargs=model_kwargs)
                x_T = x_T['sample']
            else:
                x_T = torch.randn((len(imgs), 1, conf.img_size_height, conf.img_size_width),
                                  device=device)

#             print("age_diff in metrics inside evaluate score",age_diff)
            if (epoch<=10) or ((epoch%10)==0):
                sampler.cond_shift_weight= 0
                imgs = imgs_baseline
            else:
                sampler.cond_shift_weight= 1
            pred_imgs = render_condition(conf=conf,
                                             model=model,
                                             x_T=x_T,
                                             x_start=imgs,
                                             x_start_baseline=imgs_baseline,
                                             age_diff=age_diff, 
                                             health_state=health_state,
                                             cond=None,
                                             sampler=sampler)

            # (n, 1, 1, 1) => (n, )
            scores['lpips'].append(lpips_fn.forward(imgs, pred_imgs).view(-1))

            # need to normalize into [0, 1]
            
            norm_imgs = (imgs + 1) / 2
            norm_pred_imgs = (pred_imgs + 1) / 2
            
            # (n, )
            scores['ssim'].append(
                ssim(norm_imgs, norm_pred_imgs, size_average=False))
            # (n, )
            scores['mse'].append(
                (norm_imgs - norm_pred_imgs).pow(2).mean(dim=[1, 2, 3]))
            # (n, )
            scores['psnr'].append(psnr(norm_imgs, norm_pred_imgs))
        # (N, )
        for key in scores.keys():
            scores[key] = torch.cat(scores[key]).float()
    model.train()

    barrier()

    # support multi-gpu
    outs = {
        key: [
            torch.zeros(len(scores[key]), device=device)
            for i in range(get_world_size())
        ]
        for key in scores.keys()
    }
    for key in scores.keys():
        all_gather(outs[key], scores[key])

    # final scores
    for key in scores.keys():
        scores[key] = torch.cat(outs[key]).mean().item()

    # {'lpips', 'mse', 'ssim'}
    return scores


def psnr(img1, img2):
    """
    Args:
        img1: (n, c, h, w)
    """
    v_max = 1.
    # (n,)
    mse = torch.mean((img1 - img2)**2, dim=[1, 2, 3])
    return 20 * torch.log10(v_max / torch.sqrt(mse))


# +
def evaluate_fid(
    sampler: Sampler,
    model: Model,
    conf: TrainConfig,
    device,
    train_data: Dataset,
    val_data: Dataset,
    latent_sampler: Sampler = None,
    conds_mean=None,
    conds_std=None,
    T: int = 25,
    remove_cache: bool = True,
    clip_latent_noise: bool = False,
    epoch=0,
):
    assert conf.fid_cache is not None
    if get_rank() == 0:
        # no parallel
        # validation data for a comparing FID
        val_loader = make_subset_loader(conf,
                                        dataset=val_data,
                                        batch_size=conf.batch_size_eval,
                                        shuffle=False,
                                        parallel=False)

        # put the val images to a directory
        cache_dir = f'{conf.fid_cache}_{conf.eval_num_images}'
        if (os.path.exists(cache_dir)
                and len(os.listdir(cache_dir)) < conf.eval_num_images):
            shutil.rmtree(cache_dir)

        if not os.path.exists(cache_dir):
            # write files to the cache
            # the images are normalized, hence need to denormalize first
            loader_to_path(val_loader, cache_dir, denormalize=True)

        # create the generate dir
        if os.path.exists(conf.generate_dir):
            shutil.rmtree(conf.generate_dir)
        os.makedirs(conf.generate_dir)

    barrier()

    world_size = get_world_size()
    rank = get_rank()
    batch_size = chunk_size(conf.batch_size_eval, rank, world_size)

    def filename(idx):
        return world_size * idx + rank

    model.eval()
    with torch.no_grad():

        if conf.model_type == ModelType.autoencoder:

            train_loader = make_subset_loader(conf,
                                              dataset=train_data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              parallel=True)

            i = 0
            
            # dict_keys(['_train_image', '_train_baseline_image', '_train_label', \
        # '_train_baseline_label', 'image_meta_dict', 'slicenum', 'idx', 'Subject', \
        # 'Aquisition Date', 'baseline Aquisition Date', 'Age', 'baseline Age', 'Description',\
        # 'nii path', 'baseline nii path', 'Sex', 'Health status'])
        
#         for batch in tqdm(val_loader, desc='lpips'):
#             imgs = batch['_train_image'].to(device)
#             imgs_baseline = batch['_train_baseline_image'].to(device)
#             age_diff = batch['Age'] - batch['baseline Age']
#             cond = model.encode(imgs_baseline)

            for batch in tqdm(train_loader, desc='generating images'):
                if '_train_image' in batch.keys():
                    mode_ = '_train'
                elif '_test_image' in batch.keys():
                    mode_ = '_test'
                imgs = batch[mode_+'_image'].to(device)
                imgs_baseline = batch[mode_+'_baseline_image'].to(device)
                age_diff = (str_list_tensor(batch['Age']) - str_list_tensor(batch['baseline Age'])).to(device)
                age_diff = age_diff.to(imgs.device)
                ###### for health state vector
                labels = batch[ 'Health status']

                label_encoding = {'AD': [1, 0, 0], 'CN': [0, 1, 0], 'MCI': [0, 0, 1]}

                # Create tensors with label encodings
                target_indices = torch.zeros(batch[mode_+'_image'].shape[0],12)
                
                target_indices,shifts,basis_shift = get_data_elements(batch,age_diff,target_indices)

                # Convert the list of tensors into a single tensor
                #             health_state = torch.stack(encoded_tensors)
                health_state = basis_shift
                health_state = health_state.to(age_diff.dtype)
                health_state = health_state.to(age_diff.device)

                #########################
                cond = model.encode(imgs_baseline)
                x_T = torch.randn((len(imgs), 1, conf.img_size_height, conf.img_size_width),
                                  device=device)
#                 print("inside evaluate fid imgs_baseline",imgs_baseline.shape)
#                 print("inside evaluate fid x_T",x_T.shape)
                if (epoch<=10) or ((epoch%10)==0):
                    sampler.cond_shift_weight= 0
                    imgs = imgs_baseline
                else:
                    sampler.cond_shift_weight= 1
                batch_images = render_condition(
                    conf=conf,
                    model=model,
                    x_T=x_T,
                    x_start=imgs,
                    x_start_baseline=imgs_baseline,
                    age_diff=age_diff, 
                    health_state=health_state,
                    cond=None,
                    sampler=sampler).cpu()

                # denormalize the images
                batch_images = (batch_images + 1) / 2
                # keep the generated images
                for j in range(len(batch_images)):
                    img_name = filename(i + j)
                    torchvision.utils.save_image(
                        batch_images[j],
                        os.path.join(conf.generate_dir, f'{img_name}.png'))
                i += len(imgs)
        else:
            raise NotImplementedError()
    model.train()

    barrier()

    if get_rank() == 0:
        fid = fid_score.calculate_fid_given_paths(
            [cache_dir, conf.generate_dir],
            batch_size,
            device=device,
            dims=2048)

        # remove the cache
        if remove_cache and os.path.exists(conf.generate_dir):
            shutil.rmtree(conf.generate_dir)

    barrier()

    if get_rank() == 0:
        # need to float it! unless the broadcasted value is wrong
        fid = torch.tensor(float(fid), device=device)
        broadcast(fid, 0)
    else:
        fid = torch.tensor(0., device=device)
        broadcast(fid, 0)
    fid = fid.item()
    print(f'fid ({get_rank()}):', fid)

    return fid


# -

def loader_to_path(loader: DataLoader, path: str, denormalize: bool):
    # not process safe!

    if not os.path.exists(path):
        os.makedirs(path)

    # write the loader to files
    i = 0
    for batch in tqdm(loader, desc='copy images'):
        if '_train_image' in batch.keys():
            mode_ = '_train'
        elif '_test_image' in batch.keys():
            mode_ = '_test'
        imgs = batch[mode_+'_image']
        if denormalize:
            imgs = (imgs + 1) / 2
        for j in range(len(imgs)):
            torchvision.utils.save_image(imgs[j],
                                         os.path.join(path, f'{i+j}.png'))
        i += len(imgs)
