import argparse
import json
import os
import warnings
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
from torch.utils import data
from torchvision import transforms
import image_models
import torch.nn.functional as F

warnings.filterwarnings("ignore")


class EmbeddingWithCrop(torch.nn.Module):
    def __init__(self, num_patches, embedding_model, h, w):
        #For IConQA <H>=168, <W>=326
        super(EmbeddingWithCrop, self).__init__()
        if num_patches == 25:
            self.splits = [5]
        elif num_patches == 36:
            self.splits = [6]
        elif num_patches == 14:
            self.splits = [1, 2, 3]
        elif num_patches == 30:
            self.splits = [1, 2, 3, 4]
        elif num_patches == 79:
            self.splits = [1, 2, 3, 4, 7]
        else:
            raise NotImplementedError()
        self.embedding_model = embedding_model
        self.h = h
        self.w = w
        self.max_size = max(h, w)
        self.image_size = self.max_size + 6
        self.transform1 = transforms.Resize((224, 224), antialias=True)
        self.transform2 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.pad_top = 3
        self.pad_bottom = self.max_size + 3 - self.h
        self.pad_left = 3
        self.pad_right = self.max_size + 3 - self.w

    def forward(self, x):
        patches_torch = []
        # w, h = im.shape  # width, height
        # for n in self.splits:
        #     dw, dh = w // n, h // n
        #     for j in range(n):
        #         for i in range(n):
        #             bbox = dw * i, dh * j, dw * (i + 1), dh * (j + 1)
        #             patch = img.crop(bbox)
        #             patches.append(patch)
        #H, W = x.size(2), x.size(3)
        H,W = self.h, self.w
        max_size = max(H, W)
        image_size = max_size + 6


        # Perform padding.
        padded_x = F.pad(x, (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom), mode='constant', value=255)
        for n in self.splits:
            # Calculate padding sizes


            # Calculate number of patches along each dimension
            patch_size = image_size // n

            # Split padded image into patches
            curr_patches = padded_x.unfold(2, patch_size, patch_size)
            curr_patches_1 = curr_patches.unfold(3, patch_size, patch_size)
            curr_patches_2 = curr_patches_1.contiguous()
            curr_patches_3 = curr_patches_2.view(1,3,-1,patch_size,patch_size) # [1,3,NUM-PATCHES,PATCH-SIZE,PATCH-SIZE]
            transformed_patches = self.transform1(curr_patches_3[0].permute(1, 0, 2, 3))
            transformed_patches2 = self.transform2(transformed_patches/255.)
            patches_torch.append(transformed_patches2)
            #patches = patches.contiguous().view(-1, x.size(1), n, n)
        stacked = torch.cat(patches_torch)
        image_embedding = self.embedding_model(stacked)
        return image_embedding


class EmbeddingWithCropv2(torch.nn.Module):
    def __init__(self, num_patches, embedding_model, h=168, w=326):
        #For IConQA <H>=168, <W>=326
        super(EmbeddingWithCropv2, self).__init__()
        if num_patches == 25:
            self.splits = [5]
        elif num_patches == 36:
            self.splits = [6]
        elif num_patches == 14:
            self.splits = [1, 2, 3]
        elif num_patches == 30:
            self.splits = [1, 2, 3, 4]
        elif num_patches == 79:
            self.splits = [1, 2, 3, 4, 7]
        else:
            raise NotImplementedError()
        self.embedding_model = embedding_model
        self.h = h
        self.w = w
        self.max_size = max(h, w)
        self.image_size = self.max_size + 6
        self.transform2 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.pad_top = 3
        self.pad_bottom = self.max_size + 3 - self.h
        self.pad_left = 3
        self.pad_right = self.max_size + 3 - self.w

    def forward(self, x):
        patches_torch = []
        # w, h = im.shape  # width, height
        # for n in self.splits:
        #     dw, dh = w // n, h // n
        #     for j in range(n):
        #         for i in range(n):
        #             bbox = dw * i, dh * j, dw * (i + 1), dh * (j + 1)
        #             patch = img.crop(bbox)
        #             patches.append(patch)
        #H, W = x.size(2), x.size(3)
        H,W = self.h, self.w
        max_size = self.max_size
        image_size = self.image_size
        for n in self.splits:
            patch_size = image_size // n
            curr_patches = x.unfold(2, patch_size, patch_size)
            curr_patches_1 = curr_patches.unfold(3, patch_size, patch_size)
            curr_patches_2 = curr_patches_1.contiguous()
            curr_patches_3 = curr_patches_2.view(1,3,-1,patch_size,patch_size) # [1,3,NUM-PATCHES,PATCH-SIZE,PATCH-SIZE]
            #transformed_patches = self.transform1(curr_patches_3[0].permute(1, 0, 2, 3))
            transformed_patches = F.interpolate(curr_patches_3[0].permute(1, 0, 2, 3), 224 ,mode='bilinear')
            transformed_patches2 = self.transform2(transformed_patches/255.)
            patches_torch.append(transformed_patches2)
        stacked = torch.cat(patches_torch)
        image_embedding = self.embedding_model(stacked)
        return image_embedding


class EmbeddingWithCropv2AntiAlias(torch.nn.Module):
    def __init__(self, num_patches, embedding_model, h=168, w=326):
        #For IConQA <H>=168, <W>=326
        super(EmbeddingWithCropv2AntiAlias, self).__init__()
        if num_patches == 25:
            self.splits = [5]
        elif num_patches == 36:
            self.splits = [6]
        elif num_patches == 14:
            self.splits = [1, 2, 3]
        elif num_patches == 30:
            self.splits = [1, 2, 3, 4]
        elif num_patches == 79:
            self.splits = [1, 2, 3, 4, 7]
        else:
            raise NotImplementedError()
        self.embedding_model = embedding_model
        self.h = h
        self.w = w
        self.max_size = max(h, w)
        self.image_size = self.max_size + 6
        self.transform1 = transforms.Resize((224, 224), antialias=True)
        self.transform2 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.pad_top = 3
        self.pad_bottom = self.max_size + 3 - self.h
        self.pad_left = 3
        self.pad_right = self.max_size + 3 - self.w

    def forward(self, x):
        patches_torch = []
        # w, h = im.shape  # width, height
        # for n in self.splits:
        #     dw, dh = w // n, h // n
        #     for j in range(n):
        #         for i in range(n):
        #             bbox = dw * i, dh * j, dw * (i + 1), dh * (j + 1)
        #             patch = img.crop(bbox)
        #             patches.append(patch)
        #H, W = x.size(2), x.size(3)
        H,W = self.h, self.w
        max_size = self.max_size
        image_size = self.image_size
        for n in self.splits:
            patch_size = image_size // n
            curr_patches = x.unfold(2, patch_size, patch_size)
            curr_patches_1 = curr_patches.unfold(3, patch_size, patch_size)
            curr_patches_2 = curr_patches_1.contiguous()
            curr_patches_3 = curr_patches_2.view(1,3,-1,patch_size,patch_size) # [1,3,NUM-PATCHES,PATCH-SIZE,PATCH-SIZE]
            transformed_patches = self.transform1(curr_patches_3[0].permute(1, 0, 2, 3))
            # transformed_patches = F.interpolate(curr_patches_3[0].permute(1, 0, 2, 3), 224 ,mode='bilinear', antialias=True)
            transformed_patches2 = self.transform2(transformed_patches/255.)
            patches_torch.append(transformed_patches2)
        stacked = torch.cat(patches_torch)
        image_embedding = self.embedding_model(stacked)
        return image_embedding


class ICONQADataset(data.Dataset):
    def __init__(self, input_path, output_path, arch, transform, icon_pretrained, split, task, num_patches):
        pid_splits = json.load(open(os.path.join(input_path, 'pid_splits.json')))
        self.data = pid_splits['%s_%s' % (task, split)] # len: 51766
        self.problems = json.load(open(os.path.join(input_path, 'problems.json')))
        self.input_path = input_path
        self.output_path = output_path
        self.arch = arch
        self.icon_pretrained = icon_pretrained
        self.transform = transform
        self.task = task
        self.num_patches = num_patches

    def crop_and_padding(self, img, padding=3):
        # Crop the image
        bbox = img.getbbox() # [left, top, right, bottom]
        img = img.crop(bbox)

        # Add padding spaces to the 4 sides of an image
        desired_size = max(img.size) + padding * 2
        if img.size[0] < desired_size or img.size[1] < desired_size:
            delta_w = desired_size - img.size[0]
            delta_h = desired_size - img.size[1]
            padding = (padding, padding, delta_w-padding, delta_h-padding)
            img = ImageOps.expand(img, padding, (255, 255, 255))

        return img

    def extract_patches(self, img, splits):
        patches = []
        w, h = img.size  # width, height
        for n in splits:
            dw, dh = w // n, h // n
            for j in range(n):
                for i in range(n):
                    bbox = dw * i, dh * j, dw * (i + 1), dh * (j + 1)
                    patch = img.crop(bbox)
                    patches.append(patch)
        return patches

    def resize_patches(self, patches):
        resized_patches = []
        for patch in patches:
            patch = self.transform(patch)
            resized_patches.append(patch) # [3,224,224] * num_patches
        patch_input = torch.stack(resized_patches, dim=0) # [num_patches,3,224,224]
        return patch_input

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pid = self.data[index]

        local_split = self.problems[pid]['split']
        assert local_split in ['train', 'val', 'test']

        # process images
        img_file = os.path.join(self.input_path, 'iconqa', local_split, self.task, pid, "image.png")
        print(img_file)
        img = Image.open(img_file)
        img = img.convert('RGB')
        img = self.crop_and_padding(img)

        # obtain patches from the image
        if self.num_patches == 25:
            patches = self.extract_patches(img, [5])
        elif self.num_patches == 36:
            patches = self.extract_patches(img, [6])
        elif self.num_patches == 14:
            patches = self.extract_patches(img, [1, 2, 3])
        elif self.num_patches == 30:
            patches = self.extract_patches(img, [1, 2, 3, 4])
        elif self.num_patches == 79:
            patches = self.extract_patches(img, [1, 2, 3, 4, 7])
        
        # num_patches * [3,224,224] -> [num_patches,3,224,224]
        patch_input = self.resize_patches(patches)

        # convert to Tensor so we can batch it
        img_id = torch.LongTensor([int(pid)])

        return patch_input, img_id


def preprocess_images(input_path, output_path, arch, layer, icon_pretrained, split, task, patch_split):
    """
    Generate image patch embeddings for IconQA images.
    """
    num_patches = patch_split

    # image transformer
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    data_loader = data.DataLoader(ICONQADataset(input_path, output_path,
                                                arch=arch, transform=transform,
                                                icon_pretrained= icon_pretrained,
                                                num_patches=patch_split,
                                                split=split, task=task),
                                  batch_size=1, shuffle=False, num_workers=0)

    # model
    model = image_models.get_image_model(arch, layer, icon_pretrained)
    model = model.eval().to(device)
    print("ConvNet Model:", arch, layer)

    # generate image embeddings
    embeddings = {}
    print("Starting:")
    import numpy as np
    # torch.onnx.export(model, torch.ones((1,3,224,224)), 'embedding_model.onnx', input_names=['patches'], output_names=['embeddings'],
    #                   dynamic_axes={'patches': {0: 'batch'}, 'embeddings': {0: 'batch'}})

    with torch.no_grad():
        custom_model = EmbeddingWithCrop(79, model, 168, 326)
        #res = custom_model(torch.rand((1, 3, 168, 326)))
        print("total image batches:", len(data_loader))
        for img_patches, img_id in tqdm(data_loader, total=len(data_loader)):
            img_patches = img_patches.to(device) # [batch,num_patches,3,224,224]
            model_input = img_patches.view(-1,3,224,224) # [batch*num_patches,3,224,224]
            embedding = model(model_input) # [batch*num_patches,2048,1,1]
            embedding = embedding.squeeze(3).squeeze(2) # [batch*num_patches,2048]

            #
            #Original model. Original img shape + anti-alias
            img = Image.open('../data/iconqa_data/iconqa/test/choose_txt/7/image.png')
            img_shape = img.size
            orig_model = EmbeddingWithCrop(79, model, img_shape[1], img_shape[0])
            res = orig_model(torch.from_numpy(np.array(img)).permute(2, 0, 1)[None, ...])
            error_orig = (res-embedding).numpy().__abs__().mean() #~3e-6
            error_max_orig = (res-embedding).numpy().__abs__().max() #~1e-3

            #New model. Reshape img + anti-alias
            new_model = EmbeddingWithCropv2(79, model, h=168, w=326)
            const_img_size = img.resize((326, 168))
            padded_img = data_loader.dataset.crop_and_padding(const_img_size)
            torch_img = torch.from_numpy(np.array(padded_img)).permute(2, 0, 1)[
                None, ...]
            res_new = new_model(torch_img.float())
            new_error_mean = (res_new - embedding).numpy().__abs__().mean() #~2e-3
            new_error_max = (res_new - embedding).numpy().__abs__().max() #~4e-1

            #sam
            alias_model = EmbeddingWithCropv2AntiAlias(79, model, h=168, w=326)
            const_img_size = img.resize((326, 168))
            padded_img = data_loader.dataset.crop_and_padding(const_img_size)
            torch_img = torch.from_numpy(np.array(padded_img)).permute(2, 0, 1)[
                None, ...]
            alias_res = alias_model(torch_img.float())
            alias_error_mean = (alias_res - embedding).numpy().__abs__().mean() #~2e-3
            alias_error_max = (alias_res - embedding).numpy().__abs__().max() #~4e-1
            #

            embedding = embedding.view(-1,num_patches,2048) # [batch,num_patches,2048]
            assert list(embedding.size())[1:] == [num_patches,2048]
            #print("embedding size", embedding.size()) # pool5: [batch,num_patches,2048]

            for idx in range(img_patches.size(0)):
                assert list(embedding[idx, ...].size()) == [num_patches,2048]
                embeddings[img_id[idx].item()] = embedding[idx, ...].cpu()

    print("Computing image embeddings, Done!")

    # save results
    output_path = os.path.join(output_path, "{}_{}_{}".format(arch, layer, patch_split))
    if icon_pretrained:
        output_path = output_path + "_icon"
    print("final output path:", output_path)
    os.makedirs(output_path, exist_ok=True)

    print("Saving image embedddings:")
    if not icon_pretrained:
        image_embedding_file = os.path.join(output_path,
                                        "iconqa_{0}_{1}_{2}_{3}_{4}.pth".format(split, task, arch, layer, patch_split))
    elif icon_pretrained:
        image_embedding_file = os.path.join(output_path,
                                        "iconqa_{0}_{1}_{2}_{3}_{4}_icon.pth".format(split, task, arch, layer, patch_split))
    print("Saved to {}".format(image_embedding_file))
    torch.save(embeddings, image_embedding_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Standalone utility to preprocess ICONQA images")
    # input and output
    parser.add_argument("--input_path", default="../data/iconqa_data",
                        help="path to the root directory of images")
    parser.add_argument("--output_path", default="../data/iconqa_data",
                        help="path to image features")
    # image model
    parser.add_argument("--arch", default="resnet101")
    parser.add_argument("--layer", default="pool5")
    parser.add_argument("--icon_pretrained", default=False, help='use the icon pretrained model or not')
    parser.add_argument("--patch_split", type=int, default=30, choices=[14,25,30,36,79])
    # tasks and splits
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test", "trainval", "minitrain", "minival", "minitest"])
    parser.add_argument("--task", default="fill_in_blank",
                        choices=["fill_in_blank", "choose_txt", "choose_img"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # GPU
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
    device = torch.device('cpu')
    # manual settings
    tasks = ["fill_in_blank", "choose_txt", "choose_img"]
    splits = ["test", "val", "train"]
    # splits = ["minival", "minitrain", "test", "val", "train"] # "minival", "minitrain" for quick checking
    tasks = ["choose_txt"]
    for task in tasks:
        for split in splits:
            args.task, args.split = task, split
            print("\n----------------- Processing {} for {} -----------------".format(args.task, args.split))

            # preprocess images
            for arg in vars(args):
                print('%s: %s' % (arg, getattr(args, arg)))
            preprocess_images(args.input_path, args.output_path, args.arch, args.layer, 
                              args.icon_pretrained, args.split, args.task, args.patch_split)
