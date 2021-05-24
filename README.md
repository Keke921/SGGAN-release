## Facial Structure Guided GAN for Identity-preserved Face Image De-occlusion
Yiu-ming Cheung, Mengke Li, Rong Zou
_________________
A [pytorch](http://pytorch.org/) implementation of Facial Structure Guided GAN for Identity-preserved Face Image De-occlusion
Contact: mengkejiajia@qq.com


### Dependency

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.7.1
- [scikit-learn](https://scikit-learn.org/stable/)
- [face_alignment](https://github.com/1adrianb/face-alignment)
- [pytorch_ssim](https://github.com/Po-Hsun-Su/pytorch-ssim/tree/master/pytorch_ssim)


### Dataset

- Occluded CelebA (for training).
- Occluded LFW (for testing) 

The dataset can be download from:
https://drive.google.com/drive/folders/1ISmIMmmpEVFTi8Xl2aiGR8DUBjjOEHMl?usp=sharing

### Before training

- LightCNN[1] is used to extract features. The code of LightCNN is borrowed from:
https://github.com/AlfredXiangWu/LightCNN

We use [LightCNN-29 v2](https://drive.google.com/file/d/1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS/view) provided by the author.

After download the pretrained model, directly put it in the master directory.

### Training 

- To train the **FIRST** stage of SGGAN using the train script simply specify the parameters listed in **train_phase1.py** as a flag or manually change them.
We provide the example of training on CelebA dataset with this repo:
```bash
python train_phase1.py --dataset_name="prepared_image/img_align_celeba_crop" \
		--dataset_train="prepared_image/CelebA_train_delete.txt" \
		--dataset_test="prepared_image/CelebA_test.txt" \
		--img_root_path="E:\data"
```


- To train the **SECOND** stage of SGGAN using the train script simply specify the parameters listed in **train_phase2.py** as a flag or manually change them.
We provide the example of the second stage with this repo:
```bash
python train_phase2.py --dataset_name="prepared_image/img_align_celeba_crop" \
		--dataset_train="prepared_image/CelebA_train_delete.txt" \
		--dataset_test="prepared_image/CelebA_test.txt" \
		--img_root_path="E:\data"\
		--stage1-path="./output/vae.pth"	
```

- Tips:
	- Suppose the dataset is placed in the folder: E:\data\prepared_image/img_align_celeba_crop
	- The dataset structure should be:
	- 
	e.g.
	
	|- Celeba_crop
	
	||- contaminate
	
	|||- img_align_mask
	
	|||-  img_align_mask_sv
	
	|||-  img_align_rand	
	
	|||-  img_align_sun_glass
	
	||-complete
	
	- dataset_train is the train list. We provide the list in our experiment which randomly deletes some image name of female. Because there are more female photos than males in the original dataset.
	- dataset_test is the test list. 
	- For stage2, suppose the trained moedl of the first stage is placed in the forlder: ./output/vae.pth
	
### Evaluation
We provide the example of testing on lfw dataset with this repo:
```
python test-lfw.py --dataset_name="E:\data\prepared_image\processed_lfw_aligned" \
			   --img_list='E:/data/identity_lfw.txt'\
			   --stage1_resume="./output/gen1.pth"\
			   --stage2_resume="./output/gen2.pth"
```

- Tips:
	- util.py provides a rough alignment function named "ImgAlignment" which is based on FAN [2]. If the testing images are seriously misaligned with the training images, the alignment function could be helpful.
	- Suppose the trained models are named "gen1.pth" and "gen2.pth" for stage1 and stage2, respectively. And they are placed in the folder: ./output.
		
## References
- [1] Xiang Wu, Ran He, Zhenan Sun, and Tieniu Tan. 2018. A light cnn for deep face representation with noisy labels. IEEE Transactions on Information Forensics and Security 13, 11 (2018), 2884-2896. https://doi.org/10.1109/TIFS.2018.2833032
- [2] Bulat, Adrian, and Georgios Tzimiropoulos. "How far are we from solving the 2d & 3d face alignment problem?(and a dataset of 230,000 3d facial landmarks)." Proceedings of the IEEE International Conference on Computer Vision. 2017.

## Citation

Please cite the paper if the codes are helpful for you research.

Yiu-ming Cheung, Mengke Li and Rong Zou, ※Facial Structure Guided GAN for Identity-preserved Face Image De-occlusion§, in ICME'21.
