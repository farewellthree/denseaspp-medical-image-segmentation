import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import scipy.misc
import scipy.ndimage

def save_img(CT_data):
    if CT_data.shape[2]<= 80:
        ind = CT_data.shape[2]
    else :
        ind = 80
    for i in range(ind):
        img = CT_data[:,:,i]
        scipy.misc.imsave('{}.jpg'.format(i), img)

def show_img(im,num):
    plt.imshow(im[:,:,num])
    plt.show()
    
#可视化    
def lum_trans(img):
    liver_win = [-200, 250]
    newimg = (img - liver_win[0]) / (liver_win[1] - liver_win[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    return newimg*255
    
def read_nii(volume_path):
    nii = nib.load(volume_path)

    data = nii.get_data()
    ax_codes = nib.aff2axcodes(nii.affine)

    spacing = list(nii.header['pixdim'][1:4])
    origin = [float(nii.header['qoffset_x']), float(nii.header['qoffset_y']), float(nii.header['qoffset_z'])]
    return data, ax_codes, spacing, origin


def read_nrrd(volume_path):
    mapping = {
        'left-posterior-superior': ('L', 'P', 'S'),
    }
    data, header = nrrd.read(volume_path)
    ax_codes = mapping[header['space']]
    spacing = np.diag(header['space directions'])
    origin = header['space origin']

    return data, ax_codes, spacing, origin

def resample(image,spacing,new_spacing=[1,1,1]):
    spacing = np.array(spacing)
    rate = spacing/new_spacing
    new_shape = np.round(image.shape*rate)
    new_rate = new_shape/image.shape
    new_spacing = spacing/new_rate
    image = scipy.ndimage.interpolation.zoom(image, new_rate, mode='nearest')
    return image,new_spacing

#找到目录下所有的目标文件并将其转换为训练数据
def find_files(path):
    train_path = []
    label_path = []
    
    for root,dirs,files in os.walk(path,topdown=False):
        for name in files:
            if name[-3:]=='nii' and name[0:6]=='Venous':
                if name[-7:-4]=='roi':
                    label_path.append(os.path.join(root,name))
                else:
                    train_path.append(os.path.join(root,name))
                    
    return train_path,label_path

def produce_data(train_path,label_path):
    f = open('1.txt','w')
    num_patient = 0
    num_data = 0
    
    for name1,name2 in train_path,label_path:
        num_patient += 1
        img1,code,spacing,org = read_nii(name1)
        img2 = read_nii(name2)[0]
        
        img1_pro = lum_trans(resample(img1,spacing,[1,1,5])[0])
        img2_pro = resample(img2,spacing,[1,1,5])[0]
        f.write('patient{}:'.format(num_patient)+' '+code[0]+code[1]+code[2]+'\n')
        
        for ind in range(int(img1_pro.shape[2]/3)):
            num_data+=1
            np.save('{}{}.npy'.format('data/data',num_data),img1_pro[:,:,3*ind:3*ind+3])
            np.save('{}{}.npy'.format('label/label',num_data),img2_pro[:,:,3*ind:3*ind+3])
    f.write(str(num_patient)+' '+str(num_data)+'\n')
    
    f.close()
    return  num_patient,num_data

#path1 = 'F:/lib and data/beilun/beilun/00a3bca213a0e1dd6d63519f4cce6997/CT/Venous_tra_5mm.nii'
#path2 = 'F:/lib and data/data/18fe06ca165f66d81d42a3c1c5c687c8/CT/Venous_tra_5mm.nii'
#path3 = 'F:/lib and data/beilun/segmentation-0.nii'
#path4 = 'F:\\lib and data\\data\\0c130c8a042f619baceb0d980ea6a68c\\CT\\Venous_tra_5mm.nii'
#Nii = nib.load(path3)
#img,codes,spacing,origin = read_nii(path4)
#header=Nii.header
#print (img.shape)

#img1 = resample(img,spacing)[0]
#print (img1.shape)
#show_img(img,50)
#save_img(img)
#preimg = lum_trans(img)

#plt.imshow(preimg[:,:,0:3])
#plt.show()



