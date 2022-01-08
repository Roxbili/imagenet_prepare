# Prepare ImageNet-1K

To extract ImageNet dataset as following structure:
```
  train/
  ├── n01440764
  │   ├── n01440764_10026.JPEG
  │   ├── n01440764_10027.JPEG
  │   ├── ......
  ├── ......
  val/
  ├── n01440764
  │   ├── ILSVRC2012_val_00000293.JPEG
  │   ├── ILSVRC2012_val_00002138.JPEG
  │   ├── ......
  ├── ......
```

## 1. Download from [http://www.image-net.org/challenges/LSVRC/2012/downloads](http://www.image-net.org/challenges/LSVRC/2012/downloads)
(Needed to login and you can see the download link)

## 2. For training dataset
```shell script
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
mv ILSVRC2012_img_train.tar ../
cd ..
```

## 3. For validating dataset
```shell script
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
mv ILSVRC2012_img_val.tar ../
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
cd ..
```
or
```shell script
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://files-cdn.cnblogs.com/files/luruiyuan/valprep.sh | bash
cd ..
```

## 4. For test dataset
(Don't need to make test directory, the test tar file contains it.)
```shell script
tar -xvf ILSVRC2012_img_test_v10102019.tar
```

## 5. Convert imagenet to .npy format(Optional)
```shell scripy
python imagenet2npy.py --src /your/path/to/imagenet --dst /your/path/to/imageet_npy --workers 4 --resume
```