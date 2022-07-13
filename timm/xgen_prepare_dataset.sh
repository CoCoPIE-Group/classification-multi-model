#!/bin/bash


# 进入到数据存放目录并下载数据
data_path="/data/classification-multi-model"
mkdir ${data_path} && cd ${data_path}

wget https://xgen.oss-cn-hongkong.aliyuncs.com/data/classification-multi-model/ILSVRC2012_img_train.tar
wget https://xgen.oss-cn-hongkong.aliyuncs.com/data/classification-multi-model/ILSVRC2012_img_val.tar
wget https://xgen.oss-cn-hongkong.aliyuncs.com/data/classification-multi-model/valprep.sh

# 如果缓存目录不存在则进行数据准备
cd ${data_path}
if "train/" and "val/" 不存在:
    echo "unzip train..."
    mkdir train && cp ILSVRC2012_img_train.tar train/ && cd train
    tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
    echo "step 2"
    find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
    cd ..
    ## 3. Extract the validation data and move images to subfolders:
    echo "unzip val..."
    mkdir val && cp ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
    echo "valprep..."
    cp ../valprep.sh ./ && bash valprep.sh
    echo "extract finished"
