"""
TRAIN GANOMALY

. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function

from options import Options
from lib.data import load_data
from lib.model import Ganomaly
import os
import shutil
##
def train():
    """ Training
    """

    ##
    # ARGUMENTS
    opt = Options().parse()
    ##


    if opt.testOnefilepath!="":
        #만약에 opt.testOnefilepath가 ""가 아니라면 기존 data폴더 data_로 변경 후 data폴더 생성
        os.rename("data","data_")
        shutil.copytree("data_", "data")

        folder2 = 'data/casting/test/normal'
        for filename in os.listdir(folder2):
            if os.path.isfile(folder2+'/'+filename):
                os.remove(folder2+'/'+filename)

        folder3 = 'data/casting/test/abnormal'
        for filename in os.listdir(folder3):
            if os.path.isfile(folder3+'/'+filename):
                os.remove(folder3+'/'+filename)

        #위 생성한 data폴더 안에 test안에 abnormal폴더안에 테스트하고자하는 사진(opt.testOnefilepath) 하나만 넣어놓음 
        shutil.copyfile(opt.testOnefilepath, "data/casting/test/abnormal/abnormalsample.bmp")


    #testResultOneFiles폴더가 없으면 만듬
    if os.path.isdir("testResultOneFiles"):
        c=1
    else:
        os.mkdir("testResultOneFiles")

    # LOAD DATA
    dataloader = load_data(opt)
    ##
    # LOAD MODEL
    model = Ganomaly(opt, dataloader)
    ##
    # TRAIN MODEL



    
    returntrain=""
    returntrain=model.train()
    if returntrain!="":
        if opt.testOnefilepath!="":
            # train()함수 끝난뒤에 data폴더는 삭제 후 data_폴더를 data로 복원
            shutil.rmtree("data")
            os.rename("data_","data")


if __name__ == '__main__':
    train()
