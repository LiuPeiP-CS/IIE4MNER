import os
from utils import build_vocab

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
basic_dir = 'XXX'

class Args:  # we only need to change text_data_dir for twitter2015 and twitter2017
    text_data_dir = os.path.join(basic_dir, 'data/twitter2015')  # 'train', 'dev', 'test'
    img_path = os.path.join(text_data_dir, 'ner_img')  # the address for images 
    obj_num = 5  # the number of objects
    img_feature_rcnn = os.path.join(text_data_dir, 'output/img_features/faster_rcnn.pt')  
    img_object_mrcnn = os.path.join(text_data_dir, 'output/img_features/object_label')
    img_object_label_num = 85
    img_coco_detectron_model = os.path.join(basic_dir, 'data_preprocess/img_prep')
    token_aug_vec_addr = os.path.join(text_data_dir, 'output/aug/tweet_aug.npy')
    # *******************************************************************************************************
    char_input_dim = 128
    max_wordlen = 12
    rcnn_input_dim = 2048

    epoch_num = 75

    resize = 1 # no resize for each image

    #***************************************ablation study**********************************#
    use_char_cnn = 1 # no cnn                                                                      

    aug_before = 1 # token aug　　　　　　　　　　　　　　　　　　　　　　　　　　　
    # aug_before = 2 # no token aug

    aug_gate = 1 

    ffusion_method = 2 # gate + cat

    
    # ffusion_method = 7 # 7 only has obj                                                   (-textimg)
    # ffusion_method = 8 # 8 only has img                                                   (-textobj)

    main_cat = 0  # main for cat

    alpha_seg = 1  

    seg_task = 0 # aux esd
    # seg_task = 1 # no aux esd

    added_output_dir = 'mode_' + str(obj_num)+str(epoch_num)+str(resize)+ str(use_char_cnn)+ str(aug_before) +str(aug_gate)+ str(ffusion_method)+str(main_cat)+str(alpha_seg)+str(seg_task)
    word_dict, char_dict = build_vocab(text_data_dir)  

    aug_dim = 200
    token_aug_num = 4
